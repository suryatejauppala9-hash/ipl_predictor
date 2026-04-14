"""
main.py — IPL Intelligence API v6
New in v6:
  - Momentum graph: over-by-over win probability with event annotations
  - Impact Player designation per team (IPL rule: 1 substitute allowed)
  - Ideal Playing XI endpoint with position-aware selection
  - Varied XI archetypes for simulation (aggressive/balanced/defensive)
  - Cleaner, human-readable code structure
"""
from __future__ import annotations
import math, random, os
from collections import defaultdict
from typing import Any
from functools import lru_cache

import numpy as np
import pandas as pd
import uvicorn
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Auto-generate squad CSVs if missing
# ---------------------------------------------------------------------------
def ensure_squad_csvs():
    if not os.path.exists("player_stats.csv") or not os.path.exists("matchup_stats.csv"):
        print("Generating IPL 2026 squad CSVs from hard-coded squads...")
        import ipl_squads
        ipl_squads.generate_player_stats()
        pdf = pd.read_csv("player_stats.csv")
        ipl_squads.generate_matchup_stats(pdf)

ensure_squad_csvs()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_MATCHES     = 6767
BALL_OUTCOMES = [0, 1, 2, 3, 4, 6]          # 5 excluded intentionally

DEFAULT_BAT_SR       = 148.0   # IPL 2024/25 calibrated
DEFAULT_BOWL_ECON    = 9.0
DEFAULT_DISMISS_PROB = 0.048   # ~1 wkt per 20.8 balls

# Runs per ball per phase (IPL 2024 avg score = 191, 2025 higher)
PHASE_RPB = {"powerplay": 0.150, "middle": 0.148, "death": 0.195}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="IPL Intelligence", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print("Loading data...")
matches    = pd.read_csv("ml_ready_data.csv")
team_stats = pd.read_csv("team_stats.csv", index_col=0)
h2h_df     = pd.read_csv("h2h_stats.csv")
player_df  = pd.read_csv("player_stats.csv")
matchup_df = pd.read_csv("matchup_stats.csv")
print(f"  -> {len(player_df)} players | {len(matchup_df)} matchup records")

try:
    ball_data = pd.read_csv("ball_model_data.csv")
    print(f"  -> ball_model_data.csv: {len(ball_data)} rows")
except FileNotFoundError:
    ball_data = pd.DataFrame()
    print("  ! ball_model_data.csv missing -- using calibrated heuristics")

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
h2h_lookup: dict[tuple[str, str], float] = {}
for _, row in h2h_df.iterrows():
    val = row["team_A_h2h_win_pct"]
    if pd.notna(val):
        pair = tuple(sorted([str(row["team_A"]), str(row["team_B"])]))
        h2h_lookup[pair] = float(val)


def get_h2h(t1: str, t2: str) -> float:
    pair = tuple(sorted([t1, t2]))
    raw  = float(h2h_lookup.get(pair, 0.5))
    if math.isnan(raw):
        raw = 0.5
    return raw if t1 == pair[0] else (1.0 - raw)


player_lookup: dict[str, dict[str, Any]] = {}
team_roster:   dict[str, list[str]]      = defaultdict(list)

for _, row in player_df.iterrows():
    name = str(row.get("player", "")).strip()
    team = str(row.get("team", "")).strip()
    if name and name != "nan":
        d = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
             for k, v in row.to_dict().items()}
        player_lookup[name] = d
        if team and team != "nan":
            team_roster[team].append(name)

matchup_lookup: dict[tuple[str, str], dict] = {}
for _, row in matchup_df.iterrows():
    matchup_lookup[(str(row["batter"]), str(row["bowler"]))] = row.to_dict()

# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------
MATCH_FEATURES = [
    "team1_sr", "team2_sr", "team1_econ", "team2_econ",
    "team1_bat_avg", "team2_bat_avg", "team1_bowl_avg", "team2_bowl_avg",
    "team1_is_home", "team2_is_home", "toss_winner_is_team1", "toss_decision_bat",
    "team1_h2h_win_pct",
]

print("Loading/Training match-winner model...")
if os.path.exists("match_model.pkl"):
    match_model = joblib.load("match_model.pkl")
    print("  -> match model loaded from disk")
else:
    match_model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
    match_model.fit(matches[MATCH_FEATURES], matches["target"])
    joblib.dump(match_model, "match_model.pkl")
    print("  -> match model trained and saved")

BALL_FEATURES = [
    "batter_roll_sr", "batter_cum_runs", "batter_cum_balls",
    "bowler_roll_econ", "bowler_roll_wktr", "bowler_cum_balls",
    "phase_pp", "phase_mid", "phase_death", "innings",
]

ball_model         = None
ball_model_classes = BALL_OUTCOMES

if not ball_data.empty:
    print("Loading/Training ball-outcome model...")
    if os.path.exists("ball_model.pkl") and os.path.exists("ball_classes.pkl"):
        ball_model = joblib.load("ball_model.pkl")
        ball_model_classes = joblib.load("ball_classes.pkl")
        print(f"  -> ball model loaded from disk | classes: {ball_model_classes}")
    else:
        Xb     = ball_data[BALL_FEATURES].fillna(0)
        yb_raw = ball_data["ball_outcome"].astype(int).replace({5: -1})
        mask   = yb_raw != -1
        Xb, yb_raw = Xb[mask], yb_raw[mask]
        le = LabelEncoder()
        yb = le.fit_transform(yb_raw)
        ball_model = XGBClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="mlogloss")
        ball_model.fit(Xb, yb)
        ball_model_classes = list(le.classes_)
        joblib.dump(ball_model, "ball_model.pkl")
        joblib.dump(ball_model_classes, "ball_classes.pkl")
        print(f"  -> ball model trained and saved | classes: {ball_model_classes}")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _sf(v: Any, default: float = 0.0) -> float:
    """Safe float conversion — returns default for NaN/None/errors."""
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _bat_sr(name: str) -> float:
    return _sf(player_lookup.get(name, {}).get("bat_sr"), DEFAULT_BAT_SR)


def _bowl_econ(name: str) -> float:
    return _sf(player_lookup.get(name, {}).get("bowl_econ"), DEFAULT_BOWL_ECON)


def _bowl_wktr(name: str) -> float:
    return _sf(player_lookup.get(name, {}).get("wicket_rate"), DEFAULT_DISMISS_PROB)


def _heuristic_dist(bat_sr: float, bowl_econ: float, phase: str, ball_in_over: int) -> dict[int, float]:
    """Calibrated ball-outcome distribution matching IPL 2024/25 scoring rates."""
    target_rpb = PHASE_RPB.get(phase, 0.148)
    sr_norm    = bat_sr / DEFAULT_BAT_SR
    econ_norm  = DEFAULT_BOWL_ECON / max(bowl_econ, 6.0)
    adj_rpb    = target_rpb * sr_norm * econ_norm

    if phase == "death":
        adj_rpb *= (1.0 + ball_in_over * 0.015)

    p6 = max(0.04, min(0.22, adj_rpb * 0.38))
    p4 = max(0.08, min(0.28, adj_rpb * 0.62))
    p3 = 0.012
    p2 = max(0.05, min(0.14, 0.09))
    p0 = max(0.20, min(0.52, 0.60 - adj_rpb * 1.5))
    p1 = max(0.0, 1.0 - p0 - p2 - p3 - p4 - p6)

    raw   = {0: p0, 1: p1, 2: p2, 3: p3, 4: p4, 6: p6}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _model_dist(bat_sr, bat_runs, bat_balls, bowl_econ, bowl_wktr, bowl_balls, phase, innings) -> dict[int, float]:
    pe = {"powerplay": (1, 0, 0), "middle": (0, 1, 0), "death": (0, 0, 1)}.get(phase, (0, 1, 0))
    row = pd.DataFrame([{
        "batter_roll_sr": bat_sr, "batter_cum_runs": bat_runs, "batter_cum_balls": bat_balls,
        "bowler_roll_econ": bowl_econ, "bowler_roll_wktr": bowl_wktr, "bowler_cum_balls": bowl_balls,
        "phase_pp": pe[0], "phase_mid": pe[1], "phase_death": pe[2], "innings": innings,
    }])
    probs = ball_model.predict_proba(row)[0]
    return {int(cls): float(p) for cls, p in zip(ball_model_classes, probs)}


def _dismiss_prob(batter: str, bowler: str, phase: str) -> float:
    m = matchup_lookup.get((batter, bowler))
    if m and _sf(m.get("m_balls"), 0) >= 10:
        return float(np.clip(_sf(m.get("m_dismiss_prob"), DEFAULT_DISMISS_PROB), 0.01, 0.22))
    wr = _sf(player_lookup.get(bowler, {}).get("wicket_rate"), DEFAULT_DISMISS_PROB)
    if phase == "death":        wr *= 1.10
    elif phase == "powerplay":  wr *= 1.08
    return float(np.clip(wr, 0.01, 0.22))

# ---------------------------------------------------------------------------
# Innings simulation — returns ball_log + per-over state for momentum
# ---------------------------------------------------------------------------
def _simulate_innings(batting_order: list[str], bowling_order: list[str],
                      innings: int = 1, target: int | None = None) -> dict[str, Any]:
    total_runs = total_wkts = 0
    ball_log: list[dict] = []
    over_snapshots: list[dict] = []          # one entry per completed over

    bat_idx    = 0
    on_strike  = batting_order[bat_idx % len(batting_order)]; bat_idx += 1
    non_strike = batting_order[bat_idx % len(batting_order)]; bat_idx += 1

    b_runs  = defaultdict(int);  b_balls  = defaultdict(int)
    bw_runs = defaultdict(int);  bw_balls = defaultdict(int);  bw_wkts = defaultdict(int)
    last_ball = 0
    runs = 0  # initialised here so over_snapshots never references an unbound name

    for ball_num in range(120):
        if total_wkts >= 10:
            break
        if target is not None and total_runs >= target:
            break

        over   = ball_num // 6
        b_in_o = ball_num % 6
        phase  = "powerplay" if over < 6 else ("death" if over >= 15 else "middle")
        bowler = bowling_order[over % len(bowling_order)]
        bs     = _bat_sr(on_strike)
        be     = _bowl_econ(bowler)

        if ball_model is not None:
            dist = _model_dist(bs, b_runs[on_strike], b_balls[on_strike],
                               be, _bowl_wktr(bowler), bw_balls[bowler], phase, innings)
        else:
            dist = _heuristic_dist(bs, be, phase, b_in_o)

        full_dist = {k: dist.get(k, 0.0) for k in BALL_OUTCOMES}
        s = sum(full_dist.values())
        if s > 0:
            full_dist = {k: v / s for k, v in full_dist.items()}

        is_wicket = random.random() < _dismiss_prob(on_strike, bowler, phase)

        if is_wicket:
            ball_log.append({
                "over": over + 1, "ball": b_in_o + 1,
                "batter": on_strike, "bowler": bowler,
                "runs": 0, "wicket": True, "phase": phase,
                "score": total_runs, "wkts": total_wkts + 1,
            })
            total_wkts += 1
            bw_wkts[bowler]  += 1
            b_balls[on_strike] += 1
            bw_balls[bowler]   += 1
            on_strike = batting_order[bat_idx % len(batting_order)]; bat_idx += 1
        else:
            keys  = list(full_dist.keys())
            probs = [full_dist[k] for k in keys]
            runs  = int(np.random.choice(keys, p=probs))
            total_runs       += runs
            b_runs[on_strike]  += runs;  b_balls[on_strike] += 1
            bw_runs[bowler]    += runs;  bw_balls[bowler]   += 1
            ball_log.append({
                "over": over + 1, "ball": b_in_o + 1,
                "batter": on_strike, "bowler": bowler,
                "runs": runs, "wicket": False, "phase": phase,
                "score": total_runs, "wkts": total_wkts,
            })
            if runs % 2 == 1:
                on_strike, non_strike = non_strike, on_strike

        if (ball_num + 1) % 6 == 0:
            on_strike, non_strike = non_strike, on_strike
            # Snapshot at end of each over for momentum graph
            over_snapshots.append({
                "over": over + 1,
                "runs": total_runs,
                "wickets": total_wkts,
                "bowler": bowler,
                "last_ball_event": "wicket" if is_wicket else f"{runs}",
            })

        last_ball = ball_num

    scorecard = {
        "batting": sorted([
            {"batter": b, "runs": b_runs[b], "balls": b_balls[b],
             "sr": round(b_runs[b] / max(b_balls[b], 1) * 100, 1)}
            for b in set(list(b_runs) + list(b_balls))
        ], key=lambda x: -x["runs"]),
        "bowling": sorted([
            {"bowler": bw, "runs": bw_runs[bw], "balls": bw_balls[bw], "wickets": bw_wkts[bw],
             "econ": round(bw_runs[bw] / max(bw_balls[bw] / 6, 0.1), 1)}
            for bw in set(list(bw_runs) + list(bw_balls))
        ], key=lambda x: -x["wickets"]),
    }

    return {
        "total": total_runs, "wickets": total_wkts,
        "balls_bowled": last_ball + 1,
        "ball_log": ball_log,
        "over_snapshots": over_snapshots,
        "scorecard": scorecard,
    }

# ---------------------------------------------------------------------------
# Win probability estimator (for momentum graph)
# ---------------------------------------------------------------------------
def _win_prob_at_state(batting_team: str, bowling_team: str,
                       runs_scored: int, wickets_lost: int,
                       over: int, target: int | None = None,
                       is_second_innings: bool = False) -> float:
    """
    Simple Duckworth-Lewis-inspired win probability estimate.
    Returns probability that batting_team wins from this state.
    """
    if is_second_innings and target is not None:
        balls_remaining = max(0, 120 - over * 6)
        runs_needed     = target - runs_scored
        wickets_rem     = 10 - wickets_lost
        if runs_needed <= 0:
            return 1.0
        if balls_remaining <= 0 or wickets_rem <= 0:
            return 0.0
        # Expected run rate from here vs required
        rrr = runs_needed / (balls_remaining / 6)
        proj_sr_factor = wickets_rem / 10.0      # wickets in hand
        implied_rr = (DEFAULT_BAT_SR / 100 * 6) * proj_sr_factor
        edge = (implied_rr - rrr) / max(rrr, 0.1)
        return float(np.clip(0.5 + edge * 0.25, 0.05, 0.95))
    else:
        # First innings: project final score vs typical par
        balls_bowled = over * 6
        if balls_bowled == 0:
            return 0.5
        current_rr   = runs_scored / (balls_bowled / 6)
        wicket_factor = max(0.4, (10 - wickets_lost) / 10)
        proj_score   = runs_scored + current_rr * wicket_factor * ((120 - balls_bowled) / 6)
        par_score    = 185.0  # IPL 2024/25 calibrated par
        edge = (proj_score - par_score) / par_score
        return float(np.clip(0.5 + edge * 0.6, 0.10, 0.90))


def _build_momentum_graph(inn1: dict, inn2: dict, t1: str, t2: str) -> list[dict]:
    """
    Build over-by-over win probability for the momentum graph.
    Returns list of {over, win_prob_t1, event, score_t1, score_t2, wkts_t1, wkts_t2}
    """
    target = inn1["total"] + 1
    points: list[dict] = []

    # Innings 1 — probability that t1 (batting) will win
    for snap in inn1["over_snapshots"]:
        ov  = snap["over"]
        wp  = _win_prob_at_state(t1, t2, snap["runs"], snap["wickets"],
                                 ov, is_second_innings=False)
        evt = snap["last_ball_event"]
        points.append({
            "over"       : ov,
            "innings"    : 1,
            "team"       : t1,
            "runs"       : snap["runs"],
            "wickets"    : snap["wickets"],
            "win_prob_t1": round(wp, 3),
            "event"      : f"W – {snap['bowler']}" if evt == "wicket" else evt,
            "is_wicket"  : evt == "wicket",
            "bowler"     : snap["bowler"],
        })

    # Innings 2 — probability that t1 wins (decreasing as t2 chases)
    for snap in inn2["over_snapshots"]:
        ov  = snap["over"]
        # t2 is batting, so t2 win prob = 1 - t1 win prob
        t2_wp = _win_prob_at_state(t2, t1, snap["runs"], snap["wickets"],
                                   ov, target=target, is_second_innings=True)
        t1_wp = 1.0 - t2_wp
        evt   = snap["last_ball_event"]
        points.append({
            "over"       : ov + 20,   # offset innings 2 by 20 overs for X-axis
            "innings"    : 2,
            "team"       : t2,
            "runs"       : snap["runs"],
            "wickets"    : snap["wickets"],
            "win_prob_t1": round(t1_wp, 3),
            "event"      : f"W – {snap['bowler']}" if evt == "wicket" else evt,
            "is_wicket"  : evt == "wicket",
            "bowler"     : snap["bowler"],
            "target"     : target,
        })

    return points

# ---------------------------------------------------------------------------
# Impact Player logic (IPL 2023+ rule)
# ---------------------------------------------------------------------------
IMPACT_CRITERIA = {
    # player name → reason why they're the impact player
    # Auto-detected from stats otherwise
}

def _get_impact_player(team: str, role_context: str = "balanced") -> dict[str, Any]:
    """
    Identify the Impact Player strategies for a team.
    Returns targeted substitutions depending on situation.
    """
    roster = team_roster.get(team, [])
    if not roster:
        return {
            "batting_first": {"name": "Unknown", "reason": "No squad data", "stat": "N/A"},
            "bowling_first": {"name": "Unknown", "reason": "No squad data", "stat": "N/A"}
        }

    def bat_score(name: str) -> float:
        p = player_lookup.get(name, {})
        return _sf(p.get("bat_sr"), 0) * max(_sf(p.get("bat_avg"), 0), 10) / 1000.0

    def bowl_score(name: str) -> float:
        p = player_lookup.get(name, {})
        w = _sf(p.get("bowl_wkts"), 0)
        e = max(_sf(p.get("bowl_econ"), 10), 5.5)
        return w * 2.0 / e if w > 0 else 0

    # Inline top-11 selection to avoid recursive call into _ideal_xi
    def _bat_sc(n): p=player_lookup.get(n,{}); return _sf(p.get("bat_sr"),0)*_sf(p.get("bat_avg"),0)
    def _bowl_sc(n): p=player_lookup.get(n,{}); w=_sf(p.get("bowl_wkts"),0); e=_sf(p.get("bowl_econ"),99); return w*2.5-e if w>0 else -99
    xi_names = sorted(roster, key=_bat_sc, reverse=True)[:11]
    bench = [p for p in roster if p not in xi_names]
    if not bench: bench = roster


    best_bowlers = sorted(bench, key=bowl_score, reverse=True)
    bat_first_ip = best_bowlers[0] if best_bowlers else roster[0]
    p_bf = player_lookup.get(bat_first_ip, {})
    bf_wkts = int(_sf(p_bf.get("bowl_wkts"), 0))
    bf_econ = round(_sf(p_bf.get("bowl_econ"), 0), 1)

    best_batters = sorted(bench, key=bat_score, reverse=True)
    bowl_first_ip = best_batters[0] if best_batters else roster[0]
    p_boa = player_lookup.get(bowl_first_ip, {})
    boa_sr = round(_sf(p_boa.get("bat_sr"), 0), 1)
    boa_avg = round(_sf(p_boa.get("bat_avg"), 0), 1)

    return {
        "batting_first": {
            "name": bat_first_ip,
            "reason": "Replaces out batter for 2nd Innings",
            "stat": f"{bf_wkts} wkts @ {bf_econ} econ"
        },
        "bowling_first": {
            "name": bowl_first_ip,
            "reason": "Replaces bowled-out bowler for Chase",
            "stat": f"SR {boa_sr} | Avg {boa_avg}"
        }
    }

# ---------------------------------------------------------------------------
# Ideal Playing XI (position-aware, role-balanced)
# ---------------------------------------------------------------------------
def _ideal_xi(team: str, style: str = "balanced") -> dict[str, Any]:
    """
    Select the optimal 11 from the squad.
    styles: 'balanced' | 'aggressive' | 'bowling'
    Returns batting order with roles, bowlers list, and impact player.
    """
    roster = team_roster.get(team, [])
    if not roster:
        generic = [{"name": f"Player {i}", "role": "Batter", "position": i} for i in range(1, 12)]
        return {"batting": generic, "bowling": [p["name"] for p in generic[-4:]],
                "impact_player": None, "style": style}

    def bat_score(n):
        p = player_lookup.get(n, {})
        return _sf(p.get("bat_sr"), 0) * _sf(p.get("bat_avg"), 0)

    def bowl_score(n):
        p = player_lookup.get(n, {})
        wkts = _sf(p.get("bowl_wkts"), 0)
        econ = _sf(p.get("bowl_econ"), 99)
        if wkts == 0: return -99
        return wkts * 2.5 - econ

    def allround_score(n):
        return bat_score(n) * 0.4 + max(bowl_score(n), 0) * 20

    all_players = sorted(roster, key=bat_score, reverse=True)
    bowl_sorted = sorted(roster, key=bowl_score, reverse=True)

    # Step 1: pick openers (highest bat score)
    openers   = all_players[:2]
    seen      = set(openers)

    # Step 2: pick middle order (positions 3-5) by batting avg
    middle = [n for n in sorted(roster, key=lambda n: _sf(player_lookup.get(n,{}).get("bat_avg"),0), reverse=True)
              if n not in seen][:3]
    seen.update(middle)

    # Step 3: pick a wicketkeeper if not already selected
    keepers = [n for n in roster if "Wicketkeeper" in str(player_lookup.get(n,{}).get("role",""))
               and n not in seen]
    wk = keepers[:1] if keepers else []
    seen.update(wk)

    # Step 4: pick 2 all-rounders for depth
    ars = [n for n in sorted(roster, key=allround_score, reverse=True) if n not in seen][:2]
    seen.update(ars)

    # Step 5: fill bowling slots (4 bowlers minimum)
    bowlers_needed = 11 - len(openers) - len(middle) - len(wk) - len(ars)
    bowlers = [n for n in bowl_sorted if n not in seen][:bowlers_needed]
    seen.update(bowlers)

    # Combine and ensure 11
    xi_names = openers + middle + wk + ars + bowlers
    xi_names = xi_names[:11]
    while len(xi_names) < 11:
        for n in roster:
            if n not in seen:
                xi_names.append(n); seen.add(n); break
        else:
            break

    # Style modifiers
    if style == "aggressive":
        # Swap 1 middle-order batter for a big-hitting finisher
        finishers = sorted([n for n in roster if n not in xi_names],
                           key=lambda n: _sf(player_lookup.get(n,{}).get("bat_sr"),0), reverse=True)
        if finishers and len(xi_names) > 8:
            xi_names[7] = finishers[0]

    elif style == "bowling":
        # Add an extra specialist bowler
        extra_bowlers = [n for n in bowl_sorted if n not in xi_names]
        if extra_bowlers and len(xi_names) > 8:
            xi_names[10] = extra_bowlers[0]

    # Assign positions and roles for display
    positions = ["Opener", "Opener", "No.3", "No.4", "No.5", "Wicketkeeper",
                 "All-Rounder", "All-Rounder", "Bowler", "Bowler", "Bowler"]
    batting_xi = []
    for i, name in enumerate(xi_names[:11]):
        p    = player_lookup.get(name, {})
        role = str(p.get("role", "All-rounder"))
        pos  = positions[i] if i < len(positions) else "Bowler"
        batting_xi.append({
            "name"     : name,
            "position" : pos,
            "role"     : role,
            "bat_sr"   : round(_sf(p.get("bat_sr"), 0), 1),
            "bat_avg"  : round(_sf(p.get("bat_avg"), 0), 1),
            "bowl_econ": round(_sf(p.get("bowl_econ"), 0), 1),
            "bowl_wkts": int(_sf(p.get("bowl_wkts"), 0)),
            "is_bowler": bool(bowl_score(name) > -10 and _sf(p.get("bowl_wkts"), 0) > 3),
        })

    # Best bowlers for bowling order
    bowl_xi = sorted([n["name"] for n in batting_xi if n["is_bowler"]],
                     key=bowl_score, reverse=True)[:5]
    if not bowl_xi:
        bowl_xi = [xi_names[-1]]

    impact = _get_impact_player(team)

    return {
        "batting"       : batting_xi,
        "bowling"       : bowl_xi,
        "impact_player" : impact,
        "style"         : style,
    }


def _best_xi_names(team: str, style: str = "balanced") -> tuple[list[str], list[str]]:
    xi   = _ideal_xi(team, style)
    bat  = [p["name"] for p in xi["batting"]]
    bowl = xi["bowling"]
    return bat, bowl

# ---------------------------------------------------------------------------
# Core simulation loop (6767 matches)
# ---------------------------------------------------------------------------
def _run_sim(bat1: list[str], bowl1: list[str], bat2: list[str], bowl2: list[str],
             n: int, t1: str, t2: str,
             toss_known: bool = True, toss_winner_is_t1: int = 1, toss_bat: int = 1,
             style1: str = "balanced", style2: str = "balanced") -> dict[str, Any]:
    t1s: list[int] = []
    t2s: list[int] = []
    t1w = 0
    tallies      = {str(k): 0 for k in BALL_OUTCOMES}
    tallies["W"] = 0
    total_balls  = 0
    representative: dict | None = None

    for i in range(n):
        if not toss_known:
            sim_toss_t1 = random.randint(0, 1)
            sim_bat     = random.randint(0, 1)
        else:
            sim_toss_t1 = toss_winner_is_t1
            sim_bat     = toss_bat

        t1_bats_first = (sim_toss_t1 == 1 and sim_bat == 1) or (sim_toss_t1 == 0 and sim_bat == 0)

        if t1_bats_first:
            inn1 = _simulate_innings(bat1, bowl2, innings=1)
            inn2 = _simulate_innings(bat2, bowl1, innings=2, target=inn1["total"] + 1)
            t1_score, t2_score = inn1["total"], inn2["total"]
        else:
            tmp1 = _simulate_innings(bat2, bowl1, innings=1)
            tmp2 = _simulate_innings(bat1, bowl2, innings=2, target=tmp1["total"] + 1)
            inn1, inn2   = tmp2, tmp1
            t1_score, t2_score = tmp2["total"], tmp1["total"]

        for e in inn1["ball_log"]:
            k = "W" if e["wicket"] else str(e["runs"])
            tallies[k] = tallies.get(k, 0) + 1
            total_balls += 1

        t1s.append(t1_score)
        t2s.append(t2_score)
        if t1_score > t2_score:
            t1w += 1

        if i == 0:
            # Build momentum graph for representative match
            momentum = _build_momentum_graph(inn1, inn2, t1, t2)
            representative = {
                "innings1"  : {"team": t1, "score": inn1["total"], "wickets": inn1["wickets"],
                               "ball_log": inn1["ball_log"], "scorecard": inn1["scorecard"],
                               "over_snapshots": inn1["over_snapshots"]},
                "innings2"  : {"team": t2, "score": inn2["total"], "wickets": inn2["wickets"],
                               "ball_log": inn2["ball_log"], "scorecard": inn2["scorecard"],
                               "over_snapshots": inn2["over_snapshots"]},
                "momentum"  : momentum,
            }

    total_ev = sum(tallies.values())
    dist     = {k: round(v / max(total_ev, 1), 5) for k, v in tallies.items()}
    a1, a2   = np.array(t1s), np.array(t2s)

    return {
        "n_matches"                : n,
        "team1"                    : t1,
        "team2"                    : t2,
        "toss_known"               : toss_known,
        "win_probability"          : {t1: round(t1w / n * 100, 2), t2: round((n - t1w) / n * 100, 2)},
        "predicted_scores"         : {
            t1: {"mean": round(float(a1.mean()), 1), "std": round(float(a1.std()), 1),
                 "min": int(a1.min()), "max": int(a1.max()),
                 "p10": int(np.percentile(a1, 10)), "p90": int(np.percentile(a1, 90))},
            t2: {"mean": round(float(a2.mean()), 1), "std": round(float(a2.std()), 1),
                 "min": int(a2.min()), "max": int(a2.max()),
                 "p10": int(np.percentile(a2, 10)), "p90": int(np.percentile(a2, 90))},
        },
        "ball_outcome_distribution": dist,
        "outcome_raw_counts"       : tallies,
        "total_balls_simulated"    : total_balls,
        "representative_match"     : representative,
    }

# ---------------------------------------------------------------------------
# Prediction helper (toss-known and toss-unknown modes)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=50)
def _predict_row(t1: str, t2: str, t1h: int, t2h: int, t1_h2h: float,
                 toss_t1: int, toss_bat: int, sr1: float, sr2: float, 
                 econ1: float, econ2: float, bat_avg1: float, bat_avg2: float,
                 bowl_avg1: float, bowl_avg2: float) -> tuple[int, tuple[float, float]]:
    row = pd.DataFrame([{
        "team1_sr": sr1, "team2_sr": sr2, "team1_econ": econ1, "team2_econ": econ2,
        "team1_bat_avg": bat_avg1, "team2_bat_avg": bat_avg2,
        "team1_bowl_avg": bowl_avg1, "team2_bowl_avg": bowl_avg2,
        "team1_is_home": t1h, "team2_is_home": t2h,
        "toss_winner_is_team1": toss_t1, "toss_decision_bat": toss_bat,
        "team1_h2h_win_pct": t1_h2h,
    }])
    pred  = int(match_model.predict(row)[0])
    probs = tuple(float(x) for x in match_model.predict_proba(row)[0])
    return pred, probs

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class MatchRequest(BaseModel):
    team1               : str
    team2               : str
    team1_venue_status  : str  = "neutral"
    toss_known          : bool = True
    toss_winner         : str  = ""
    toss_decision       : str  = "bat"

class SimulateRequest(BaseModel):
    team1               : str
    team2               : str
    toss_known          : bool = True
    toss_winner         : str  = ""
    toss_decision       : str  = "bat"
    style1              : str  = "balanced"   # balanced | aggressive | bowling
    style2              : str  = "balanced"
    n_matches           : int  = N_MATCHES

class CustomSimRequest(BaseModel):
    team1               : str
    team2               : str
    team1_batting       : list[str] | None = None
    team1_bowling       : list[str] | None = None
    team2_batting       : list[str] | None = None
    team2_bowling       : list[str] | None = None
    toss_known          : bool = True
    toss_winner         : str  = ""
    toss_decision       : str  = "bat"
    n_matches           : int  = N_MATCHES

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def home(request: Request):
    teams = sorted(team_stats.index.tolist())
    return templates.TemplateResponse("index.html", {"request": request, "teams": teams})


@app.post("/predict")
def predict(data: MatchRequest):
    try:
        t1, t2 = data.team1, data.team2
        if t1 == t2:
            return JSONResponse({"error": "Teams must be different"}, status_code=400)

        t1_h2h = get_h2h(t1, t2)
        t1h    = 1 if data.team1_venue_status == "home" else 0
        t2h    = 1 if data.team1_venue_status == "away" else 0

        def ts(team, col):
            try:   return _sf(team_stats.loc[team, col])
            except KeyError: return 0.0

        sr1, sr2 = ts(t1, "sr"), ts(t2, "sr")
        econ1, econ2 = ts(t1, "econ"), ts(t2, "econ")
        bat_avg1, bat_avg2 = ts(t1, "bat_avg"), ts(t2, "bat_avg")
        bowl_avg1, bowl_avg2 = ts(t1, "bowl_avg"), ts(t2, "bowl_avg")

        insights = []
        if sr1 > sr2 + 5: insights.append(f"{t1} has superior batting aggression (SR {round(sr1,1)} vs {round(sr2,1)})")
        elif sr2 > sr1 + 5: insights.append(f"{t2} has superior batting aggression (SR {round(sr2,1)} vs {round(sr1,1)})")
        
        if econ1 < econ2 - 0.5: insights.append(f"{t1} bowling is much tighter (Econ {round(econ1,1)} vs {round(econ2,1)})")
        elif econ2 < econ1 - 0.5: insights.append(f"{t2} bowling is much tighter (Econ {round(econ2,1)} vs {round(econ1,1)})")
        
        if t1_h2h > 0.6: insights.append(f"{t1} dominates historically ({round(t1_h2h*100,0)}% win rate)")
        elif t1_h2h < 0.4: insights.append(f"{t2} dominates historically ({round((1-t1_h2h)*100,0)}% win rate)")
        
        if not insights: insights.append("Both teams are quite evenly matched based on recent forms.")

        call_p = lambda tw, td: _predict_row(t1, t2, t1h, t2h, t1_h2h, tw, td, sr1, sr2, econ1, econ2, bat_avg1, bat_avg2, bowl_avg1, bowl_avg2)

        if data.toss_known:
            tw   = 1 if data.toss_winner == t1 else 0
            td   = 1 if data.toss_decision == "bat" else 0
            pred, probs = call_p(tw, td)
            winner   = t1 if pred == 1 else t2
            win_p    = float(probs[1] if pred == 1 else probs[0]) * 100
            toss_scenarios = None
        else:
            scenarios = [
                {"label": f"{t1} wins toss & bats",   "tw": 1, "td": 1},
                {"label": f"{t1} wins toss & bowls",  "tw": 1, "td": 0},
                {"label": f"{t2} wins toss & bats",   "tw": 0, "td": 1},
                {"label": f"{t2} wins toss & bowls",  "tw": 0, "td": 0},
            ]
            avg_t1_prob    = 0.0
            toss_scenarios = []
            for sc in scenarios:
                _, probs_sc = call_p(sc["tw"], sc["td"])
                sc_t1 = float(probs_sc[1]) * 100
                avg_t1_prob += sc_t1 / 4.0
                toss_scenarios.append({"scenario": sc["label"], "team1_win_pct": round(sc_t1, 1)})

            winner = t1 if avg_t1_prob >= 50 else t2
            win_p  = avg_t1_prob if winner == t1 else (100 - avg_t1_prob)
            probs  = [1 - avg_t1_prob / 100, avg_t1_prob / 100]

        impact_t1 = _get_impact_player(t1)
        impact_t2 = _get_impact_player(t2)

        return {
            "winner"          : winner,
            "probability"     : round(win_p, 1),
            "confidence"      : round(float(max(probs)) * 100, 1),
            "toss_known"      : data.toss_known,
            "toss_scenarios"  : toss_scenarios,
            "insights"        : insights,
            "impact_players"  : {t1: impact_t1, t2: impact_t2},
            "team1_stats"     : {"sr": round(sr1, 1), "bat_avg": round(bat_avg1, 1),
                                 "econ": round(econ1, 1), "h2h": round(t1_h2h * 100, 1)},
            "team2_stats"     : {"sr": round(sr2, 1), "bat_avg": round(bat_avg2, 1),
                                 "econ": round(econ2, 1), "h2h": round((1 - t1_h2h) * 100, 1)},
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/simulate")
def simulate(data: SimulateRequest):
    try:
        t1, t2 = data.team1, data.team2
        if t1 == t2:
            return JSONResponse({"error": "Teams must be different"}, status_code=400)
        n    = max(1, min(data.n_matches, N_MATCHES))
        bat1, bowl1 = _best_xi_names(t1, data.style1)
        bat2, bowl2 = _best_xi_names(t2, data.style2)
        tw   = 1 if data.toss_winner == t1 else 0
        td   = 1 if data.toss_decision == "bat" else 0
        result = _run_sim(bat1, bowl1, bat2, bowl2, n, t1, t2,
                          toss_known=data.toss_known, toss_winner_is_t1=tw, toss_bat=td,
                          style1=data.style1, style2=data.style2)
        if ball_model is None:
            result["warning"] = "ball_data_missing"
        result["playing11"]    = {t1: bat1, t2: bat2}
        result["impact_players"] = {t1: _get_impact_player(t1), t2: _get_impact_player(t2)}
        result["ideal_xi"]     = {
            t1: _ideal_xi(t1, data.style1),
            t2: _ideal_xi(t2, data.style2),
        }
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/simulate-custom")
def simulate_custom(data: CustomSimRequest):
    try:
        t1, t2 = data.team1, data.team2
        if t1 == t2:
            return JSONResponse({"error": "Teams must be different"}, status_code=400)
        n     = max(1, min(data.n_matches, N_MATCHES))
        xi1   = _ideal_xi(t1); xi2 = _ideal_xi(t2)
        bat1  = data.team1_batting or [p["name"] for p in xi1["batting"]]
        bowl1 = data.team1_bowling or xi1["bowling"]
        bat2  = data.team2_batting or [p["name"] for p in xi2["batting"]]
        bowl2 = data.team2_bowling or xi2["bowling"]
        tw    = 1 if data.toss_winner == t1 else 0
        td    = 1 if data.toss_decision == "bat" else 0
        result = _run_sim(bat1, bowl1, bat2, bowl2, n, t1, t2,
                          toss_known=data.toss_known, toss_winner_is_t1=tw, toss_bat=td)
        if ball_model is None:
            result["warning"] = "ball_data_missing"
        result["playing11"]     = {t1: bat1, t2: bat2}
        result["impact_players"] = {t1: _get_impact_player(t1), t2: _get_impact_player(t2)}
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/ideal-xi/{team}")
def ideal_xi(team: str, style: str = "balanced"):
    try:
        xi = _ideal_xi(team, style)
        return {"team": team, "style": style, **xi}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/playing11/{team}")
def playing11(team: str):
    try:
        xi = _ideal_xi(team, "balanced")
        return {"team": team, "playing11": xi["batting"], "bowlers": xi["bowling"],
                "impact_player": xi["impact_player"]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/squad/{team}")
def get_squad(team: str):
    try:
        roster = team_roster.get(team, [])
        if not roster:
            return JSONResponse({"error": f"No squad found for '{team}'"}, status_code=404)
        players = []
        for name in roster:
            p = player_lookup.get(name, {})
            players.append({
                "name"        : name,
                "role"        : str(p.get("role", "Unknown")),
                "bat_runs"    : int(_sf(p.get("bat_runs"), 0)),
                "bat_balls"   : int(_sf(p.get("bat_balls"), 0)),
                "bat_sr"      : round(_sf(p.get("bat_sr"), 0), 1),
                "bat_avg"     : round(_sf(p.get("bat_avg"), 0), 1),
                "boundary_pct": round(_sf(p.get("boundary_pct"), 0), 1),
                "dot_pct"     : round(_sf(p.get("dot_pct"), 0), 1),
                "bowl_wkts"   : int(_sf(p.get("bowl_wkts"), 0)),
                "bowl_balls"  : int(_sf(p.get("bowl_balls"), 0)),
                "bowl_econ"   : round(_sf(p.get("bowl_econ"), 0), 1),
                "bowl_avg"    : round(_sf(p.get("bowl_avg"), 0), 1),
            })
        players.sort(key=lambda x: -x["bat_runs"])
        return {"team": team, "squad": players}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/player-stats")
def player_stats(team: str | None = None):
    try:
        df = player_df.copy()
        if team and "team" in df.columns:
            df = df[df["team"] == team]
        cols = [c for c in ["player", "team", "role", "bat_runs", "bat_balls", "bat_sr",
                             "bat_avg", "boundary_pct", "dot_pct", "bowl_wkts", "bowl_balls",
                             "bowl_econ", "bowl_avg"] if c in df.columns]
        return JSONResponse(df[cols].fillna(0).round(2).to_dict(orient="records"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)