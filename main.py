"""
main.py — IPL Intelligence API v7
===================================
Changes from v6:
  - Loads calibrated, tuned models from match_model.joblib / ball_model.joblib
    (produced by data_cleaning.py pipeline) with graceful fallback to in-process training
  - Leakage-free cumulative ball features — shift inside group
  - Impact Player logic, Ideal XI, Squad API all preserved
  - All simulation / prediction endpoints unchanged for frontend compatibility
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ── Auto-generate squad CSVs if missing ──────────────────────────────────────
def _ensure_squad_csvs() -> None:
    if not os.path.exists("player_stats.csv") or not os.path.exists("matchup_stats.csv"):
        print("Generating IPL 2026 squad CSVs from hard-coded squads...")
        import ipl_squads
        ipl_squads.generate_player_stats()
        pdf = pd.read_csv("player_stats.csv")
        ipl_squads.generate_matchup_stats(pdf)

_ensure_squad_csvs()

# ── Constants ─────────────────────────────────────────────────────────────────
N_MATCHES     = 500
BALL_OUTCOMES = [0, 1, 2, 3, 4, 6]

# ── Prediction cache (cleared between simulate requests) ──────────────────────
_PRED_CACHE: dict[tuple, dict] = {}

DEFAULT_BAT_SR       = 148.0
DEFAULT_BOWL_ECON    = 9.0
DEFAULT_DISMISS_PROB = 0.048

PHASE_RPB = {"powerplay": 0.150, "middle": 0.148, "death": 0.195}

MATCH_FEATURES = [
    "team1_sr", "team2_sr",
    "team1_econ", "team2_econ",
    "team1_bat_avg", "team2_bat_avg",
    "team1_bowl_avg", "team2_bowl_avg",
    "team1_is_home", "team2_is_home",
    "toss_winner_is_team1", "toss_decision_bat",
    "team1_h2h_win_pct",
    "team1_last5_wins", "team2_last5_wins",
    "team1_venue_winrate", "team2_venue_winrate",
    "team1_chase_winrate", "team2_chase_winrate",
    "team1_phase_pp_rr", "team2_phase_pp_rr",
    "team1_phase_mid_rr", "team2_phase_mid_rr",
    "team1_phase_death_rr", "team2_phase_death_rr",
    "team1_death_econ", "team2_death_econ",
    # upgrade features
    "sr_diff", "econ_diff", "form_diff",
    "team1_encoded", "team2_encoded",
    "team1_matchup_strength", "team2_matchup_strength",
    "team1_depth", "team2_depth",
]

BALL_FEATURES = [
    "batter_cum_runs", "batter_cum_balls", "batter_roll_sr",
    "striker_sr_vs_pace", "striker_sr_vs_spin",
    "dot_ball_pressure",
    "bowler_cum_balls", "bowler_cum_wkts",
    "bowler_roll_econ", "bowler_roll_wktr",
    "bowler_economy_this_phase",
    "batter_vs_bowler_matchup_sr",
    "innings", "over", "ball_in_over",
    "phase_pp", "phase_mid", "phase_death",
    "wickets_in_hand",
    "runs_in_innings",
    "balls_remaining",
    "required_run_rate",
    # upgrade features
    "last12_runs", "last12_wickets", "pressure_index",
]

TRAIN_SEASONS_END = 2022
VAL_SEASON        = 2023

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="IPL Intelligence", version="7.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
matches_df = pd.read_csv("ml_ready_data.csv")

# 🔧 SAFETY PATCH: ensure 'season' exists
if "season" not in matches_df.columns:
    print("  ! 'season' column missing — reconstructing...")

    if "date" in matches_df.columns:
        matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")
        matches_df["season"] = matches_df["date"].dt.year
    else:
        # fallback: assume mid-era IPL
        matches_df["season"] = 2020

    matches_df["season"] = matches_df["season"].fillna(2020).astype(int)
team_stats = pd.read_csv("team_stats.csv", index_col=0)
h2h_df     = pd.read_csv("h2h_stats.csv")
player_df  = pd.read_csv("player_stats.csv")

# Filter to 2026 squad players only
import ipl_squads as _squads

_squad_team_map = {
    name: team
    for team, players in _squads.SQUADS.items()
    for name, *_ in players
}
_squad_names = set(_squad_team_map.keys())

# Keep only 2026 squad members, fix their team assignment
player_df = player_df[player_df["player"].isin(_squad_names)].copy()
player_df["team"] = player_df["player"].map(_squad_team_map)
matchup_df = pd.read_csv("matchup_stats.csv")
print(f"  → {len(player_df)} players | {len(matchup_df)} matchup records")

try:
    ball_data = pd.read_csv("ball_model_data.csv")
    print(f"  → ball_model_data.csv: {len(ball_data):,} rows")
except FileNotFoundError:
    ball_data = pd.DataFrame()
    print("  ! ball_model_data.csv missing — will use calibrated heuristics")

# ── Lookup tables ─────────────────────────────────────────────────────────────
h2h_lookup: dict[tuple[str, str], float] = {}
for _, row in h2h_df.iterrows():
    val = row["team_A_h2h_win_pct"]
    if pd.notna(val):
        pair = tuple(sorted([str(row["team_A"]), str(row["team_B"])]))
        h2h_lookup[pair] = float(val)


def get_h2h(t1: str, t2: str) -> float:
    pair = tuple(sorted([t1, t2]))
    raw  = float(h2h_lookup.get(pair, 0.5))
    raw  = 0.5 if math.isnan(raw) else raw
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

# ── Load team label encoder ───────────────────────────────────────────────────
try:
    _team_label_enc = joblib.load("team_encoder.joblib")
    print("  → team_encoder.joblib loaded")
except FileNotFoundError:
    _team_label_enc = None
    print("  ! team_encoder.joblib missing — encoding defaults to 0")

def _team_enc(name: str) -> int:
    if _team_label_enc is None:
        return 0
    try:
        return int(_team_label_enc.transform([name])[0])
    except Exception:
        return 0

# ── Load or train models ───────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _train_match_model_inline(matches: pd.DataFrame, feature_cols: list[str]):
    """Fallback: train without tuning or calibration."""
    avail = [f for f in feature_cols if f in matches.columns]

    # 🛡️ SAFETY: handle missing season
    if "season" not in matches.columns:
        print("  ! No season column — using full data for training")
        train = matches
        val = pd.DataFrame()
    else:
        train = matches[matches["season"] <= TRAIN_SEASONS_END]
        val   = matches[matches["season"] == VAL_SEASON]

        if len(train) == 0:
            print("  ! Empty train split — fallback to full dataset")
            train = matches

    X_tr, y_tr = train[avail].fillna(0).values, train["target"].values
    base = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         eval_metric="logloss", verbosity=0)
    if len(val) > 10:
        X_vl, y_vl = val[avail].fillna(0).values, val["target"].values
        base.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                 early_stopping_rounds=50, verbose=False)
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_vl, y_vl)
    else:
        base.fit(X_tr, y_tr, verbose=False)
        cal = base   # no calibration if val is tiny
    return cal, avail


def _train_ball_model_inline(ball: pd.DataFrame, feature_cols: list[str]):
    avail  = [f for f in feature_cols if f in ball.columns]
    le     = LabelEncoder()
    y_raw  = ball["ball_outcome"].astype(int).replace(5, 4)
    y_enc  = le.fit_transform(y_raw)
    train  = ball[ball["season"] <= TRAIN_SEASONS_END] if "season" in ball.columns else ball
    val    = ball[ball["season"] == VAL_SEASON] if "season" in ball.columns else pd.DataFrame()
    if len(train) == 0:
        train = ball

    X_tr, y_tr = train[avail].fillna(0).values, y_enc[:len(train)]
    base = XGBClassifier(n_estimators=200, learning_rate=0.08, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         eval_metric="mlogloss", verbosity=0)
    if len(val) > 100:
        X_vl = val[avail].fillna(0).values
        y_vl = y_enc[len(train):len(train)+len(val)]
        base.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                 early_stopping_rounds=50, verbose=False)
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_vl, y_vl)
    else:
        base.fit(X_tr, y_tr, verbose=False)
        cal = base
    classes = list(le.classes_)
    return cal, classes, avail


# Priority: load pre-built joblib → fallback to inline training
print("Loading / training match-winner model...")
_match_features_used = [f for f in MATCH_FEATURES if f in matches_df.columns]

if Path("match_model.joblib").exists():
    _bundle      = joblib.load("match_model.joblib")
    match_model  = _bundle["model"]
    _match_features_used = _bundle.get("features", _match_features_used)
    print("  → loaded match_model.joblib")
else:
    print("  ! match_model.joblib not found — training inline (run data_cleaning.py for tuned model)")
    match_model, _match_features_used = _train_match_model_inline(matches_df, MATCH_FEATURES)

print("Loading / training ball-outcome model...")
ball_model         = None
ball_model_classes = BALL_OUTCOMES
_ball_features_used = [f for f in BALL_FEATURES if not ball_data.empty and f in ball_data.columns]

if Path("ball_model.joblib").exists():
    _bbundle         = joblib.load("ball_model.joblib")
    ball_model       = _bbundle["model"]
    ball_model_classes = _bbundle.get("classes", BALL_OUTCOMES)
    _ball_features_used = _bbundle.get("features", _ball_features_used)
    print("  → loaded ball_model.joblib")
elif not ball_data.empty:
    print("  ! ball_model.joblib not found — training inline")
    ball_model, ball_model_classes, _ball_features_used = _train_ball_model_inline(
        ball_data, BALL_FEATURES)
else:
    print("  ! No ball data — using calibrated heuristics")

# ── Utilities ─────────────────────────────────────────────────────────────────

def _bat_sr(name: str) -> float:
    return _safe_float(player_lookup.get(name, {}).get("bat_sr"), DEFAULT_BAT_SR)


def _bowl_econ(name: str) -> float:
    return _safe_float(player_lookup.get(name, {}).get("bowl_econ"), DEFAULT_BOWL_ECON)


def _bowl_wktr(name: str) -> float:
    return _safe_float(player_lookup.get(name, {}).get("wicket_rate"), DEFAULT_DISMISS_PROB)


def _heuristic_dist(bat_sr: float, bowl_econ: float,
                    phase: str, ball_in_over: int) -> dict[int, float]:
    """Calibrated ball-outcome distribution (IPL 2024/25 scoring)."""
    target_rpb = PHASE_RPB.get(phase, 0.148)
    sr_norm    = bat_sr / DEFAULT_BAT_SR
    econ_norm  = DEFAULT_BOWL_ECON / max(bowl_econ, 6.0)
    adj_rpb    = target_rpb * sr_norm * econ_norm

    if phase == "death":
        adj_rpb *= (1.0 + ball_in_over * 0.015)

    p6  = max(0.04, min(0.22, adj_rpb * 0.38))
    p4  = max(0.08, min(0.28, adj_rpb * 0.62))
    p3  = 0.012
    p2  = max(0.05, min(0.14, 0.09))
    p0  = max(0.20, min(0.52, 0.60 - adj_rpb * 1.5))
    p1  = max(0.0, 1.0 - p0 - p2 - p3 - p4 - p6)

    raw   = {0: p0, 1: p1, 2: p2, 3: p3, 4: p4, 6: p6}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _model_dist(bat_sr: float, bat_runs: float, bat_balls: float,
                bat_sr_pace: float, bat_sr_spin: float,
                dot_pressure: int,
                bowl_econ: float, bowl_wktr: float, bowl_balls: float,
                bowl_wkts: float, bowl_phase_econ: float,
                matchup_sr: float,
                innings: int, over: int, ball_in_over: int,
                phase: str,
                wickets_in_hand: int, runs_in_innings: int,
                balls_remaining: int, req_rr: float,
                last12_runs: float = 0.0, last12_wickets: float = 0.0,
                pressure_index: float = 1.0) -> dict[int, float]:
    # ── Cache key (coarse-grained to maximise hit rate) ───────────────────────
    cache_key = (
        innings, over, ball_in_over, phase, wickets_in_hand,
        runs_in_innings, balls_remaining,
        round(req_rr, 1), round(bat_sr, 0), round(bowl_econ, 1),
    )
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key]

    pe = {"powerplay": (1, 0, 0), "middle": (0, 1, 0), "death": (0, 0, 1)}.get(phase, (0, 1, 0))
    row_dict = {
        "batter_cum_runs":              bat_runs,
        "batter_cum_balls":             bat_balls,
        "batter_roll_sr":               bat_sr,
        "striker_sr_vs_pace":           bat_sr_pace,
        "striker_sr_vs_spin":           bat_sr_spin,
        "dot_ball_pressure":            dot_pressure,
        "bowler_cum_balls":             bowl_balls,
        "bowler_cum_wkts":              bowl_wkts,
        "bowler_roll_econ":             bowl_econ,
        "bowler_roll_wktr":             bowl_wktr,
        "bowler_economy_this_phase":    bowl_phase_econ,
        "batter_vs_bowler_matchup_sr":  matchup_sr,
        "innings":                      innings,
        "over":                         over,
        "ball_in_over":                 ball_in_over,
        "phase_pp":   pe[0],
        "phase_mid":  pe[1],
        "phase_death":pe[2],
        "wickets_in_hand":   wickets_in_hand,
        "runs_in_innings":   runs_in_innings,
        "balls_remaining":   balls_remaining,
        "required_run_rate": req_rr,
        "last12_runs":       last12_runs,
        "last12_wickets":    last12_wickets,
        "pressure_index":    pressure_index,
    }
    # Build numpy array directly — avoids per-ball DataFrame overhead
    avail = {f: row_dict.get(f, 0.0) for f in _ball_features_used}
    row   = np.array([[avail[f] for f in _ball_features_used]], dtype=np.float32)
    probs = ball_model.predict_proba(row)[0]
    result = {int(cls): float(p) for cls, p in zip(ball_model_classes, probs)}
    _PRED_CACHE[cache_key] = result
    return result


def _dismiss_prob(batter: str, bowler: str, phase: str) -> float:
    m = matchup_lookup.get((batter, bowler))
    if m and _safe_float(m.get("m_balls"), 0) >= 10:
        return float(np.clip(_safe_float(m.get("m_dismiss_prob"), DEFAULT_DISMISS_PROB), 0.01, 0.22))
    wr = _safe_float(player_lookup.get(bowler, {}).get("wicket_rate"), DEFAULT_DISMISS_PROB)
    if phase == "death":       wr *= 1.10
    elif phase == "powerplay": wr *= 1.08
    return float(np.clip(wr, 0.01, 0.22))


def _team_matchup_strength(team: str, opponent: str) -> float:
    """Mean matchup SR for team's batters vs opponent's bowlers."""
    batters  = team_roster.get(team, [])
    bowlers  = team_roster.get(opponent, [])
    srs = []
    for batter in batters:
        for bowler in bowlers:
            m = matchup_lookup.get((batter, bowler))
            if m and _safe_float(m.get("m_balls"), 0) >= 6:
                sr = _safe_float(m.get("m_sr"), 0)
                if sr > 0:
                    srs.append(sr)
    return float(np.mean(srs)) if srs else 130.0


def _batting_depth(team: str) -> float:
    """Mean bat SR of players in positions 5–8 (0-based indices 4–7)."""
    roster = team_roster.get(team, [])
    if not roster:
        return 130.0
    sorted_players = sorted(roster,
                            key=lambda n: _safe_float(player_lookup.get(n, {}).get("bat_avg"), 0),
                            reverse=True)
    depth_players = sorted_players[4:8]
    if not depth_players:
        return 130.0
    srs = [_safe_float(player_lookup.get(n, {}).get("bat_sr"), 130.0) for n in depth_players]
    return float(np.mean(srs))

# ── Innings simulation ────────────────────────────────────────────────────────

def _simulate_innings(batting_order: list[str], bowling_order: list[str],
                      innings: int = 1, target: int | None = None) -> dict[str, Any]:
    from collections import deque
    total_runs = total_wkts = 0
    ball_log: list[dict]      = []
    over_snapshots: list[dict] = []

    bat_idx    = 0
    on_strike  = batting_order[bat_idx % len(batting_order)]; bat_idx += 1
    non_strike = batting_order[bat_idx % len(batting_order)]; bat_idx += 1

    # Per-ball cumulative state (leakage-free: updated AFTER each delivery)
    b_runs:  defaultdict[str, int] = defaultdict(int)
    b_balls: defaultdict[str, int] = defaultdict(int)
    b_dots:  defaultdict[str, int] = defaultdict(int)   # consecutive dots
    bw_runs: defaultdict[str, int] = defaultdict(int)
    bw_balls:defaultdict[str, int] = defaultdict(int)
    bw_wkts: defaultdict[str, int] = defaultdict(int)
    # Phase-specific bowl runs/balls
    bw_phase_runs: defaultdict[tuple, int]  = defaultdict(int)
    bw_phase_balls:defaultdict[tuple, int]  = defaultdict(int)

    # Rolling last-12-ball window: (runs, is_wicket)
    last12_deque: deque = deque(maxlen=12)

    last_ball = 0
    last_dist: dict | None = None

    for ball_num in range(120):
        if total_wkts >= 10:
            break
        if target is not None and total_runs >= target:
            break

        over_num   = ball_num // 6
        b_in_over  = ball_num % 6
        phase      = ("powerplay" if over_num < 6
                      else ("death" if over_num >= 15 else "middle"))
        bowler     = bowling_order[over_num % len(bowling_order)]

        # ── Pre-delivery state (no leakage — values from PREVIOUS balls) ──
        bs     = _bat_sr(on_strike)
        be     = _bowl_econ(bowler)
        bwtr   = _bowl_wktr(bowler)
        p      = player_lookup.get(on_strike, {})
        # bowl type for SR-vs-pace/spin
        bowl_type = str(player_lookup.get(bowler, {}).get("bowling_type", "pace")).lower()
        bs_pace = _safe_float(p.get("bat_sr"), bs)  # fallback to overall
        bs_spin = _safe_float(p.get("bat_sr"), bs)

        matchup = matchup_lookup.get((on_strike, bowler), {})
        matchup_sr = _safe_float(matchup.get("m_sr"), bs)

        phase_key = (bowler, phase)
        phase_econ = (_safe_float(bw_phase_runs[phase_key] /
                                  max(bw_phase_balls[phase_key] / 6, 0.01), be)
                      if bw_phase_balls[phase_key] > 0 else be)

        req_rr = 0.0
        if innings == 2 and target is not None:
            balls_left = max(120 - ball_num, 1)
            req_rr = max(0.0, (target - total_runs) / (balls_left / 6))

        # ── Compute last12 / pressure for model ────────────────────────────
        current_rr = total_runs / max(ball_num / 6, 0.01) if ball_num > 0 else 0.0
        l12_runs   = sum(r for r, _ in last12_deque)
        l12_wkts   = sum(1 for _, w in last12_deque if w)
        p_index    = (req_rr / max(current_rr, 0.5)) if innings == 2 and current_rr > 0 else 1.0

        # ── Ball outcome distribution ──────────────────────────────────────
        if ball_model is not None and _ball_features_used and (ball_num % 2 == 0 or last_dist is None):
            last_dist = _model_dist(
                bat_sr=b_runs[on_strike] / max(b_balls[on_strike], 1) * 100 if b_balls[on_strike] else bs,
                bat_runs=b_runs[on_strike], bat_balls=b_balls[on_strike],
                bat_sr_pace=bs_pace, bat_sr_spin=bs_spin,
                dot_pressure=b_dots[on_strike],
                bowl_econ=(bw_runs[bowler] / max(bw_balls[bowler] / 6, 0.01)
                           if bw_balls[bowler] > 0 else be),
                bowl_wktr=(bw_wkts[bowler] / max(bw_balls[bowler], 1)
                           if bw_balls[bowler] > 0 else bwtr),
                bowl_balls=bw_balls[bowler], bowl_wkts=bw_wkts[bowler],
                bowl_phase_econ=phase_econ,
                matchup_sr=matchup_sr,
                innings=innings, over=over_num + 1, ball_in_over=b_in_over,
                phase=phase,
                wickets_in_hand=10 - total_wkts,
                runs_in_innings=total_runs,
                balls_remaining=120 - ball_num,
                req_rr=req_rr,
                last12_runs=l12_runs, last12_wickets=l12_wkts,
                pressure_index=p_index,
            )
        dist = last_dist if last_dist is not None else _heuristic_dist(bs, be, phase, b_in_over)

        full_dist = {k: dist.get(k, 0.0) for k in BALL_OUTCOMES}
        s = sum(full_dist.values())
        if s > 0:
            full_dist = {k: v / s for k, v in full_dist.items()}

        # ── 2A: Pressure + momentum behaviour modifiers ────────────────────
        # 1. Pressure modifier (2nd innings chase)
        if innings == 2 and req_rr > 10:
            full_dist[4] = full_dist.get(4, 0) * 1.25
            full_dist[6] = full_dist.get(6, 0) * 1.30
            full_dist[0] = full_dist.get(0, 0) * 0.85

        # 2. Wickets-in-hand conservatism
        if (10 - total_wkts) <= 3:
            full_dist[6] = full_dist.get(6, 0) * 0.70
            full_dist[4] = full_dist.get(4, 0) * 0.80
            full_dist[0] = full_dist.get(0, 0) * 1.15

        # 3. Death over boundary boost
        if over_num >= 15:
            full_dist[4] = full_dist.get(4, 0) * 1.15
            full_dist[6] = full_dist.get(6, 0) * 1.20

        # 4. Momentum (last 12 balls)
        last12_run_sum = sum(r for r, _ in last12_deque) if last12_deque else 0
        if last12_run_sum > 20:
            full_dist[4] = full_dist.get(4, 0) * 1.12
            full_dist[6] = full_dist.get(6, 0) * 1.15

        # Renormalise after all modifiers
        s = sum(full_dist.values())
        if s > 0:
            full_dist = {k: v / s for k, v in full_dist.items()}

        is_wicket = random.random() < _dismiss_prob(on_strike, bowler, phase)

        # ── Record outcome — update state AFTER delivery ───────────────────
        if is_wicket:
            ball_log.append({
                "over": over_num + 1, "ball": b_in_over + 1,
                "batter": on_strike, "bowler": bowler,
                "runs": 0, "wicket": True, "phase": phase,
                "score": total_runs, "wkts": total_wkts + 1,
            })
            total_wkts += 1
            bw_wkts[bowler]    += 1
            b_balls[on_strike] += 1
            bw_balls[bowler]   += 1
            bw_phase_balls[phase_key] += 1
            b_dots[on_strike] = 0   # reset dot streak on dismissal
            on_strike = batting_order[bat_idx % len(batting_order)]; bat_idx += 1
            last12_deque.append((0, True))
        else:
            keys  = list(full_dist.keys())
            probs = [full_dist[k] for k in keys]
            runs  = int(np.random.choice(keys, p=probs))
            total_runs         += runs
            b_runs[on_strike]  += runs
            b_balls[on_strike] += 1
            bw_runs[bowler]    += runs
            bw_balls[bowler]   += 1
            bw_phase_runs[phase_key]  += runs
            bw_phase_balls[phase_key] += 1

            if runs == 0:
                b_dots[on_strike] += 1
            else:
                b_dots[on_strike] = 0

            last12_deque.append((runs, False))

            ball_log.append({
                "over": over_num + 1, "ball": b_in_over + 1,
                "batter": on_strike, "bowler": bowler,
                "runs": runs, "wicket": False, "phase": phase,
                "score": total_runs, "wkts": total_wkts,
            })
            if runs % 2 == 1:
                on_strike, non_strike = non_strike, on_strike

        if (ball_num + 1) % 6 == 0:
            on_strike, non_strike = non_strike, on_strike
            over_snapshots.append({
                "over":             over_num + 1,
                "runs":             total_runs,
                "wickets":          total_wkts,
                "bowler":           bowler,
                "last_ball_event":  "wicket" if is_wicket else str(runs if not is_wicket else 0),
            })

        last_ball = ball_num

    scorecard = {
        "batting": sorted([
            {"batter": b, "runs": b_runs[b], "balls": b_balls[b],
             "sr": round(b_runs[b] / max(b_balls[b], 1) * 100, 1)}
            for b in set(list(b_runs) + list(b_balls))
        ], key=lambda x: -x["runs"]),
        "bowling": sorted([
            {"bowler": bw, "runs": bw_runs[bw], "balls": bw_balls[bw],
             "wickets": bw_wkts[bw],
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

# ── Win probability estimator ─────────────────────────────────────────────────

def _win_prob_at_state(batting_team: str, bowling_team: str,
                       runs_scored: int, wickets_lost: int,
                       over: int, target: int | None = None,
                       is_second_innings: bool = False) -> float:
    if is_second_innings and target is not None:
        balls_remaining = max(0, 120 - over * 6)
        runs_needed     = target - runs_scored
        wickets_rem     = 10 - wickets_lost
        if runs_needed <= 0:    return 1.0
        if balls_remaining <= 0 or wickets_rem <= 0: return 0.0
        rrr          = runs_needed / (balls_remaining / 6)
        proj_factor  = wickets_rem / 10.0
        implied_rr   = (DEFAULT_BAT_SR / 100 * 6) * proj_factor
        edge = (implied_rr - rrr) / max(rrr, 0.1)
        return float(np.clip(0.5 + edge * 0.25, 0.05, 0.95))
    else:
        balls_bowled = over * 6
        if balls_bowled == 0: return 0.5
        current_rr   = runs_scored / (balls_bowled / 6)
        wkt_factor   = max(0.4, (10 - wickets_lost) / 10)
        proj_score   = runs_scored + current_rr * wkt_factor * ((120 - balls_bowled) / 6)
        par_score    = 185.0
        edge = (proj_score - par_score) / par_score
        return float(np.clip(0.5 + edge * 0.6, 0.10, 0.90))


def _build_momentum_graph(inn1: dict, inn2: dict, t1: str, t2: str) -> list[dict]:
    target = inn1["total"] + 1
    points: list[dict] = []

    for snap in inn1["over_snapshots"]:
        ov  = snap["over"]
        wp  = _win_prob_at_state(t1, t2, snap["runs"], snap["wickets"],
                                 ov, is_second_innings=False)
        evt = snap["last_ball_event"]
        points.append({
            "over": ov, "innings": 1, "team": t1,
            "runs": snap["runs"], "wickets": snap["wickets"],
            "win_prob_t1": round(wp, 3),
            "event":      f"W – {snap['bowler']}" if evt == "wicket" else evt,
            "is_wicket":  evt == "wicket",
            "bowler":     snap["bowler"],
        })

    for snap in inn2["over_snapshots"]:
        ov    = snap["over"]
        t2_wp = _win_prob_at_state(t2, t1, snap["runs"], snap["wickets"],
                                   ov, target=target, is_second_innings=True)
        t1_wp = 1.0 - t2_wp
        evt   = snap["last_ball_event"]
        points.append({
            "over": ov + 20, "innings": 2, "team": t2,
            "runs": snap["runs"], "wickets": snap["wickets"],
            "win_prob_t1": round(t1_wp, 3),
            "event":       f"W – {snap['bowler']}" if evt == "wicket" else evt,
            "is_wicket":   evt == "wicket",
            "bowler":      snap["bowler"],
            "target":      target,
        })
    return points

# ── Impact player ─────────────────────────────────────────────────────────────

def _get_impact_player(team: str, role_context: str = "balanced") -> dict[str, str]:
    roster = team_roster.get(team, [])
    if not roster:
        return {"name": "Unknown", "reason": "No squad data", "stat": "N/A"}

    def impact_score(name: str) -> float:
        p    = player_lookup.get(name, {})
        bat  = _safe_float(p.get("bat_sr"), 0) * _safe_float(p.get("bat_avg"), 0) / 1000
        bowl = (_safe_float(p.get("bowl_wkts"), 0) * 2.0
                / max(_safe_float(p.get("bowl_econ"), 10), 6))
        return bat + bowl

    ranked     = sorted(roster, key=impact_score, reverse=True)
    candidates = ranked[6:12] if len(ranked) > 8 else ranked[-3:]
    best       = max(candidates, key=impact_score)
    p    = player_lookup.get(best, {})
    role = str(p.get("role", "All-rounder"))
    sr   = _safe_float(p.get("bat_sr"), 0)
    wkts = int(_safe_float(p.get("bowl_wkts"), 0))
    econ = _safe_float(p.get("bowl_econ"), 0)

    if role in ("Batter",) and sr > 155:
        reason = f"Explosive finisher · SR {sr}"
    elif wkts > 40 and econ < 8.5:
        reason = f"Death-over specialist · {wkts} wkts @ {econ} econ"
    elif role == "All-rounder":
        reason = f"All-round impact · SR {sr} + {wkts} wkts"
    else:
        reason = f"Match-winning potential · {role}"

    return {"name": best, "reason": reason, "stat": f"SR {sr}" if sr > 0 else f"{wkts} wkts"}

# ── Ideal XI ──────────────────────────────────────────────────────────────────

def _ideal_xi(team: str, style: str = "balanced") -> dict[str, Any]:
    roster = team_roster.get(team, [])
    if not roster:
        generic = [{"name": f"Player {i}", "role": "Batter", "position": i}
                   for i in range(1, 12)]
        return {"batting": generic, "bowling": [p["name"] for p in generic[-4:]],
                "impact_player": None, "style": style}

    def bat_score(n):
        p = player_lookup.get(n, {})
        return _safe_float(p.get("bat_sr"), 0) * _safe_float(p.get("bat_avg"), 0)

    def bowl_score(n):
        p    = player_lookup.get(n, {})
        wkts = _safe_float(p.get("bowl_wkts"), 0)
        econ = _safe_float(p.get("bowl_econ"), 99)
        if wkts == 0: return -99
        return wkts * 2.5 - econ

    def allround_score(n):
        return bat_score(n) * 0.4 + max(bowl_score(n), 0) * 20

    all_players  = sorted(roster, key=bat_score, reverse=True)
    bowl_sorted  = sorted(roster, key=bowl_score, reverse=True)

    openers = all_players[:2]
    seen    = set(openers)

    middle = [n for n in sorted(roster,
              key=lambda n: _safe_float(player_lookup.get(n, {}).get("bat_avg"), 0), reverse=True)
              if n not in seen][:3]
    seen.update(middle)

    keepers = [n for n in roster
               if "Wicketkeeper" in str(player_lookup.get(n, {}).get("role", ""))
               and n not in seen]
    wk = keepers[:1] if keepers else []
    seen.update(wk)

    ars = [n for n in sorted(roster, key=allround_score, reverse=True)
           if n not in seen][:2]
    seen.update(ars)

    bowlers_needed = 11 - len(openers) - len(middle) - len(wk) - len(ars)
    bowlers = [n for n in bowl_sorted if n not in seen][:bowlers_needed]
    seen.update(bowlers)

    xi_names = openers + middle + wk + ars + bowlers
    xi_names = xi_names[:11]
    while len(xi_names) < 11:
        added = False
        for n in roster:
            if n not in seen:
                xi_names.append(n); seen.add(n); added = True; break
        if not added: break

    if style == "aggressive":
        finishers = sorted([n for n in roster if n not in xi_names],
                           key=lambda n: _safe_float(player_lookup.get(n, {}).get("bat_sr"), 0),
                           reverse=True)
        if finishers and len(xi_names) > 8:
            xi_names[7] = finishers[0]
    elif style == "bowling":
        extra_bowlers = [n for n in bowl_sorted if n not in xi_names]
        if extra_bowlers and len(xi_names) > 8:
            xi_names[10] = extra_bowlers[0]

    positions = ["Opener", "Opener", "No.3", "No.4", "No.5", "Wicketkeeper",
                 "All-Rounder", "All-Rounder", "Bowler", "Bowler", "Bowler"]
    batting_xi = []
    for i, name in enumerate(xi_names[:11]):
        p    = player_lookup.get(name, {})
        role = str(p.get("role", "All-rounder"))
        pos  = positions[i] if i < len(positions) else "Bowler"
        batting_xi.append({
            "name":      name,
            "position":  pos,
            "role":      role,
            "bat_sr":    round(_safe_float(p.get("bat_sr"), 0), 1),
            "bat_avg":   round(_safe_float(p.get("bat_avg"), 0), 1),
            "bowl_econ": round(_safe_float(p.get("bowl_econ"), 0), 1),
            "bowl_wkts": int(_safe_float(p.get("bowl_wkts"), 0)),
            "is_bowler": bool(bowl_score(name) > -10 and
                              _safe_float(p.get("bowl_wkts"), 0) > 3),
        })

    bowl_xi = sorted([n["name"] for n in batting_xi if n["is_bowler"]],
                     key=bowl_score, reverse=True)[:5]
    if not bowl_xi:
        bowl_xi = [xi_names[-1]]

    return {"batting": batting_xi, "bowling": bowl_xi,
            "impact_player": _get_impact_player(team), "style": style}


def _best_xi_names(team: str, style: str = "balanced") -> tuple[list[str], list[str]]:
    xi   = _ideal_xi(team, style)
    bat  = [p["name"] for p in xi["batting"]]
    bowl = xi["bowling"]
    return bat, bowl

# ── Core simulation ───────────────────────────────────────────────────────────

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
            inn1, inn2     = tmp2, tmp1
            t1_score, t2_score = tmp2["total"], tmp1["total"]

        for e in inn1["ball_log"]:
            k = "W" if e["wicket"] else str(e["runs"])
            tallies[k] = tallies.get(k, 0) + 1
            total_balls += 1

        t1s.append(t1_score); t2s.append(t2_score)
        if t1_score > t2_score:
            t1w += 1

        if i == 0:
            momentum = _build_momentum_graph(inn1, inn2, t1, t2)
            representative = {
                "innings1": {"team": t1, "score": inn1["total"], "wickets": inn1["wickets"],
                             "ball_log": inn1["ball_log"], "scorecard": inn1["scorecard"],
                             "over_snapshots": inn1["over_snapshots"]},
                "innings2": {"team": t2, "score": inn2["total"], "wickets": inn2["wickets"],
                             "ball_log": inn2["ball_log"], "scorecard": inn2["scorecard"],
                             "over_snapshots": inn2["over_snapshots"]},
                "momentum": momentum,
            }

    total_ev = sum(tallies.values())
    dist     = {k: round(v / max(total_ev, 1), 5) for k, v in tallies.items()}
    a1, a2   = np.array(t1s), np.array(t2s)

    # ── 2B: Simulation insights ───────────────────────────────────────────────
    margin_arr   = a1 - a2
    close_matches = int(np.sum(np.abs(margin_arr) <= 15))
    close_match_pct = round(close_matches / n * 100, 1)

    # ── 2C: Confidence level ──────────────────────────────────────────────────
    gap = abs(t1w / n - 0.5) * 2
    confidence_level = "High" if gap > 0.30 else ("Medium" if gap > 0.15 else "Low")

    return {
        "n_matches"                : n,
        "team1"                    : t1,
        "team2"                    : t2,
        "toss_known"               : toss_known,
        "win_probability"          : {t1: round(t1w / n * 100, 2),
                                      t2: round((n - t1w) / n * 100, 2)},
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
        "confidence_level"         : confidence_level,
        "simulation_insights"      : {
            "avg_score_t1":     round(float(a1.mean()), 1),
            "avg_score_t2":     round(float(a2.mean()), 1),
            "winning_range_t1": [int(np.percentile(a1, 25)), int(np.percentile(a1, 75))],
            "winning_range_t2": [int(np.percentile(a2, 25)), int(np.percentile(a2, 75))],
            "close_match_pct":  close_match_pct,
        },
    }

def _build_explanation(t1: str, t2: str, t1_stats: dict, t2_stats: dict) -> list[dict]:
    """Build ordered list of factors that explain the prediction result."""
    factors = []
    sr_diff = t1_stats["sr"] - t2_stats["sr"]
    if abs(sr_diff) > 2:
        factors.append({
            "label":    f"{'Stronger' if sr_diff > 0 else 'Weaker'} batting SR",
            "team":     t1 if sr_diff > 0 else t2,
            "impact":   round(min(abs(sr_diff) * 0.4, 8), 1),
            "positive": sr_diff > 0,
        })
    econ_diff = t2_stats["econ"] - t1_stats["econ"]
    if abs(econ_diff) > 0.3:
        factors.append({
            "label":    "Better bowling economy",
            "team":     t1 if econ_diff > 0 else t2,
            "impact":   round(min(abs(econ_diff) * 2, 6), 1),
            "positive": econ_diff > 0,
        })
    h2h_edge = t1_stats["h2h"] - 50
    if abs(h2h_edge) > 5:
        factors.append({
            "label":    "H2H head-to-head record",
            "team":     t1 if h2h_edge > 0 else t2,
            "impact":   round(min(abs(h2h_edge) * 0.15, 5), 1),
            "positive": h2h_edge > 0,
        })
    return factors[:4]


def _get_key_matchups(batters_t1: list, bowlers_t2: list,
                      batters_t2: list, bowlers_t1: list, n: int = 3) -> list[dict]:
    """Top N most decisive batter-vs-bowler matchups across both teams."""
    results = []
    for batter in batters_t1[:6]:
        for bowler in bowlers_t2[:3]:
            m = matchup_lookup.get((batter, bowler))
            if m and _safe_float(m.get("m_balls"), 0) >= 6:
                sr    = _safe_float(m.get("m_sr"), 100)
                balls = int(_safe_float(m.get("m_balls"), 0))
                results.append({
                    "batter":     batter,
                    "bowler":     bowler,
                    "sr":         round(sr, 1),
                    "balls":      balls,
                    "dismissals": int(balls * _safe_float(m.get("m_dismiss_prob"), 0)),
                    "advantage":  ("batter" if sr > 130 else ("bowler" if sr < 100 else "neutral")),
                })
    for batter in batters_t2[:6]:
        for bowler in bowlers_t1[:3]:
            m = matchup_lookup.get((batter, bowler))
            if m and _safe_float(m.get("m_balls"), 0) >= 6:
                sr    = _safe_float(m.get("m_sr"), 100)
                balls = int(_safe_float(m.get("m_balls"), 0))
                results.append({
                    "batter":     batter,
                    "bowler":     bowler,
                    "sr":         round(sr, 1),
                    "balls":      balls,
                    "dismissals": int(balls * _safe_float(m.get("m_dismiss_prob"), 0)),
                    "advantage":  ("batter" if sr > 130 else ("bowler" if sr < 100 else "neutral")),
                })
    results.sort(key=lambda x: abs(x["sr"] - 115), reverse=True)
    return results[: n * 2]


# ── Prediction helper ─────────────────────────────────────────────────────────

def _make_feature_row(t1: str, t2: str, t1h: int, t2h: int, t1_h2h: float,
                      toss_t1: int, toss_bat: int) -> dict[str, float]:
    def ts(team: str, col: str) -> float:
        try:   return _safe_float(team_stats.loc[team, col])
        except KeyError: return 0.0

    return {
        "team1_sr":               ts(t1, "sr"),
        "team2_sr":               ts(t2, "sr"),
        "team1_econ":             ts(t1, "econ"),
        "team2_econ":             ts(t2, "econ"),
        "team1_bat_avg":          ts(t1, "bat_avg"),
        "team2_bat_avg":          ts(t2, "bat_avg"),
        "team1_bowl_avg":         ts(t1, "bowl_avg") or 28.0,
        "team2_bowl_avg":         ts(t2, "bowl_avg") or 28.0,
        "team1_is_home":          t1h,
        "team2_is_home":          t2h,
        "toss_winner_is_team1":   toss_t1,
        "toss_decision_bat":      toss_bat,
        "team1_h2h_win_pct":      t1_h2h,
        # New features: use neutral defaults if not in team_stats
        "team1_last5_wins":       ts(t1, "last5_wins") or 0.5,
        "team2_last5_wins":       ts(t2, "last5_wins") or 0.5,
        "team1_venue_winrate":    0.5,
        "team2_venue_winrate":    0.5,
        "team1_chase_winrate":    0.5,
        "team2_chase_winrate":    0.5,
        "team1_phase_pp_rr":      ts(t1, "phase_pp_rr") or 8.5,
        "team2_phase_pp_rr":      ts(t2, "phase_pp_rr") or 8.5,
        "team1_phase_mid_rr":     ts(t1, "phase_mid_rr") or 8.0,
        "team2_phase_mid_rr":     ts(t2, "phase_mid_rr") or 8.0,
        "team1_phase_death_rr":   ts(t1, "phase_death_rr") or 10.5,
        "team2_phase_death_rr":   ts(t2, "phase_death_rr") or 10.5,
        "team1_death_econ":       ts(t1, "death_econ") or 9.5,
        "team2_death_econ":       ts(t2, "death_econ") or 9.5,
        # Upgrade features
        "sr_diff":                ts(t1, "sr") - ts(t2, "sr"),
        "econ_diff":              ts(t2, "econ") - ts(t1, "econ"),
        "form_diff":              (ts(t1, "last5_wins") or 0.5) - (ts(t2, "last5_wins") or 0.5),
        "team1_encoded":          _team_enc(t1),
        "team2_encoded":          _team_enc(t2),
        "team1_matchup_strength": _team_matchup_strength(t1, t2),
        "team2_matchup_strength": _team_matchup_strength(t2, t1),
        "team1_depth":            _batting_depth(t1),
        "team2_depth":            _batting_depth(t2),
    }


def _predict_row(t1: str, t2: str, t1h: int, t2h: int, t1_h2h: float,
                 toss_t1: int, toss_bat: int) -> tuple[int, list[float]]:
    row_dict = _make_feature_row(t1, t2, t1h, t2h, t1_h2h, toss_t1, toss_bat)
    avail    = [f for f in _match_features_used if f in row_dict]
    row      = pd.DataFrame([{f: row_dict[f] for f in avail}])[avail]
    pred     = int(match_model.predict(row)[0])
    probs    = list(match_model.predict_proba(row)[0])
    return pred, probs

# ── Schemas ───────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    team1              : str
    team2              : str
    team1_venue_status : str  = "neutral"
    toss_known         : bool = True
    toss_winner        : str  = ""
    toss_decision      : str  = "bat"

class SimulateRequest(BaseModel):
    team1         : str
    team2         : str
    toss_known    : bool = True
    toss_winner   : str  = ""
    toss_decision : str  = "bat"
    style1        : str  = "balanced"
    style2        : str  = "balanced"
    n_matches     : int  = N_MATCHES

class CustomSimRequest(BaseModel):
    team1           : str
    team2           : str
    team1_batting   : list[str]
    team1_bowling   : list[str]
    team2_batting   : list[str]
    team2_bowling   : list[str]
    toss_known      : bool = True
    toss_winner     : str  = ""
    toss_decision   : str  = "bat"
    n_matches       : int  = N_MATCHES

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    teams = sorted(team_stats.index.tolist())
    response = templates.TemplateResponse(
        "index.html", {"request": request, "teams": teams})
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


async def _predict_impl(data: MatchRequest):
    t1, t2 = data.team1, data.team2
    if t1 == t2:
        return JSONResponse({"error": "Teams must be different"}, status_code=400)

    t1_h2h = get_h2h(t1, t2)
    t1h    = 1 if data.team1_venue_status == "home"  else 0
    t2h    = 1 if data.team1_venue_status == "away"  else 0

    def ts(team: str, col: str) -> float:
        try:   return _safe_float(team_stats.loc[team, col])
        except KeyError: return 0.0

    if data.toss_known:
        tw   = 1 if data.toss_winner == t1 else 0
        td   = 1 if data.toss_decision == "bat" else 0
        pred, probs = _predict_row(t1, t2, t1h, t2h, t1_h2h, tw, td)
        winner          = t1 if pred == 1 else t2
        win_p           = float(probs[1] if pred == 1 else probs[0]) * 100
        toss_scenarios  = None
    else:
        scenarios = [
            {"label": f"{t1} wins toss & bats",  "tw": 1, "td": 1},
            {"label": f"{t1} wins toss & bowls", "tw": 1, "td": 0},
            {"label": f"{t2} wins toss & bats",  "tw": 0, "td": 1},
            {"label": f"{t2} wins toss & bowls", "tw": 0, "td": 0},
        ]
        avg_t1_prob    = 0.0
        toss_scenarios = []
        for sc in scenarios:
            _, probs_sc = _predict_row(t1, t2, t1h, t2h, t1_h2h, sc["tw"], sc["td"])
            sc_t1 = float(probs_sc[1]) * 100
            avg_t1_prob += sc_t1 / 4.0
            toss_scenarios.append({
                "scenario": sc["label"], "label": sc["label"],
                "team1_win_pct": round(sc_t1, 1)})

        winner = t1 if avg_t1_prob >= 50 else t2
        win_p  = avg_t1_prob if winner == t1 else (100 - avg_t1_prob)
        probs  = [1 - avg_t1_prob / 100, avg_t1_prob / 100]

    t1_stats_dict = {
        "sr":      round(ts(t1, "sr"), 1),
        "bat_avg": round(ts(t1, "bat_avg"), 1),
        "econ":    round(ts(t1, "econ"), 1),
        "bowl_avg":round(ts(t1, "bowl_avg") or 28.0, 1),
        "h2h":     round(t1_h2h * 100, 1),
    }
    t2_stats_dict = {
        "sr":      round(ts(t2, "sr"), 1),
        "bat_avg": round(ts(t2, "bat_avg"), 1),
        "econ":    round(ts(t2, "econ"), 1),
        "bowl_avg":round(ts(t2, "bowl_avg") or 28.0, 1),
        "h2h":     round((1 - t1_h2h) * 100, 1),
    }

    return {
        "winner"         : winner,
        "probability"    : round(win_p, 1),
        "confidence"     : round(float(max(probs)) * 100, 1),
        "toss_known"     : data.toss_known,
        "toss_scenarios" : toss_scenarios,
        "impact_players" : {t1: _get_impact_player(t1), t2: _get_impact_player(t2)},
        "team1_stats"    : t1_stats_dict,
        "team2_stats"    : t2_stats_dict,
        "explanation"    : _build_explanation(t1, t2, t1_stats_dict, t2_stats_dict),
    }


@app.post("/predict")
async def predict(data: MatchRequest):
    return await _predict_impl(data)


@app.get("/predict")
async def predict_get(
    team1: str,
    team2: str,
    venue_status: str = "neutral",
    toss_known: bool = True,
    toss_winner: str = "",
    toss_decision: str = "bat",
):
    data = MatchRequest(
        team1=team1, team2=team2,
        team1_venue_status=venue_status,
        toss_known=toss_known,
        toss_winner=toss_winner,
        toss_decision=toss_decision,
    )
    return await _predict_impl(data)


@app.post("/simulate")
async def simulate(data: SimulateRequest):
    global _PRED_CACHE
    _PRED_CACHE = {}
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
    result["playing11"]     = {t1: bat1, t2: bat2}
    result["impact_players"] = {t1: _get_impact_player(t1), t2: _get_impact_player(t2)}
    result["ideal_xi"]      = {t1: _ideal_xi(t1, data.style1), t2: _ideal_xi(t2, data.style2)}
    result["key_matchups"]  = _get_key_matchups(bat1, bowl2, bat2, bowl1)
    return result


@app.post("/simulate-custom")
async def simulate_custom(data: CustomSimRequest):
    global _PRED_CACHE
    _PRED_CACHE = {}
    t1, t2 = data.team1, data.team2
    if t1 == t2:
        return JSONResponse({"error": "Teams must be different"}, status_code=400)
    n    = max(1, min(data.n_matches, N_MATCHES))
    xi1  = _ideal_xi(t1); xi2 = _ideal_xi(t2)
    bat1  = data.team1_batting or [p["name"] for p in xi1["batting"]]
    bowl1 = data.team1_bowling or xi1["bowling"]
    bat2  = data.team2_batting or [p["name"] for p in xi2["batting"]]
    bowl2 = data.team2_bowling or xi2["bowling"]
    tw    = 1 if data.toss_winner == t1 else 0
    td    = 1 if data.toss_decision == "bat" else 0
    result = _run_sim(bat1, bowl1, bat2, bowl2, n, t1, t2,
                      toss_known=data.toss_known, toss_winner_is_t1=tw, toss_bat=td)
    result["playing11"]     = {t1: bat1, t2: bat2}
    result["impact_players"] = {t1: _get_impact_player(t1), t2: _get_impact_player(t2)}
    return result


@app.get("/ideal-xi/{team}")
async def ideal_xi(team: str, style: str = "balanced"):
    xi = _ideal_xi(team, style)
    return {"team": team, "style": style, **xi}


@app.get("/playing11/{team}")
async def playing11(team: str):
    xi = _ideal_xi(team, "balanced")
    return {"team": team, "playing11": xi["batting"], "bowlers": xi["bowling"],
            "impact_player": xi["impact_player"]}


@app.get("/squad/{team}")
async def get_squad(team: str):
    roster = team_roster.get(team, [])
    if not roster:
        return JSONResponse({"error": f"No squad found for '{team}'"}, status_code=404)
    players = []
    for name in roster:
        p = player_lookup.get(name, {})
        players.append({
            "name":         name,
            "role":         str(p.get("role", "Unknown")),
            "bowling_type": str(p.get("bowling_type", "none")),
            "bat_runs":     int(_safe_float(p.get("bat_runs"), 0)),
            "bat_balls":    int(_safe_float(p.get("bat_balls"), 0)),
            "bat_sr":       round(_safe_float(p.get("bat_sr"), 0), 1),
            "bat_avg":      round(_safe_float(p.get("bat_avg"), 0), 1),
            "boundary_pct": round(_safe_float(p.get("boundary_pct"), 0), 1),
            "dot_pct":      round(_safe_float(p.get("dot_pct"), 0), 1),
            "bowl_wkts":    int(_safe_float(p.get("bowl_wkts"), 0)),
            "bowl_balls":   int(_safe_float(p.get("bowl_balls"), 0)),
            "bowl_econ":    round(_safe_float(p.get("bowl_econ"), 0), 1),
            "bowl_avg":     round(_safe_float(p.get("bowl_avg"), 0), 1),
        })
    players.sort(key=lambda x: -x["bat_runs"])
    return {"team": team, "squad": players}


@app.get("/player-stats")
async def player_stats_endpoint(team: str | None = None):
    df = player_df.copy()
    if team and "team" in df.columns:
        df = df[df["team"] == team]
    cols = [c for c in ["player", "team", "role", "bowling_type", "bat_runs", "bat_balls",
                         "bat_sr", "bat_avg", "boundary_pct", "dot_pct",
                         "bowl_wkts", "bowl_balls", "bowl_econ", "bowl_avg"]
            if c in df.columns]
    return JSONResponse(df[cols].fillna(0).round(2).to_dict(orient="records"))


@app.get("/model-info")
async def model_info():
    """Return model metadata and last evaluation metrics."""
    info: dict = {
        "match_model_features": _match_features_used,
        "ball_model_features":  _ball_features_used,
        "ball_model_classes":   ball_model_classes,
    }
    if Path("results.json").exists():
        with open("results.json") as f:
            info["results"] = json.load(f)
    return info


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)