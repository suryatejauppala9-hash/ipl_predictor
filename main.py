"""
main.py — IPL Match Predictor API
FastAPI backend with:
  • POST /predict       — XGBoost win probability (match-level)
  • POST /simulate      — Monte Carlo match simulation (6767 runs) with
                          full ball-outcome probability distributions
  • GET  /player-stats  — Player batting + bowling stats
  • GET  /              — Jinja2 HTML dashboard
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from xgboost import XGBClassifier

# ── Constants ────────────────────────────────────────────────────────────────

N_MONTE_CARLO = 6767          # Monte Carlo simulations per predict call
BALL_OUTCOMES  = [0, 1, 2, 3, 4, 6]  # every possible runs-off-bat outcome

# Default league-average stats used when a player is unknown
DEFAULT_BAT_SR   = 130.0
DEFAULT_BAT_AVG  = 25.0
DEFAULT_BOWL_ECON = 8.0
DEFAULT_DISMISS_PROB = 0.05   # ~1 wicket per 20 balls

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="IPL Match Predictor", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load Data ─────────────────────────────────────────────────────────────────

print("Loading pre-processed data...")
matches    = pd.read_csv('ml_ready_data.csv')
team_stats = pd.read_csv('team_stats.csv', index_col=0)
h2h_df     = pd.read_csv('h2h_stats.csv')

# Player stats (may not exist yet — graceful fallback)
try:
    player_df  = pd.read_csv('player_stats.csv')
    matchup_df = pd.read_csv('matchup_stats.csv')
    print("  → Player stats loaded")
except FileNotFoundError:
    player_df  = pd.DataFrame()
    matchup_df = pd.DataFrame()
    print("  ⚠ player_stats.csv / matchup_stats.csv not found — simulation will use team averages")

# Ball-level model data (for XGBoost ball predictor)
try:
    ball_data = pd.read_csv('ball_model_data.csv')
    print("  → Ball model data loaded")
except FileNotFoundError:
    ball_data = pd.DataFrame()
    print("  ⚠ ball_model_data.csv not found — simulation will use heuristics")

# ── Build Lookup Structures ───────────────────────────────────────────────────

# Head-to-head lookup: {(teamA, teamB)} → team_A win %   (A = lexicographic min)
h2h_lookup: dict[tuple[str, str], float] = {}
for _, row in h2h_df.iterrows():
    pair = tuple(sorted([row['team_A'], row['team_B']]))
    h2h_lookup[pair] = float(row['team_A_h2h_win_pct'])

# Player lookup by name → stats dict
player_lookup: dict[str, dict[str, float]] = {}
if not player_df.empty:
    for _, row in player_df.iterrows():
        name = str(row.get('player', ''))
        if name:
            player_lookup[name] = row.to_dict()

# Matchup lookup: {(batter, bowler)} → {m_sr, m_dismiss_prob, ...}
matchup_lookup: dict[tuple[str, str], dict[str, float]] = {}
if not matchup_df.empty:
    for _, row in matchup_df.iterrows():
        key = (str(row['batter']), str(row['bowler']))
        matchup_lookup[key] = row.to_dict()

# ── Match-Level XGBoost Model ─────────────────────────────────────────────────

MATCH_FEATURES = [
    'team1_sr', 'team2_sr', 'team1_econ', 'team2_econ',
    'team1_bat_avg', 'team2_bat_avg', 'team1_bowl_avg', 'team2_bowl_avg',
    'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat',
    'team1_h2h_win_pct'
]

print("Training XGBoost match-winner model...")
match_model = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False,
    eval_metric='logloss'
)
match_model.fit(matches[MATCH_FEATURES], matches['target'])
print("  → Match model ready")

# ── Ball-Level XGBoost Model ──────────────────────────────────────────────────

BALL_FEATURES = [
    'batter_roll_sr', 'batter_cum_runs', 'batter_cum_balls',
    'bowler_roll_econ', 'bowler_roll_wktr', 'bowler_cum_balls',
    'phase_pp', 'phase_mid', 'phase_death',
    'innings',
]

ball_model = None
if not ball_data.empty:
    print("Training XGBoost ball-outcome model...")
    from sklearn.preprocessing import LabelEncoder
    Xb = ball_data[BALL_FEATURES].fillna(0)
    yb = ball_data['ball_outcome'].astype(int)
    ball_model = XGBClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    ball_model.fit(Xb, yb)
    ball_model_classes = list(ball_model.classes_)   # sorted list of run values seen
    print(f"  → Ball model ready | classes: {ball_model_classes}")
else:
    ball_model_classes = BALL_OUTCOMES

# ── Simulation Helpers ────────────────────────────────────────────────────────

def _get_batter_sr(name: str) -> float:
    p = player_lookup.get(name, {})
    return float(p.get('bat_sr', DEFAULT_BAT_SR))

def _get_bowler_econ(name: str) -> float:
    p = player_lookup.get(name, {})
    return float(p.get('bowl_econ', DEFAULT_BOWL_ECON))

def _heuristic_ball_dist(bat_sr: float, bowl_econ: float, phase: str) -> dict[int, float]:
    """
    Compute probability distribution over BALL_OUTCOMES = [0,1,2,3,4,6]
    from first principles using batter SR and bowler economy rate.
    """
    # Expected runs per ball
    rpb = bowl_econ / 6.0
    rpb = max(0.5, min(rpb, 2.5))    # clip to sensible range

    # Phase multiplier
    phase_mult = {'powerplay': 1.12, 'middle': 1.0, 'death': 1.18}.get(phase, 1.0)
    rpb *= phase_mult

    # Boundary rates driven by batting SR
    sr_norm = bat_sr / 130.0         # 1.0 = league average
    p6  = max(0.02, min(0.15, 0.06 * sr_norm))
    p4  = max(0.05, min(0.25, 0.12 * sr_norm))
    p3  = 0.01
    p2  = max(0.04, min(0.12, 0.06))
    p0  = max(0.25, min(0.65, 0.55 / sr_norm))
    p1  = max(0.0, 1.0 - p0 - p2 - p3 - p4 - p6)

    raw = {0: p0, 1: p1, 2: p2, 3: p3, 4: p4, 6: p6}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _model_ball_dist(
    bat_sr: float, bat_cum_runs: float, bat_cum_balls: float,
    bowl_econ: float, bowl_wktr: float, bowl_cum_balls: float,
    phase: str, innings: int
) -> dict[int, float]:
    """Use trained XGBoost ball model to get probability distribution."""
    phase_enc = {'powerplay': (1, 0, 0), 'middle': (0, 1, 0), 'death': (0, 0, 1)}.get(phase, (0, 1, 0))
    row = pd.DataFrame([{
        'batter_roll_sr'   : bat_sr,
        'batter_cum_runs'  : bat_cum_runs,
        'batter_cum_balls' : bat_cum_balls,
        'bowler_roll_econ' : bowl_econ,
        'bowler_roll_wktr' : bowl_wktr,
        'bowler_cum_balls' : bowl_cum_balls,
        'phase_pp'         : phase_enc[0],
        'phase_mid'        : phase_enc[1],
        'phase_death'      : phase_enc[2],
        'innings'          : innings,
    }])
    probs = ball_model.predict_proba(row)[0]
    return {int(cls): float(p) for cls, p in zip(ball_model_classes, probs)}


def _dismiss_prob(batter: str, bowler: str, phase: str, innings: int) -> float:
    """Estimate dismissal probability from matchup or bowler averages."""
    key = (batter, bowler)
    m = matchup_lookup.get(key)
    if m and m.get('m_balls', 0) >= 10:
        return float(m.get('m_dismiss_prob', DEFAULT_DISMISS_PROB))

    # Fallback: bowler's overall wicket rate
    p = player_lookup.get(bowler, {})
    wr = float(p.get('wicket_rate', DEFAULT_DISMISS_PROB))

    # Death phase boosts dismissal chances slightly
    if phase == 'death':
        wr *= 1.15
    elif phase == 'powerplay':
        wr *= 1.05
    return float(np.clip(wr, 0.01, 0.25))


def _simulate_innings(
    batting_order: list[str],
    bowling_order: list[str],
    innings: int = 1,
    target: int | None = None,
) -> dict[str, Any]:
    """
    Simulate one innings ball by ball.
    Returns total runs, wickets, ball log, and per-ball outcome distributions.
    """
    total_runs  = 0
    total_wkts  = 0
    ball_log    = []
    distributions: list[dict] = []   # full prob dist per delivery

    # Batter state
    MAX_WICKETS = 10
    MAX_BALLS   = 120
    bat_idx     = 0
    on_strike   = batting_order[bat_idx] if bat_idx < len(batting_order) else "Batter 1"
    bat_idx     += 1
    non_strike  = batting_order[bat_idx] if bat_idx < len(batting_order) else "Batter 2"
    bat_idx     += 1

    batter_runs:  dict[str, int] = defaultdict(int)
    batter_balls: dict[str, int] = defaultdict(int)
    bowler_balls: dict[str, int] = defaultdict(int)
    bowler_runs:  dict[str, int] = defaultdict(int)
    bowler_wkts:  dict[str, int] = defaultdict(int)

    balls_bowled = 0
    current_over = 0
    ball_in_over = 0

    for ball_num in range(MAX_BALLS):
        if total_wkts >= MAX_WICKETS:
            break
        if target is not None and total_runs >= target:
            break

        over   = ball_num // 6
        b_in_o = ball_num % 6
        phase  = 'powerplay' if over < 6 else ('death' if over >= 15 else 'middle')

        bowler = bowling_order[over % len(bowling_order)]

        bat_sr   = _get_batter_sr(on_strike)
        bowl_econ= _get_bowler_econ(bowler)

        # Full probability distribution over every outcome
        if ball_model is not None:
            dist = _model_ball_dist(
                bat_sr          = bat_sr,
                bat_cum_runs    = batter_runs[on_strike],
                bat_cum_balls   = batter_balls[on_strike],
                bowl_econ       = bowl_econ,
                bowl_wktr       = float(player_lookup.get(bowler, {}).get('wicket_rate', DEFAULT_DISMISS_PROB)),
                bowl_cum_balls  = bowler_balls[bowler],
                phase           = phase,
                innings         = innings,
            )
        else:
            dist = _heuristic_ball_dist(bat_sr, bowl_econ, phase)

        # Ensure all 7 outcomes (0–6 + wicket) represented
        full_dist = {k: dist.get(k, 0.0) for k in BALL_OUTCOMES}
        total_d   = sum(full_dist.values())
        full_dist = {k: v / total_d for k, v in full_dist.items()}

        # Dismissal probability (separate from run distribution)
        d_prob = _dismiss_prob(on_strike, bowler, phase, innings)

        distributions.append({
            'ball'         : f"{over+1}.{b_in_o+1}",
            'batter'       : on_strike,
            'bowler'       : bowler,
            'phase'        : phase,
            'dist'         : {str(k): round(v, 4) for k, v in full_dist.items()},
            'wicket_prob'  : round(d_prob, 4),
        })

        # Sample outcome
        is_wicket = random.random() < d_prob
        if is_wicket:
            runs = 0
            ball_log.append({
                'over': over + 1, 'ball': b_in_o + 1,
                'batter': on_strike, 'bowler': bowler,
                'runs': 0, 'wicket': True, 'phase': phase,
            })
            total_wkts += 1
            bowler_wkts[bowler] += 1
            batter_balls[on_strike] += 1
            bowler_balls[bowler]    += 1

            # Bring in next batter
            if bat_idx < len(batting_order):
                on_strike = batting_order[bat_idx]
                bat_idx  += 1
            else:
                on_strike = f"Batter {bat_idx+1}"
                bat_idx  += 1
        else:
            keys   = list(full_dist.keys())
            probs  = [full_dist[k] for k in keys]
            runs   = int(np.random.choice(keys, p=probs))
            total_runs += runs
            batter_runs[on_strike]  += runs
            batter_balls[on_strike] += 1
            bowler_runs[bowler]     += runs
            bowler_balls[bowler]    += 1

            ball_log.append({
                'over': over + 1, 'ball': b_in_o + 1,
                'batter': on_strike, 'bowler': bowler,
                'runs': runs, 'wicket': False, 'phase': phase,
            })

            # Rotate strike on odd runs
            if runs % 2 == 1:
                on_strike, non_strike = non_strike, on_strike

        # End of over: rotate strike + swap bowler conceptually
        if (ball_num + 1) % 6 == 0:
            on_strike, non_strike = non_strike, on_strike

    scorecard = {
        'batting': [
            {'batter': b, 'runs': batter_runs[b], 'balls': batter_balls[b],
             'sr': round(batter_runs[b] / max(batter_balls[b], 1) * 100, 1)}
            for b in set(batter_runs) | set(batter_balls)
        ],
        'bowling': [
            {'bowler': bw, 'runs': bowler_runs[bw], 'balls': bowler_balls[bw],
             'wickets': bowler_wkts[bw],
             'econ': round(bowler_runs[bw] / max(bowler_balls[bw] / 6, 0.1), 1)}
            for bw in set(bowler_runs) | set(bowler_balls)
        ],
    }

    return {
        'total'        : total_runs,
        'wickets'      : total_wkts,
        'balls_bowled' : min(ball_num + 1, MAX_BALLS),
        'ball_log'     : ball_log,
        'distributions': distributions,
        'scorecard'    : scorecard,
    }


def _build_order(team: str, role: str = 'batting') -> list[str]:
    """Build a batting or bowling XI from player_lookup for a given team."""
    pool = [
        name for name, stats in player_lookup.items()
        if str(stats.get('team', stats.get('batting_team', stats.get('bowling_team', '')))) == team
    ]
    if not pool:
        return [f"{team} {role.title()} {i}" for i in range(1, 12)]

    if role == 'batting':
        # Sort by batting SR descending (top strikers first)
        pool.sort(key=lambda n: float(player_lookup[n].get('bat_sr', 0)), reverse=True)
    else:
        # Sort by bowling economy ascending (best bowlers first)
        pool.sort(key=lambda n: float(player_lookup[n].get('bowl_econ', 99)))

    return (pool * 2)[:11]   # repeat if squad < 11

# ── Request / Response Schemas ─────────────────────────────────────────────────

class MatchRequest(BaseModel):
    team1: str
    team2: str
    team1_venue_status: str = 'neutral'   # home | away | neutral
    toss_winner: str = ''
    toss_decision: str = 'bat'            # bat | field

class SimulateRequest(BaseModel):
    team1: str
    team2: str
    n_simulations: int = N_MONTE_CARLO    # default 6767

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def home(request: Request):
    teams = sorted(team_stats.index.tolist())
    return templates.TemplateResponse("index.html", {"request": request, "teams": teams})


@app.post("/predict")
async def predict(data: MatchRequest):
    t1, t2 = data.team1, data.team2
    if t1 == t2:
        return JSONResponse({"error": "Teams must be different"}, status_code=400)

    pair             = tuple(sorted([t1, t2]))
    team_A_win_pct   = h2h_lookup.get(pair, 0.5)
    t1_h2h           = team_A_win_pct if t1 == pair[0] else (1 - team_A_win_pct)

    t1_home = 1 if data.team1_venue_status == 'home' else 0
    t2_home = 1 if data.team1_venue_status == 'away' else 0
    toss_w  = 1 if data.toss_winner == t1 else 0
    toss_d  = 1 if data.toss_decision == 'bat' else 0

    row = pd.DataFrame([{
        'team1_sr': team_stats.loc[t1, 'sr'],        'team2_sr': team_stats.loc[t2, 'sr'],
        'team1_econ': team_stats.loc[t1, 'econ'],    'team2_econ': team_stats.loc[t2, 'econ'],
        'team1_bat_avg': team_stats.loc[t1, 'bat_avg'], 'team2_bat_avg': team_stats.loc[t2, 'bat_avg'],
        'team1_bowl_avg': team_stats.loc[t1, 'bowl_avg'], 'team2_bowl_avg': team_stats.loc[t2, 'bowl_avg'],
        'team1_is_home': t1_home, 'team2_is_home': t2_home,
        'toss_winner_is_team1': toss_w, 'toss_decision_bat': toss_d,
        'team1_h2h_win_pct': t1_h2h,
    }])

    pred   = match_model.predict(row)[0]
    probs  = match_model.predict_proba(row)[0]
    winner = t1 if pred == 1 else t2
    win_p  = float(probs[1] if pred == 1 else probs[0]) * 100

    return {
        "winner"     : winner,
        "probability": round(win_p, 1),
        "confidence" : round(max(probs) * 100, 1),
        "team1_stats": {
            "sr": round(float(team_stats.loc[t1, 'sr']), 1),
            "bat_avg": round(float(team_stats.loc[t1, 'bat_avg']), 1),
            "econ": round(float(team_stats.loc[t1, 'econ']), 1),
            "h2h": round(t1_h2h * 100, 1),
        },
        "team2_stats": {
            "sr": round(float(team_stats.loc[t2, 'sr']), 1),
            "bat_avg": round(float(team_stats.loc[t2, 'bat_avg']), 1),
            "econ": round(float(team_stats.loc[t2, 'econ']), 1),
            "h2h": round((1 - t1_h2h) * 100, 1),
        },
    }


@app.post("/simulate")
async def simulate_match(data: SimulateRequest):
    """
    Monte Carlo match simulation — runs N_SIMULATIONS (default 6767) full innings,
    returns:
      • win_probability (team1 win %)
      • avg / std predicted scores for both teams
      • full probability distribution over every ball-outcome value
      • one representative ball-by-ball game log
      • per-over run distribution heatmap data
    """
    t1, t2   = data.team1, data.team2
    n_sims   = max(1, min(data.n_simulations, N_MONTE_CARLO))

    if t1 == t2:
        return JSONResponse({"error": "Teams must be different"}, status_code=400)

    bat_order_t1  = _build_order(t1, 'batting')
    bowl_order_t1 = _build_order(t1, 'bowling')
    bat_order_t2  = _build_order(t2, 'batting')
    bowl_order_t2 = _build_order(t2, 'bowling')

    t1_scores, t2_scores = [], []
    t1_wins = 0

    # Store distributions from first simulation for detailed output
    representative_sim = None

    # Aggregate per-outcome probability across ALL balls across ALL sims
    outcome_tallies: dict[str, int] = {str(k): 0 for k in BALL_OUTCOMES}
    outcome_tallies['W'] = 0
    total_balls = 0

    # Per-over average runs (innings 1, averaged across sims)
    over_runs_t1: dict[int, list[int]] = defaultdict(list)

    print(f"Running {n_sims} Monte Carlo simulations...")

    for sim_idx in range(n_sims):
        # ── Innings 1: team1 bats ──────────────────────────────────────
        inn1 = _simulate_innings(bat_order_t1, bowl_order_t2, innings=1)
        t1_score = inn1['total']

        # Tally outcomes from this innings
        for entry in inn1['ball_log']:
            if entry['wicket']:
                outcome_tallies['W'] += 1
            else:
                outcome_tallies[str(entry['runs'])] += 1
            total_balls += 1

        # Per-over accumulation (sim 0 only for speed)
        if sim_idx == 0:
            over_acc: dict[int, int] = defaultdict(int)
            for entry in inn1['ball_log']:
                over_acc[entry['over']] += entry['runs']
            for ov, r in over_acc.items():
                over_runs_t1[ov].append(r)

        # ── Innings 2: team2 chases ────────────────────────────────────
        inn2 = _simulate_innings(bat_order_t2, bowl_order_t1, innings=2, target=t1_score + 1)
        t2_score = inn2['total']

        t1_scores.append(t1_score)
        t2_scores.append(t2_score)
        if t1_score > t2_score:
            t1_wins += 1

        # Keep first simulation as representative
        if sim_idx == 0:
            representative_sim = {
                'innings1': {
                    'team'     : t1,
                    'score'    : t1_score,
                    'wickets'  : inn1['wickets'],
                    'ball_log' : inn1['ball_log'],
                    'scorecard': inn1['scorecard'],
                    'distributions': inn1['distributions'][:30],  # first 30 balls
                },
                'innings2': {
                    'team'     : t2,
                    'score'    : t2_score,
                    'wickets'  : inn2['wickets'],
                    'ball_log' : inn2['ball_log'],
                    'scorecard': inn2['scorecard'],
                },
            }

    # Normalised probability distribution over outcomes (across all balls in all sims)
    total_events = sum(outcome_tallies.values())
    outcome_probabilities = {
        k: round(v / max(total_events, 1), 5)
        for k, v in outcome_tallies.items()
    }

    # Per-over average
    over_avg = {ov: round(float(np.mean(runs)), 2) for ov, runs in over_runs_t1.items()}

    t1_arr, t2_arr = np.array(t1_scores), np.array(t2_scores)

    return {
        "n_simulations"    : n_sims,
        "team1"            : t1,
        "team2"            : t2,
        "win_probability"  : {
            t1: round(t1_wins / n_sims * 100, 2),
            t2: round((n_sims - t1_wins) / n_sims * 100, 2),
        },
        "predicted_scores" : {
            t1: {
                "mean": round(float(t1_arr.mean()), 1),
                "std" : round(float(t1_arr.std()), 1),
                "min" : int(t1_arr.min()),
                "max" : int(t1_arr.max()),
                "p10" : int(np.percentile(t1_arr, 10)),
                "p90" : int(np.percentile(t1_arr, 90)),
            },
            t2: {
                "mean": round(float(t2_arr.mean()), 1),
                "std" : round(float(t2_arr.std()), 1),
                "min" : int(t2_arr.min()),
                "max" : int(t2_arr.max()),
                "p10" : int(np.percentile(t2_arr, 10)),
                "p90" : int(np.percentile(t2_arr, 90)),
            },
        },
        # Full probability distribution over every possible ball outcome
        "ball_outcome_distribution": outcome_probabilities,
        # Breakdown: {runs: count, wickets: count} across all simulated balls
        "outcome_raw_counts": outcome_tallies,
        "total_balls_simulated": total_balls,
        # Per-over average runs (team1 batting, first sim only for speed)
        "over_avg_runs": over_avg,
        # One representative full match
        "representative_match": representative_sim,
    }


@app.get("/player-stats")
async def get_player_stats(team: str | None = None):
    if player_df.empty:
        return JSONResponse({"error": "player_stats.csv not found. Run data_cleaning.py first."}, status_code=404)

    df = player_df.copy()
    if team:
        df = df[df['team'] == team]

    cols = ['player', 'team', 'bat_runs', 'bat_balls', 'bat_sr', 'bat_avg',
            'boundary_pct', 'dot_pct', 'bowl_runs', 'bowl_balls',
            'bowl_wkts', 'bowl_econ', 'bowl_avg', 'wicket_rate']
    cols = [c for c in cols if c in df.columns]

    return JSONResponse(df[cols].fillna(0).round(2).to_dict(orient='records'))


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)