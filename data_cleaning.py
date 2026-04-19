"""
data_cleaning.py — IPL Ball-by-Ball Data Preprocessor (v2)
===========================================================
Pipeline: clean → feature_eng → train → calibrate → evaluate

Outputs
-------
ml_ready_data.csv       match-level features + target
team_stats.csv          latest rolling team stats
h2h_stats.csv           head-to-head historical win %
player_stats.csv        per-player batting/bowling stats
matchup_stats.csv       batsman vs bowler historical matchup data
ball_model_data.csv     ball-level features for XGBoost ball-outcome model
match_model.joblib      trained + calibrated match-winner model
ball_model.joblib       trained + calibrated ball-outcome model
results.json            evaluation metrics for both models
shap_match.png          SHAP summary for match model
shap_ball.png           SHAP summary for ball model
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Optional imports (non-fatal) ──────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  ! optuna not installed — skipping hyperparameter search")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  ! shap not installed — skipping SHAP plots")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Constants ─────────────────────────────────────────────────────────────────

TEAM_MAPPING = {
    "Delhi Daredevils":             "Delhi Capitals",
    "Kings XI Punjab":              "Punjab Kings",
    "Deccan Chargers":              "Sunrisers Hyderabad",
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
    "Rising Pune Supergiant":       "Chennai Super Kings",   # absorbed
    "Pune Warriors":                "Delhi Capitals",        # absorbed
    "Kochi Tuskers Kerala":         "Kolkata Knight Riders", # absorbed
}

ACTIVE_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders",
    "Royal Challengers Bengaluru", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans",
]

VENUE_HOME_TEAM = {
    "M Chinnaswamy Stadium":                        "Royal Challengers Bengaluru",
    "Wankhede Stadium":                             "Mumbai Indians",
    "MA Chidambaram Stadium":                       "Chennai Super Kings",
    "Eden Gardens":                                 "Kolkata Knight Riders",
    "Arun Jaitley Stadium":                         "Delhi Capitals",
    "Rajiv Gandhi International Stadium":           "Sunrisers Hyderabad",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Kings",
    "Sawai Mansingh Stadium":                       "Rajasthan Royals",
    "Narendra Modi Stadium":                        "Gujarat Titans",
    "Ekana Cricket Stadium":                        "Lucknow Super Giants",
}

VENUE_MAPPING = {
    "Brabourne Stadium, Mumbai":                            "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai":                  "Dr DY Patil Sports Academy",
    "M.Chinnaswamy Stadium":                               "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru":                    "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk":                     "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai":            "MA Chidambaram Stadium",
    "Narendra Modi Stadium, Ahmedabad":                    "Narendra Modi Stadium",
    "Rajiv Gandhi International Stadium, Uppal":           "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad":"Rajiv Gandhi International Stadium",
    "Rajiv Gandhi Intl. Cricket Stadium":                  "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium, Jaipur":                      "Sawai Mansingh Stadium",
    "Arun Jaitley Stadium, Delhi":                         "Arun Jaitley Stadium",
    "Feroz Shah Kotla":                                    "Arun Jaitley Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh":
        "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium, Mohali":
        "Punjab Cricket Association IS Bindra Stadium",
    "Maharashtra Cricket Association Stadium, Pune":       "Maharashtra Cricket Association Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala":
        "Himachal Pradesh Cricket Association Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh":
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
}

MATCH_FEATURES = [
    "team1_sr", "team2_sr",
    "team1_econ", "team2_econ",
    "team1_bat_avg", "team2_bat_avg",
    "team1_bowl_avg", "team2_bowl_avg",
    "team1_is_home", "team2_is_home",
    "toss_winner_is_team1", "toss_decision_bat",
    "team1_h2h_win_pct",
    # new features (section 5)
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
    # pre-delivery cumulative batter state
    "batter_cum_runs", "batter_cum_balls",
    "batter_roll_sr",
    "striker_sr_vs_pace", "striker_sr_vs_spin",
    "dot_ball_pressure",
    # pre-delivery cumulative bowler state
    "bowler_cum_balls", "bowler_cum_wkts",
    "bowler_roll_econ", "bowler_roll_wktr",
    "bowler_economy_this_phase",
    # matchup
    "batter_vs_bowler_matchup_sr",
    # match state
    "innings", "over", "ball_in_over",
    "phase_pp", "phase_mid", "phase_death",
    "wickets_in_hand",
    "runs_in_innings",
    "balls_remaining",
    "required_run_rate",   # 0 for 1st innings
    # upgrade features — rolling window & pressure
    "last12_runs", "last12_wickets", "pressure_index",
]

TRAIN_SEASONS_END = 2022   # inclusive
VAL_SEASON        = 2023
TEST_SEASON_START = 2024


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide and preserve pandas compatibility."""
    if np.isscalar(numerator) and np.isscalar(denominator):
        return default if denominator == 0 else numerator / denominator

    result = pd.Series(np.divide(numerator, denominator))
    result = result.replace([np.inf, -np.inf], np.nan).fillna(default)
    return result


def _over_phase(over_int: int) -> str:
    if over_int < 6:
        return "powerplay"
    if over_int < 15:
        return "middle"
    return "death"


# ── 1. Load & clean ───────────────────────────────────────────────────────────

def load_and_clean(csv_path: str = "ipl_data.csv") -> pd.DataFrame:
    print("Loading raw data...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Standardise team names
    for col in ["batting_team", "bowling_team", "toss_winner", "match_won_by"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_MAPPING)

    # Active teams only
    df = df[
        df["batting_team"].isin(ACTIVE_TEAMS) &
        df["bowling_team"].isin(ACTIVE_TEAMS)
    ].copy()

    # Parse season / date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["season"] = df["date"].dt.year
    else:
        df["season"] = 2020  # fallback

    # Venue normalisation
    if "venue" in df.columns:
        df["venue"] = df["venue"].replace(VENUE_MAPPING)

    # Derived ball-level columns
    df["is_wicket"]      = df["player_out"].notna().astype(int)
    df["is_boundary_4"]  = (df["runs_total"] == 4).astype(int)
    df["is_boundary_6"]  = (df["runs_total"] == 6).astype(int)
    df["is_dot"]         = (df["runs_total"] == 0).astype(int)
    df["over_int"]       = df["ball"].apply(lambda b: int(b))
    df["ball_in_over"]   = df["ball"].apply(lambda b: int(round((b % 1) * 10)))
    df["over_phase"]     = df["over_int"].apply(_over_phase)

    # Bowling type heuristic (used for pace/spin SR features)
    # We don't have pitch data; approximate from name/historical tendency.
    # Will be replaced per player from ipl_squads if available.
    if "bowling_style" not in df.columns:
        df["bowling_style"] = "pace"  # default; overridden below

    df = df.sort_values(["date", "match_id", "innings", "ball"]).reset_index(drop=True)
    print(f"  → {len(df):,} deliveries | seasons {df['season'].min()}–{df['season'].max()}")
    return df


# ── 2. Rolling team stats (match-level, no leakage) ───────────────────────────
def _safe_div(num, den, fill=0.0):
    import numpy as np
    import pandas as pd

    if isinstance(num, (pd.Series, np.ndarray)):
        result = np.where(den == 0, fill, num / den)
        return pd.Series(result)
    
    return fill if den == 0 else num / den

def build_team_stats(df: pd.DataFrame):
    """
    Compute dynamic rolling last-5-match batting / bowling stats.
    Returns (match_bat_df, match_bowl_df, latest_team_stats_df).
    """
    print("Building rolling team stats (last-5 matches)...")

    match_bat = (
        df.groupby(["date", "match_id", "batting_team"])
        .agg(runs=("runs_total", "sum"),
             balls=("ball", "count"),
             wickets=("is_wicket", "sum"))
        .reset_index()
        .sort_values(["batting_team", "date"])
    )
    match_bowl = (
        df.groupby(["date", "match_id", "bowling_team"])
        .agg(runs_c=("runs_total", "sum"),
             balls_b=("ball", "count"),
             wkts_t=("is_wicket", "sum"))
        .reset_index()
        .sort_values(["bowling_team", "date"])
    )

    # Rolling 5-match sums — shift(1) ensures no same-match leakage
    for grp, df_ref, cols in [
        ("batting_team",  match_bat,  ["runs", "balls", "wickets"]),
        ("bowling_team",  match_bowl, ["runs_c", "balls_b", "wkts_t"]),
    ]:
        for col in cols:
            df_ref[f"roll_{col}"] = (
                df_ref.groupby(grp)[col]
                .transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
            )

    match_bat["dyn_sr"]      = (_safe_div(match_bat["roll_runs"],
                                          match_bat["roll_balls"], 1.3) * 100).fillna(130.0)
    match_bat["dyn_bat_avg"] = _safe_div(match_bat["roll_runs"],
                                          match_bat["roll_wickets"].replace(0, 1), 25.0).fillna(25.0)
    match_bowl["dyn_econ"]   = (_safe_div(match_bowl["roll_runs_c"],
                                           match_bowl["roll_balls_b"] / 6, 8.0)).fillna(8.0)
    match_bowl["dyn_bowl_avg"] = _safe_div(match_bowl["roll_runs_c"],
                                            match_bowl["roll_wkts_t"].replace(0, 1), 28.0).fillna(28.0)

    # Latest snapshot per team (for the prediction API)
    latest_bat  = match_bat.groupby("batting_team").last()[["dyn_sr", "dyn_bat_avg"]]
    latest_bowl = match_bowl.groupby("bowling_team").last()[["dyn_econ", "dyn_bowl_avg"]]
    team_stats  = pd.concat([latest_bat, latest_bowl], axis=1)
    team_stats.rename(columns={"dyn_sr": "sr", "dyn_bat_avg": "bat_avg",
                                "dyn_econ": "econ", "dyn_bowl_avg": "bowl_avg"}, inplace=True)
    team_stats.to_csv("team_stats.csv")
    print("  → team_stats.csv saved")
    return match_bat, match_bowl, team_stats


# ── 3. Phase run rates & new match-level features ────────────────────────────

def build_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-match phase run rates: powerplay / middle / death."""
    phase_rr = (
        df.groupby(["match_id", "batting_team", "over_phase"])
        .agg(runs=("runs_total", "sum"), balls=("ball", "count"))
        .reset_index()
    )
    phase_rr["rr"] = _safe_div(phase_rr["runs"], phase_rr["balls"] / 6, 0.0)
    phase_wide = phase_rr.pivot_table(
        index=["match_id", "batting_team"], columns="over_phase", values="rr", fill_value=0.0
    ).reset_index()
    phase_wide.columns = [f"phase_{c}_rr" if c not in ("match_id", "batting_team") else c
                          for c in phase_wide.columns]
    # Death bowling economy for bowling team
    death_df = df[df["over_phase"] == "death"].copy()
    death_bowl = (
        death_df.groupby(["match_id", "bowling_team"])
        .agg(runs=("runs_total", "sum"), balls=("ball", "count"))
        .reset_index()
    )
    death_bowl["death_econ"] = _safe_div(death_bowl["runs"], death_bowl["balls"] / 6, 9.0)
    return phase_wide, death_bowl


def build_venue_chase_features(df: pd.DataFrame, matches_base: pd.DataFrame) -> pd.DataFrame:
    """Venue win rate and chase/defend win rate per team."""
    # We compute on match level (innings==1 rows already de-duplicated)
    m = matches_base.copy()

    # Venue win rate — rolling (use historical only, shift 1)
    venue_wins = []
    for team in ACTIVE_TEAMS:
        t_df = m[(m["team1"] == team) | (m["team2"] == team)].sort_values("date").copy()
        t_df["team_won"] = t_df.apply(lambda r: 1 if r["match_won_by"] == team else 0, axis=1)
        t_df["venue_match"] = t_df["venue"]
        grp = t_df.groupby("venue_match")["team_won"].transform(
            lambda x: x.expanding().mean().shift(1).fillna(0.5))
        t_df["venue_wr"] = grp
        t_df["team"] = team
        venue_wins.append(t_df[["match_id", "team", "venue_wr"]])
    venue_df = pd.concat(venue_wins, ignore_index=True)

    # Chase win rate
    chase_wins = []
    for team in ACTIVE_TEAMS:
        t_df = m[(m["team1"] == team) | (m["team2"] == team)].sort_values("date").copy()
        t_df["team_won"]   = (t_df["match_won_by"] == team).astype(int)
        t_df["team_chased"] = t_df.apply(
            lambda r: 1 if (r["team2"] == team and r["toss_decision"] == "bat") or
                           (r["team1"] == team and r["toss_decision"] == "field") else 0, axis=1)
        t_df["chase_wr"] = (
            t_df.groupby("team_chased")["team_won"]
            .transform(lambda x: x.expanding().mean().shift(1).fillna(0.5))
        )
        t_df["team"] = team
        chase_wins.append(t_df[["match_id", "team", "chase_wr"]])
    chase_df = pd.concat(chase_wins, ignore_index=True)

    return venue_df, chase_df


# ── 4. Match-level ML dataset ─────────────────────────────────────────────────

def build_match_features(df: pd.DataFrame,
                          match_bat: pd.DataFrame,
                          match_bowl: pd.DataFrame,
                          matchup_df: pd.DataFrame | None = None) -> pd.DataFrame:
    print("Building match-level feature set...")

    # Base: one row per match (innings 1 only for team assignment)
    matches = (
        df[df["innings"] == 1]
        .drop_duplicates(subset=["match_id"])
        .copy()
    )
    matches.rename(columns={"batting_team": "team1", "bowling_team": "team2"}, inplace=True)
    matches.dropna(subset=["match_won_by", "team1", "team2",
                            "venue", "toss_winner", "toss_decision"], inplace=True)

    # Join rolling batting & bowling stats for each team at match time
    for side in ["team1", "team2"]:
        bat_cols = match_bat[["match_id", "batting_team", "dyn_sr", "dyn_bat_avg"]]
        matches = matches.merge(
            bat_cols,
            left_on=["match_id", side],
            right_on=["match_id", "batting_team"],
            how="left", suffixes=("", f"_{side}")
        ).drop(columns=["batting_team"])
        matches.rename(columns={"dyn_sr": f"{side}_sr", "dyn_bat_avg": f"{side}_bat_avg"},
                       inplace=True)

        bowl_cols = match_bowl[["match_id", "bowling_team", "dyn_econ", "dyn_bowl_avg"]]
        matches = matches.merge(
            bowl_cols,
            left_on=["match_id", side],
            right_on=["match_id", "bowling_team"],
            how="left"
        ).drop(columns=["bowling_team"])
        matches.rename(columns={"dyn_econ": f"{side}_econ", "dyn_bowl_avg": f"{side}_bowl_avg"},
                       inplace=True)

    # H2H
    matches["team_A"] = matches.apply(lambda r: min(r["team1"], r["team2"]), axis=1)
    matches["team_B"] = matches.apply(lambda r: max(r["team1"], r["team2"]), axis=1)
    matches["matchup"] = matches["team_A"] + " vs " + matches["team_B"]
    matches["team_A_won"] = (matches["match_won_by"] == matches["team_A"]).astype(int)
    matches = matches.sort_values("date")
    matches["team_A_h2h_win_pct"] = (
        matches.groupby("matchup")["team_A_won"]
        .transform(lambda x: x.expanding().mean().shift(1).fillna(0.5))
    )
    matches["team1_h2h_win_pct"] = matches.apply(
        lambda r: r["team_A_h2h_win_pct"] if r["team1"] == r["team_A"] else 1 - r["team_A_h2h_win_pct"],
        axis=1
    )
    # Save H2H table
    (matches.drop_duplicates(subset=["matchup"], keep="last")
     [["team_A", "team_B", "team_A_h2h_win_pct"]]
     .to_csv("h2h_stats.csv", index=False))
    print("  → h2h_stats.csv saved")

    # Home/away
    matches["venue_home_team"]       = matches["venue"].map(VENUE_HOME_TEAM)
    matches["team1_is_home"]         = (matches["team1"] == matches["venue_home_team"]).astype(int)
    matches["team2_is_home"]         = (matches["team2"] == matches["venue_home_team"]).astype(int)
    matches["toss_winner_is_team1"]  = (matches["toss_winner"] == matches["team1"]).astype(int)
    matches["toss_decision_bat"]     = (matches["toss_decision"] == "bat").astype(int)

    # Rolling last-5 wins per team
    match_results_t1 = matches[["date", "match_id", "team1", "match_won_by"]].copy()
    match_results_t2 = matches[["date", "match_id", "team2", "match_won_by"]].copy()
    match_results_t1["team"] = match_results_t1["team1"]
    match_results_t2["team"] = match_results_t2["team2"]
    all_results = pd.concat([
        match_results_t1.assign(won=lambda r: (r["match_won_by"] == r["team"]).astype(int)),
        match_results_t2.assign(won=lambda r: (r["match_won_by"] == r["team"]).astype(int)),
    ]).sort_values("date")
    all_results["last5_wins"] = (
        all_results.groupby("team")["won"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1).fillna(0.5))
    )
    l5 = all_results[["match_id", "team", "last5_wins"]].drop_duplicates(subset=["match_id", "team"])
    matches = matches.merge(l5.rename(columns={"team": "team1", "last5_wins": "team1_last5_wins"}),
                            on=["match_id", "team1"], how="left")
    matches = matches.merge(l5.rename(columns={"team": "team2", "last5_wins": "team2_last5_wins"}),
                            on=["match_id", "team2"], how="left")
    matches[["team1_last5_wins", "team2_last5_wins"]] = (
        matches[["team1_last5_wins", "team2_last5_wins"]].fillna(0.5))

    # Phase run rates
    phase_wide, death_bowl = build_phase_features(df)
    for side in ["team1", "team2"]:
        matches = matches.merge(
            phase_wide.rename(columns={"batting_team": side}),
            on=["match_id", side], how="left"
        )
        for ph in ["powerplay", "middle", "death"]:
            col_in  = f"phase_{ph}_rr"
            col_out = f"{side}_phase_{ph}_rr"
            if col_in in matches.columns:
                matches.rename(columns={col_in: col_out}, inplace=True)

    for side in ["team1", "team2"]:
        matches = matches.merge(
            death_bowl.rename(columns={"bowling_team": side, "death_econ": f"{side}_death_econ"}),
            on=["match_id", side], how="left"
        )

    # Venue / chase win rates
    venue_df, chase_df = build_venue_chase_features(df, matches)
    for side in ["team1", "team2"]:
        matches = matches.merge(
            venue_df.rename(columns={"team": side, "venue_wr": f"{side}_venue_winrate"}),
            on=["match_id", side], how="left"
        )
        matches = matches.merge(
            chase_df.rename(columns={"team": side, "chase_wr": f"{side}_chase_winrate"}),
            on=["match_id", side], how="left"
        )

    # Target
    matches["target"] = (matches["match_won_by"] == matches["team1"]).astype(int)

    # ── Upgrade 1A: Difference features ──────────────────────────────────────
    matches["sr_diff"]   = matches["team1_sr"]         - matches["team2_sr"]
    matches["econ_diff"] = matches["team2_econ"]       - matches["team1_econ"]
    matches["form_diff"] = matches["team1_last5_wins"] - matches["team2_last5_wins"]

    # ── Upgrade 1B: Team identity encoding ───────────────────────────────────
    all_team_names = list(set(matches["team1"].tolist() + matches["team2"].tolist()))
    team_label_enc = LabelEncoder()
    team_label_enc.fit(all_team_names)
    matches["team1_encoded"] = team_label_enc.transform(matches["team1"])
    matches["team2_encoded"] = team_label_enc.transform(matches["team2"])
    joblib.dump(team_label_enc, "team_encoder.joblib")
    print("  → team_encoder.joblib saved")

    # Fill NaN in new features with neutral values
    for col in MATCH_FEATURES:
        if col in matches.columns:
            fill = 0.5 if "winrate" in col or "h2h" in col or "last5" in col else 0.0
            matches[col] = matches[col].fillna(fill)

    matches = matches.dropna(subset=[f for f in MATCH_FEATURES if f in matches.columns])
    avail_features = [f for f in MATCH_FEATURES if f in matches.columns]

    # ── Upgrade 1C: Matchup strength per match ────────────────────────────────
    if matchup_df is not None and not matchup_df.empty:
        player_stats_df = pd.read_csv("player_stats.csv") if Path("player_stats.csv").exists() else pd.DataFrame()

        def _top_bowlers(team_name, n=5):
            if player_stats_df.empty or "bowling_team" not in player_stats_df.columns:
                return []
            sub = player_stats_df[player_stats_df["bowling_team"] == team_name]
            sub = sub.sort_values("bowl_wkts", ascending=False)
            return sub["bowler"].dropna().tolist()[:n]

        def _batters_for_team(team_name, n=8):
            if player_stats_df.empty or "batting_team" not in player_stats_df.columns:
                return []
            sub = player_stats_df[player_stats_df["batting_team"] == team_name]
            sub = sub.sort_values("bat_runs", ascending=False)
            return sub["batter"].dropna().tolist()[:n]

        t1_ms, t2_ms = [], []
        for _, row in matches.iterrows():
            t1_bat  = _batters_for_team(row["team1"])
            t2_bowl = _top_bowlers(row["team2"])
            t2_bat  = _batters_for_team(row["team2"])
            t1_bowl = _top_bowlers(row["team1"])
            t1_ms.append(compute_matchup_strength(t1_bat, t2_bowl, matchup_df))
            t2_ms.append(compute_matchup_strength(t2_bat, t1_bowl, matchup_df))
        matches["team1_matchup_strength"] = t1_ms
        matches["team2_matchup_strength"] = t2_ms
    else:
        matches["team1_matchup_strength"] = 100.0
        matches["team2_matchup_strength"] = 100.0

    # ── Upgrade 1D: Batting depth (avg SR of positions 5–8) ──────────────────
    # Compute per match from ball-by-ball data using batting position order
    depth_data = (
        df.groupby(["match_id", "innings", "batting_team"])
        .apply(lambda g: (
            g.groupby("batter")["runs_total"].sum()
             .reset_index()
             .assign(rank=lambda x: x["runs_total"].rank(method="first", ascending=False))
        ))
        .reset_index(drop=True)
    ) if not df.empty else pd.DataFrame()

    def _match_depth(match_id, team):
        if df.empty:
            return 130.0
        sub = df[(df["match_id"] == match_id) & (df["batting_team"] == team)]
        if sub.empty:
            return 130.0
        batter_order = sub.groupby("batter").agg(
            first_ball=("ball", "min"), runs=("runs_total", "sum"), balls=("ball", "count")
        ).sort_values("first_ball").reset_index()
        depth = batter_order.iloc[4:8]  # positions 5–8 (0-indexed 4–7)
        if depth.empty:
            return 130.0
        srs = _safe_div(depth["runs"], depth["balls"].replace(0, 1)) * 100
        return float(srs.mean()) if len(srs) else 130.0

    t1_depth, t2_depth = [], []
    for _, row in matches.iterrows():
        t1_depth.append(_match_depth(row["match_id"], row["team1"]))
        t2_depth.append(_match_depth(row["match_id"], row["team2"]))
    matches["team1_depth"] = t1_depth
    matches["team2_depth"] = t2_depth

    avail_features = [f for f in MATCH_FEATURES if f in matches.columns]
    matches[avail_features + ["target", "season", "match_id"]].to_csv("ml_ready_data.csv", index=False)
    print(f"  → ml_ready_data.csv saved ({len(matches)} matches, {len(avail_features)} features)")
    # 🔧 ensure season exists in final dataset
    if "season" not in matches.columns:
        if "date" in matches.columns:
            matches["season"] = pd.to_datetime(matches["date"], errors="coerce").dt.year
        else:
            matches["season"] = 2020

    matches["season"] = matches["season"].fillna(2020).astype(int)
    return matches, avail_features


# ── 5. Player stats ───────────────────────────────────────────────────────────

def build_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing player-level batting & bowling stats...")

    bat = (
        df.groupby(["batter", "batting_team"])
        .agg(bat_runs=("runs_total", "sum"),
             bat_balls=("ball", "count"),
             bat_4s=("is_boundary_4", "sum"),
             bat_6s=("is_boundary_6", "sum"),
             bat_dots=("is_dot", "sum"),
             bat_dismissals=("is_wicket", "sum"))
        .reset_index()
    )
    bat["bat_sr"]       = _safe_div(bat["bat_runs"], bat["bat_balls"]) * 100
    bat["bat_avg"]      = _safe_div(bat["bat_runs"], bat["bat_dismissals"].replace(0, 1))
    bat["boundary_pct"] = _safe_div(bat["bat_4s"] + bat["bat_6s"], bat["bat_balls"]) * 100
    bat["dot_pct"]      = _safe_div(bat["bat_dots"], bat["bat_balls"]) * 100

    # SR vs pace / spin (approximate: columns may not exist)
    sr_pace = _build_sr_by_style(df, "pace")
    sr_spin = _build_sr_by_style(df, "spin")
    bat = bat.merge(sr_pace, on="batter", how="left").fillna({"sr_vs_pace": bat["bat_sr"]})
    bat = bat.merge(sr_spin, on="batter", how="left").fillna({"sr_vs_spin": bat["bat_sr"]})

    bowl = (
        df.groupby(["bowler", "bowling_team"])
        .agg(bowl_runs=("runs_total", "sum"),
             bowl_balls=("ball", "count"),
             bowl_wkts=("is_wicket", "sum"))
        .reset_index()
    )
    bowl["bowl_econ"]   = _safe_div(bowl["bowl_runs"], bowl["bowl_balls"] / 6)
    bowl["bowl_sr"]     = _safe_div(bowl["bowl_balls"], bowl["bowl_wkts"].replace(0, 1))
    bowl["bowl_avg"]    = _safe_div(bowl["bowl_runs"], bowl["bowl_wkts"].replace(0, 1))
    bowl["wicket_rate"] = _safe_div(bowl["bowl_wkts"], bowl["bowl_balls"])

    # Phase economy
    for phase in ["powerplay", "middle", "death"]:
        ph_df = df[df["over_phase"] == phase]
        ph_bowl = (
            ph_df.groupby("bowler")
            .agg(r=("runs_total", "sum"), b=("ball", "count"))
            .reset_index()
        )
        ph_bowl[f"econ_{phase}"] = _safe_div(ph_bowl["r"], ph_bowl["b"] / 6)
        bowl = bowl.merge(ph_bowl[["bowler", f"econ_{phase}"]], on="bowler", how="left")

    player_stats = pd.merge(
        bat, bowl,
        left_on=["batter", "batting_team"],
        right_on=["bowler", "bowling_team"],
        how="outer", suffixes=("_bat", "_bowl")
    )
    player_stats["player"] = player_stats["batter"].fillna(player_stats["bowler"])
    player_stats["team"]   = player_stats["batting_team"].fillna(player_stats["bowling_team"])
    player_stats.to_csv("player_stats.csv", index=False)
    print("  → player_stats.csv saved")
    return player_stats


def _build_sr_by_style(df: pd.DataFrame, style: str) -> pd.DataFrame:
    """Approximate SR vs pace/spin using bowling_style column if available."""
    if "bowling_style" not in df.columns:
        return pd.DataFrame(columns=["batter", f"sr_vs_{style}"])
    sub = df[df["bowling_style"].str.lower().str.contains(style, na=False)]
    if sub.empty:
        return pd.DataFrame(columns=["batter", f"sr_vs_{style}"])
    g = sub.groupby("batter").agg(r=("runs_total", "sum"), b=("ball", "count")).reset_index()
    g[f"sr_vs_{style}"] = _safe_div(g["r"], g["b"]) * 100
    return g[["batter", f"sr_vs_{style}"]]


# ── 6. Matchup stats ──────────────────────────────────────────────────────────

def build_matchup_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("Building batsman vs bowler matchup table...")
    matchup = (
        df.groupby(["batter", "bowler"])
        .agg(m_runs=("runs_total", "sum"),
             m_balls=("ball", "count"),
             m_dismissals=("is_wicket", "sum"),
             m_4s=("is_boundary_4", "sum"),
             m_6s=("is_boundary_6", "sum"))
        .reset_index()
    )
    matchup["m_sr"]           = _safe_div(matchup["m_runs"], matchup["m_balls"]) * 100
    matchup["m_dismiss_prob"] = _safe_div(matchup["m_dismissals"], matchup["m_balls"])
    matchup.to_csv("matchup_stats.csv", index=False)
    print("  → matchup_stats.csv saved")
    return matchup


def compute_matchup_strength(team_batters: list, opponent_bowlers: list,
                              matchup_df: pd.DataFrame, default: float = 100.0) -> float:
    """Mean SR for team's batters vs opponent's top bowlers from matchup_df."""
    srs = []
    mu_index = matchup_df.set_index(["batter", "bowler"]) if not matchup_df.empty else None
    for batter in team_batters:
        for bowler in opponent_bowlers:
            try:
                row = mu_index.loc[(batter, bowler)]
                sr  = float(row["m_sr"])
                if not (np.isnan(sr) or np.isinf(sr)):
                    srs.append(sr)
            except (KeyError, TypeError):
                pass
    return float(np.mean(srs)) if srs else default


# ── 7. Ball-level ML dataset (LEAKAGE-FREE) ───────────────────────────────────

def build_ball_features(df: pd.DataFrame, matchup_df: pd.DataFrame) -> pd.DataFrame:
    print("Building ball-level XGBoost training dataset (leakage-free)...")

    df = df.sort_values(["batter", "date", "match_id", "innings", "ball"]).copy()

    # === FIX: shift INSIDE group to avoid cross-player leakage ===
    for grp, src_col, out_col in [
        ("batter", "runs_total", "batter_cum_runs"),
        ("bowler", "runs_total", "bowler_cum_runs"),
    ]:
        df[out_col] = df.groupby(grp)["runs_total"].transform(
            lambda x: x.cumsum().shift(1).fillna(0)
        )

    df["batter_cum_balls"] = df.groupby("batter").cumcount()        # balls faced before this delivery
    df["bowler_cum_balls"] = df.groupby("bowler").cumcount()

    df["batter_cum_wkts"]  = df.groupby("batter")["is_wicket"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df["bowler_cum_wkts"]  = df.groupby("bowler")["is_wicket"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )

    df["batter_roll_sr"]  = _safe_div(df["batter_cum_runs"],
                                       df["batter_cum_balls"].replace(0, 1)) * 100
    df["bowler_roll_econ"] = _safe_div(df["bowler_cum_runs"],
                                        (df["bowler_cum_balls"].replace(0, 1) / 6))
    df["bowler_roll_wktr"] = _safe_div(df["bowler_cum_wkts"],
                                        df["bowler_cum_balls"].replace(0, 1))

    # SR vs pace / spin (fallback: overall SR)
    df["striker_sr_vs_pace"] = df["batter_roll_sr"]
    df["striker_sr_vs_spin"] = df["batter_roll_sr"]
    if "bowling_style" in df.columns:
        pace_mask = df["bowling_style"].str.lower().str.contains("pace", na=False)
        spin_mask = ~pace_mask
        df.loc[~pace_mask, "striker_sr_vs_pace"] = df.loc[~pace_mask, "batter_roll_sr"]
        df.loc[~spin_mask, "striker_sr_vs_spin"] = df.loc[~spin_mask, "batter_roll_sr"]

    # Matchup SR (batter vs bowler historical, pre-match only — approximation via merge)
    mu = matchup_df[["batter", "bowler", "m_sr"]].rename(columns={"m_sr": "batter_vs_bowler_matchup_sr"})
    df = df.merge(mu, on=["batter", "bowler"], how="left")
    df["batter_vs_bowler_matchup_sr"] = df["batter_vs_bowler_matchup_sr"].fillna(df["batter_roll_sr"])

    # Phase one-hot
    df["phase_pp"]    = (df["over_phase"] == "powerplay").astype(int)
    df["phase_mid"]   = (df["over_phase"] == "middle").astype(int)
    df["phase_death"] = (df["over_phase"] == "death").astype(int)

    # Bowler economy this phase (per match)
    phase_econ = (
        df.groupby(["match_id", "innings", "bowler", "over_phase"])
        .apply(lambda g: _safe_div(g["runs_total"].sum(), len(g) / 6, 9.0))
        .reset_index(name="bowler_economy_this_phase")
    )
    df = df.merge(phase_econ, on=["match_id", "innings", "bowler", "over_phase"], how="left")
    df["bowler_economy_this_phase"] = df["bowler_economy_this_phase"].fillna(df["bowler_roll_econ"])

    # Match state features
    # These require cumulative within-innings state
    df = df.sort_values(["match_id", "innings", "ball"]).reset_index(drop=True)

    df["runs_in_innings"] = df.groupby(["match_id", "innings"])["runs_total"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df["wickets_in_innings"] = df.groupby(["match_id", "innings"])["is_wicket"].transform(
        lambda x: x.cumsum().shift(1).fillna(0)
    )
    df["wickets_in_hand"]  = 10 - df["wickets_in_innings"]
    df["ball_seq"]         = df.groupby(["match_id", "innings"]).cumcount()
    df["balls_remaining"]  = 120 - df["ball_seq"]
    df["over"]             = df["over_int"]

    # Dot-ball pressure (consecutive dots faced by batter in this innings)
    def _dot_pressure(series):
        """Consecutive dots so far by this batter."""
        result = []
        streak = 0
        for val in series:
            result.append(streak)
            if val == 0:
                streak += 1
            else:
                streak = 0
        return result

    df["dot_ball_pressure"] = (
        df.groupby(["match_id", "innings", "batter"])["runs_total"]
        .transform(lambda x: pd.Series(_dot_pressure(x), index=x.index))
    )

    # Required run rate (2nd innings only)
    df["required_run_rate"] = 0.0
    inn2 = df["innings"] == 2
    if inn2.any():
        # target = 1st innings score + 1 (approx via match-level)
        inn1_totals = (
            df[df["innings"] == 1]
            .groupby("match_id")["runs_total"]
            .sum()
            .reset_index()
            .rename(columns={"runs_total": "inn1_total"})
        )
        df = df.merge(inn1_totals, on="match_id", how="left")
        df["inn1_total"] = df["inn1_total"].fillna(160)
        df.loc[inn2, "required_run_rate"] = _safe_div(
            df.loc[inn2, "inn1_total"] + 1 - df.loc[inn2, "runs_in_innings"],
            df.loc[inn2, "balls_remaining"] / 6, 999.0
        )
        df["required_run_rate"] = df["required_run_rate"].clip(0, 36)

    # Target: runs outcome (clipped; wickets NOT included — post-delivery info)
    df["ball_outcome"] = df["runs_total"].clip(upper=6)
    df.loc[df["ball_outcome"] == 5, "ball_outcome"] = 4

    # ── Upgrade 1E: Rolling window (last 12 balls) & pressure index ───────────
    df["last12_runs"] = df.groupby(["match_id", "innings"])["runs_total"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum()
    )
    df["last12_wickets"] = df.groupby(["match_id", "innings"])["is_wicket"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum()
    )

    # current_rr = runs scored / overs bowled so far
    df["current_rr"] = _safe_div(df["runs_in_innings"], (df["ball_seq"] / 6).replace(0, 0.01), 0.0)
    df["pressure_index"] = np.where(
        (df["innings"] == 2) & (df["current_rr"] > 0),
        df["required_run_rate"] / df["current_rr"].clip(lower=0.5),
        1.0
    )
    df["pressure_index"] = df["pressure_index"].fillna(1.0).clip(0, 5)

    avail_ball_features = [f for f in BALL_FEATURES if f in df.columns]
    ball_df = df[avail_ball_features + ["ball_outcome", "season"]].dropna()
    ball_df.to_csv("ball_model_data.csv", index=False)
    print(f"  → ball_model_data.csv saved ({len(ball_df):,} deliveries, "
          f"{len(avail_ball_features)} features)")
    return ball_df, avail_ball_features


# ── 8. Time-based train / val / test split ────────────────────────────────────

def time_split(df: pd.DataFrame, feature_cols: list[str], target_col: str = "target"):
    train = df[df["season"] <= TRAIN_SEASONS_END]
    val   = df[df["season"] == VAL_SEASON]
    test  = df[df["season"] >= TEST_SEASON_START]
    X_train, y_train = train[feature_cols].values, train[target_col].values
    X_val,   y_val   = val[feature_cols].values,   val[target_col].values
    X_test,  y_test  = test[feature_cols].values,  test[target_col].values
    print(f"  Split → train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── 9. Optuna hyperparameter tuning ───────────────────────────────────────────

def tune_xgb(X_train, y_train, X_val, y_val, n_trials: int = 40,
             n_classes: int = 2) -> dict:
    if not HAS_OPTUNA:
        return dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                    min_child_weight=3, eval_metric="logloss" if n_classes == 2 else "mlogloss",
                    random_state=42)

    objective_metric = "logloss" if n_classes == 2 else "mlogloss"

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 600, step=100),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 6),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 5),
            subsample        = trial.suggest_float("subsample", 0.7, 0.9),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            gamma            = trial.suggest_float("gamma", 0.0, 0.3),
            eval_metric      = objective_metric,
            early_stopping_rounds = 50,
            random_state     = 42,
            verbosity        = 0,
        )
        if n_classes > 2:
            params["num_class"] = n_classes
            params["objective"] = "multi:softprob"
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = m.predict_proba(X_val)
        return log_loss(y_val, proba)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    best["eval_metric"]  = objective_metric
    if n_classes > 2:
        best["num_class"] = n_classes
        best["objective"] = "multi:softprob"
    print(f"  Best params: {best}")
    return best


# ── 10. Train & calibrate ─────────────────────────────────────────────────────

def train_match_model(X_train, y_train, X_val, y_val, feature_names: list[str]):
    print("Training match-winner model...")
    params = tune_xgb(X_train, y_train, X_val, y_val, n_trials=30)
    base   = XGBClassifier(**params)
    base.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=False)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    # Store feature names on wrapper for explainability
    cal.feature_names_in_ = np.array(feature_names)
    joblib.dump({"model": cal, "features": feature_names}, "match_model.joblib")
    print("  → match_model.joblib saved")
    return cal, base


def train_ball_model(X_train, y_train, X_val, y_val, classes, feature_names: list[str]):
    print("Training ball-outcome model...")
    n_cls  = len(classes)
    params = tune_xgb(X_train, y_train, X_val, y_val, n_trials=30, n_classes=n_cls)
    base   = XGBClassifier(**params)
    base.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=False)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    cal.feature_names_in_ = np.array(feature_names)
    joblib.dump({"model": cal, "features": feature_names, "classes": classes}, "ball_model.joblib")
    print("  → ball_model.joblib saved")
    return cal, base


# ── 11. Evaluation ────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str, classes=None) -> dict:
    preds  = model.predict(X_test)
    proba  = model.predict_proba(X_test)

    if classes is None or len(classes) == 2:
        # Binary
        metrics = {
            "accuracy"   : round(accuracy_score(y_test, preds), 4),
            "roc_auc"    : round(roc_auc_score(y_test, proba[:, 1]), 4),
            "log_loss"   : round(log_loss(y_test, proba), 4),
            "brier_score": round(brier_score_loss(y_test, proba[:, 1]), 4),
        }
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=list(range(len(classes))))
        metrics = {
            "accuracy" : round(accuracy_score(y_test, preds), 4),
            "log_loss" : round(log_loss(y_test, proba), 4),
        }
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_bin, proba, multi_class="ovr"), 4)
        except Exception:
            pass

    print(f"\n  [{name}] TEST SET METRICS")
    for k, v in metrics.items():
        print(f"    {k:<15}: {v}")

    # Calibration curve (binary only)
    if HAS_MPL and (classes is None or len(classes) == 2):
        try:
            fraction, mean_pred = calibration_curve(y_test, proba[:, 1], n_bins=10)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            ax.plot(mean_pred, fraction, "s-", label=name)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"Calibration — {name}")
            ax.legend()
            fig.tight_layout()
            fname = f"calibration_{name.replace(' ', '_')}.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"  → {fname} saved")
        except Exception as e:
            print(f"  ! Calibration plot failed: {e}")

    return metrics


# ── 12. SHAP plots ────────────────────────────────────────────────────────────

def shap_plot(base_model, X_sample, feature_names: list[str], title: str, fname: str):
    if not HAS_SHAP or not HAS_MPL:
        return
    try:
        X_df  = pd.DataFrame(X_sample[:500], columns=feature_names)
        expln = shap.TreeExplainer(base_model)
        vals  = expln.shap_values(X_df)
        if isinstance(vals, list):
            vals = vals[1]  # binary positive class
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(vals, X_df, max_display=15, show=False, plot_type="bar")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close("all")
        print(f"  → {fname} saved")
    except Exception as e:
        print(f"  ! SHAP plot failed: {e}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def clean_and_prepare_data(csv_path: str = "ipl_data.csv") -> None:
    results: dict = {}

    # ── Step 1: Load & clean ─────────────────────────────────────
    df = load_and_clean(csv_path)

    # ── Step 2: Team stats ────────────────────────────────────────
    match_bat, match_bowl, _ = build_team_stats(df)

    # ── Step 3: Player stats ──────────────────────────────────────
    build_player_stats(df)

    # ── Step 4: Matchup stats ─────────────────────────────────────
    matchup_df = build_matchup_stats(df)

    # ── Step 5: Match-level features ──────────────────────────────
    matches_df, match_feats = build_match_features(df, match_bat, match_bowl, matchup_df)

    # ── Step 6: Ball-level features ───────────────────────────────
    ball_df, ball_feats = build_ball_features(df, matchup_df)

    # ══════════════ MATCH MODEL ════════════════════════════════════
    print("\n=== MATCH-WINNER MODEL ===")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = time_split(matches_df, match_feats)
    if len(X_tr) == 0:
        print("  ! Not enough match data for train/val/test split — skipping model training")
    else:
        cal_m, base_m = train_match_model(X_tr, y_tr, X_vl, y_vl, match_feats)
        m_metrics = evaluate_model(cal_m, X_te, y_te, "Match Winner")
        results["match_model"] = {"features": match_feats, **m_metrics}
        shap_plot(base_m, X_tr, match_feats, "Match Model — Feature Importance", "shap_match.png")

    # ══════════════ BALL MODEL ═════════════════════════════════════
    print("\n=== BALL-OUTCOME MODEL ===")
    le   = LabelEncoder()
    y_ball = le.fit_transform(ball_df["ball_outcome"].astype(int).replace(5, 4))
    ball_df = ball_df.copy()
    ball_df["ball_outcome_enc"] = y_ball

    Xb_tr, yb_tr, Xb_vl, yb_vl, Xb_te, yb_te = time_split(
        ball_df.assign(target=ball_df["ball_outcome_enc"]), ball_feats, target_col="target")
    if len(Xb_tr) == 0:
        print("  ! Not enough ball data — skipping ball model training")
    else:
        classes = list(le.classes_)
        cal_b, base_b = train_ball_model(Xb_tr, yb_tr, Xb_vl, yb_vl, classes, ball_feats)
        b_metrics = evaluate_model(cal_b, Xb_te, yb_te, "Ball Outcome", classes=classes)
        results["ball_model"] = {"features": ball_feats, "classes": [int(c) for c in classes],
                                  **b_metrics}
        shap_plot(base_b, Xb_tr, ball_feats, "Ball Model — Feature Importance", "shap_ball.png")

    # ── Save results ──────────────────────────────────────────────
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n  → results.json saved")
    print("\nPipeline complete.")


if __name__ == "__main__":
    clean_and_prepare_data()