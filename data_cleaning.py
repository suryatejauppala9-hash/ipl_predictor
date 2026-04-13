"""
data_cleaning.py — IPL Ball-by-Ball Data Preprocessor
Outputs:
  ml_ready_data.csv      — match-level features for win-prediction model
  team_stats.csv         — latest rolling team stats
  h2h_stats.csv          — head-to-head historical win percentages
  player_stats.csv       — per-player batting/bowling stats for simulation
  matchup_stats.csv      — batsman vs bowler historical matchup data
  ball_model_data.csv    — ball-level features for XGBoost ball-outcome model
"""

import pandas as pd
import numpy as np

# ── Constants 

TEAM_MAPPING = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

ACTIVE_TEAMS = [
    'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Royal Challengers Bengaluru', 'Delhi Capitals', 'Rajasthan Royals',
    'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
]

VENUE_MAPPING = {
    'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
    'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
    'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
    'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
    'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
    'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
    'Rajiv Gandhi Intl. Cricket Stadium': 'Rajiv Gandhi International Stadium',
    'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
    'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
    'Feroz Shah Kotla': 'Arun Jaitley Stadium',
}

VENUE_HOME_TEAM = {
    'M Chinnaswamy Stadium': 'Royal Challengers Bengaluru',
    'Wankhede Stadium': 'Mumbai Indians',
    'MA Chidambaram Stadium': 'Chennai Super Kings',
    'Eden Gardens': 'Kolkata Knight Riders',
    'Arun Jaitley Stadium': 'Delhi Capitals',
    'Rajiv Gandhi International Stadium': 'Sunrisers Hyderabad',
    'Punjab Cricket Association IS Bindra Stadium': 'Punjab Kings',
    'Sawai Mansingh Stadium': 'Rajasthan Royals',
    'Narendra Modi Stadium': 'Gujarat Titans',
    'Ekana Cricket Stadium': 'Lucknow Super Giants',
}

# ── Helpers 

def get_over_phase(ball_number: float) -> str:
    """Classify each delivery into powerplay / middle / death."""
    over = int(ball_number)
    if over < 6:
        return 'powerplay'
    elif over < 15:
        return 'middle'
    return 'death'


def safe_divide(numerator, denominator, default=0.0):
    """Avoid zero-division; return default when denominator is zero."""
    return np.where(denominator == 0, default, numerator / denominator)

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def clean_and_prepare_data(csv_path: str = 'ipl_data.csv') -> None:
    print("Loading raw data...")
    df = pd.read_csv(csv_path)

    # ── 1. Standardise team names ─────────────────────────────────────────
    for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_MAPPING)

    # ── 2. Keep only active-team matches ─────────────────────────────────
    df = df[
        df['batting_team'].isin(ACTIVE_TEAMS) &
        df['bowling_team'].isin(ACTIVE_TEAMS)
    ].copy()

    # ── 3. Standardise venue names ────────────────────────────────────────
    if 'venue' in df.columns:
        df['venue'] = df['venue'].replace(VENUE_MAPPING)

    # ── 4. Derived columns ────────────────────────────────────────────────
    df['is_wicket'] = df['player_out'].notnull().astype(int)
    df['is_boundary_4'] = (df['runs_total'] == 4).astype(int)
    df['is_boundary_6'] = (df['runs_total'] == 6).astype(int)
    df['is_dot'] = (df['runs_total'] == 0).astype(int)
    df['over_phase'] = df['ball'].apply(get_over_phase)

    df = df.sort_values(['date', 'match_id', 'innings', 'ball'])

    # ── 5. Rolling team form (last-5 matches) ────────────────────────────
    print("Calculating dynamic rolling team form (Last 5 Matches)...")

    match_bat = (
        df.groupby(['date', 'match_id', 'batting_team'])
        .agg(runs=('runs_total', 'sum'), balls=('ball', 'count'), wickets=('is_wicket', 'sum'))
        .reset_index()
        .sort_values('date')
    )
    match_bowl = (
        df.groupby(['date', 'match_id', 'bowling_team'])
        .agg(runs_c=('runs_total', 'sum'), balls_b=('ball', 'count'), wkts_t=('is_wicket', 'sum'))
        .reset_index()
        .sort_values('date')
    )

    for grp_col, df_ref, cols in [
        ('batting_team', match_bat, ['runs', 'balls', 'wickets']),
        ('bowling_team', match_bowl, ['runs_c', 'balls_b', 'wkts_t']),
    ]:
        for col in cols:
            df_ref[f'roll_{col}'] = (
                df_ref.groupby(grp_col)[col]
                .transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
            )

    match_bat['dyn_sr']      = (safe_divide(match_bat['roll_runs'], match_bat['roll_balls'], 1.3) * 100).fillna(130.0)
    match_bat['dyn_bat_avg'] = safe_divide(match_bat['roll_runs'], match_bat['roll_wickets'].replace(0, 1), 25.0).fillna(25.0)
    match_bowl['dyn_econ']   = (safe_divide(match_bowl['roll_runs_c'], match_bowl['roll_balls_b'] / 6, 8.0)).fillna(8.0)
    match_bowl['dyn_bowl_avg'] = safe_divide(match_bowl['roll_runs_c'], match_bowl['roll_wkts_t'].replace(0, 1), 28.0).fillna(28.0)

    latest_bat  = match_bat.groupby('batting_team').tail(1).set_index('batting_team')[['dyn_sr', 'dyn_bat_avg']]
    latest_bowl = match_bowl.groupby('bowling_team').tail(1).set_index('bowling_team')[['dyn_econ', 'dyn_bowl_avg']]
    team_stats  = pd.concat([latest_bat, latest_bowl], axis=1)
    team_stats.rename(columns={'dyn_sr': 'sr', 'dyn_bat_avg': 'bat_avg', 'dyn_econ': 'econ', 'dyn_bowl_avg': 'bowl_avg'}, inplace=True)
    team_stats.to_csv('team_stats.csv')
    print("  → team_stats.csv saved")

    # ── 6. Player-level stats ─────────────────────────────────────────────
    print("Computing player-level batting & bowling stats...")

    bat_stats = (
        df.groupby(['batter', 'batting_team'])
        .agg(
            bat_runs=('runs_total', 'sum'),
            bat_balls=('ball', 'count'),
            bat_4s=('is_boundary_4', 'sum'),
            bat_6s=('is_boundary_6', 'sum'),
            bat_dots=('is_dot', 'sum'),
            bat_dismissals=('is_wicket', 'sum'),
        )
        .reset_index()
    )
    bat_stats['bat_sr']      = safe_divide(bat_stats['bat_runs'], bat_stats['bat_balls']) * 100
    bat_stats['bat_avg']     = safe_divide(bat_stats['bat_runs'], bat_stats['bat_dismissals'].replace(0, 1))
    bat_stats['boundary_pct']= safe_divide(bat_stats['bat_4s'] + bat_stats['bat_6s'], bat_stats['bat_balls']) * 100
    bat_stats['dot_pct']     = safe_divide(bat_stats['bat_dots'], bat_stats['bat_balls']) * 100

    bowl_stats = (
        df.groupby(['bowler', 'bowling_team'])
        .agg(
            bowl_runs=('runs_total', 'sum'),
            bowl_balls=('ball', 'count'),
            bowl_wkts=('is_wicket', 'sum'),
        )
        .reset_index()
    )
    bowl_stats['bowl_econ']   = safe_divide(bowl_stats['bowl_runs'], bowl_stats['bowl_balls'] / 6)
    bowl_stats['bowl_sr']     = safe_divide(bowl_stats['bowl_balls'], bowl_stats['bowl_wkts'].replace(0, 1))
    bowl_stats['bowl_avg']    = safe_divide(bowl_stats['bowl_runs'], bowl_stats['bowl_wkts'].replace(0, 1))
    bowl_stats['wicket_rate'] = safe_divide(bowl_stats['bowl_wkts'], bowl_stats['bowl_balls'])

    player_stats = pd.merge(
        bat_stats, bowl_stats,
        left_on=['batter', 'batting_team'], right_on=['bowler', 'bowling_team'],
        how='outer', suffixes=('_bat', '_bowl')
    )
    player_stats['player'] = player_stats['batter'].fillna(player_stats['bowler'])
    player_stats['team']   = player_stats['batting_team'].fillna(player_stats['bowling_team'])
    player_stats.to_csv('player_stats.csv', index=False)
    print("  → player_stats.csv saved")

    # ── 7. Batsman vs Bowler matchup stats ───────────────────────────────
    print("Building batsman vs bowler matchup table...")

    matchup = (
        df.groupby(['batter', 'bowler'])
        .agg(
            m_runs=('runs_total', 'sum'),
            m_balls=('ball', 'count'),
            m_dismissals=('is_wicket', 'sum'),
            m_4s=('is_boundary_4', 'sum'),
            m_6s=('is_boundary_6', 'sum'),
        )
        .reset_index()
    )
    matchup['m_sr']       = safe_divide(matchup['m_runs'], matchup['m_balls']) * 100
    matchup['m_dismiss_prob'] = safe_divide(matchup['m_dismissals'], matchup['m_balls'])
    matchup.to_csv('matchup_stats.csv', index=False)
    print("  → matchup_stats.csv saved")

    # ── 8. Ball-level ML dataset ─────────────────────────────────────────
    print("Building ball-level XGBoost training dataset...")

    # Rolling batsman stats up to (but not including) current ball
    df = df.sort_values(['batter', 'date', 'match_id', 'innings', 'ball'])
    df['batter_cum_runs']   = df.groupby('batter')['runs_total'].cumsum().shift(1).fillna(0)
    df['batter_cum_balls']  = df.groupby('batter').cumcount()
    df['batter_cum_wkts']   = df.groupby('batter')['is_wicket'].cumsum().shift(1).fillna(0)
    df['batter_roll_sr']    = safe_divide(df['batter_cum_runs'], df['batter_cum_balls'].replace(0, 1)) * 100

    df = df.sort_values(['bowler', 'date', 'match_id', 'innings', 'ball'])
    df['bowler_cum_runs']   = df.groupby('bowler')['runs_total'].cumsum().shift(1).fillna(0)
    df['bowler_cum_balls']  = df.groupby('bowler').cumcount()
    df['bowler_cum_wkts']   = df.groupby('bowler')['is_wicket'].cumsum().shift(1).fillna(0)
    df['bowler_roll_econ']  = safe_divide(df['bowler_cum_runs'], (df['bowler_cum_balls'].replace(0, 1) / 6))
    df['bowler_roll_wktr']  = safe_divide(df['bowler_cum_wkts'], df['bowler_cum_balls'].replace(0, 1))

    df['phase_pp']   = (df['over_phase'] == 'powerplay').astype(int)
    df['phase_mid']  = (df['over_phase'] == 'middle').astype(int)
    df['phase_death']= (df['over_phase'] == 'death').astype(int)

    # Target: runs outcome clipped to {0,1,2,3,4,6}
    df['ball_outcome'] = df['runs_total'].clip(upper=6)
    df.loc[df['ball_outcome'] == 5, 'ball_outcome'] = 4  # treat 5 as 4 (rare overthrows)

    ball_features = [
        'batter_roll_sr', 'batter_cum_runs', 'batter_cum_balls',
        'bowler_roll_econ', 'bowler_roll_wktr', 'bowler_cum_balls',
        'phase_pp', 'phase_mid', 'phase_death',
        'innings', 'is_wicket', 'ball_outcome'
    ]
    ball_df = df[ball_features].dropna()
    ball_df.to_csv('ball_model_data.csv', index=False)
    print("  → ball_model_data.csv saved")

    # ── 9. Match-level ML dataset (existing logic, preserved) ────────────
    print("Extracting match-level data & Head-to-Head...")

    matches = df[df['innings'] == 1].drop_duplicates(subset=['match_id']).copy()
    matches.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2'}, inplace=True)
    matches.dropna(subset=['match_won_by', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision'], inplace=True)

    for side, col_prefix in [('team1', 'team1'), ('team2', 'team2')]:
        matches = pd.merge(
            matches,
            match_bat[['match_id', 'batting_team', 'dyn_sr', 'dyn_bat_avg']],
            left_on=['match_id', side], right_on=['match_id', 'batting_team'], how='left'
        ).rename(columns={'dyn_sr': f'{col_prefix}_sr', 'dyn_bat_avg': f'{col_prefix}_bat_avg'}).drop(columns=['batting_team'])
        matches = pd.merge(
            matches,
            match_bowl[['match_id', 'bowling_team', 'dyn_econ', 'dyn_bowl_avg']],
            left_on=['match_id', side], right_on=['match_id', 'bowling_team'], how='left'
        ).rename(columns={'dyn_econ': f'{col_prefix}_econ', 'dyn_bowl_avg': f'{col_prefix}_bowl_avg'}).drop(columns=['bowling_team'])

    matches['team_A']    = matches.apply(lambda x: min(x['team1'], x['team2']), axis=1)
    matches['team_B']    = matches.apply(lambda x: max(x['team1'], x['team2']), axis=1)
    matches['matchup']   = matches['team_A'] + " vs " + matches['team_B']
    matches['team_A_won'] = (matches['match_won_by'] == matches['team_A']).astype(int)
    matches['team_A_h2h_win_pct'] = (
        matches.groupby('matchup')['team_A_won']
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0.5)
    )
    matches['team1_h2h_win_pct'] = matches.apply(
        lambda x: x['team_A_h2h_win_pct'] if x['team1'] == x['team_A'] else (1 - x['team_A_h2h_win_pct']),
        axis=1
    )

    matches.drop_duplicates(subset=['matchup'], keep='last')[
        ['team_A', 'team_B', 'team_A_h2h_win_pct']
    ].to_csv('h2h_stats.csv', index=False)
    print("  → h2h_stats.csv saved")

    matches['venue_home_team']    = matches['venue'].map(VENUE_HOME_TEAM)
    matches['team1_is_home']      = (matches['team1'] == matches['venue_home_team']).astype(int)
    matches['team2_is_home']      = (matches['team2'] == matches['venue_home_team']).astype(int)
    matches['toss_winner_is_team1'] = (matches['toss_winner'] == matches['team1']).astype(int)
    matches['toss_decision_bat']  = (matches['toss_decision'] == 'bat').astype(int)
    matches['target']             = (matches['match_won_by'] == matches['team1']).astype(int)

    features = [
        'team1_sr', 'team2_sr', 'team1_econ', 'team2_econ',
        'team1_bat_avg', 'team2_bat_avg', 'team1_bowl_avg', 'team2_bowl_avg',
        'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat',
        'team1_h2h_win_pct'
    ]
    matches.dropna(subset=features, inplace=True)
    matches[features + ['target']].to_csv('ml_ready_data.csv', index=False)
    print("  → ml_ready_data.csv saved")

    print("\nPre-processing complete.")


if __name__ == "__main__":
    clean_and_prepare_data()