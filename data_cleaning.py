import pandas as pd

def clean_and_prepare_data():
    print("Loading raw data...")
    df = pd.read_csv('ipl_data.csv')

    # 1. Map legacy teams
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals', 
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad', 
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
    }
    for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
        if col in df.columns:
            df[col] = df[col].replace(team_mapping)

    # 2. Strict Filter: 10 active IPL teams
    active_teams = [
        'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
        'Royal Challengers Bengaluru', 'Delhi Capitals', 'Rajasthan Royals',
        'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
    ]
    df = df[df['batting_team'].isin(active_teams) & df['bowling_team'].isin(active_teams)]

    # 3. Exhaustive Venue Mapping
    venue_mapping = {
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium', 'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium', 'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
        'Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
        'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association IS Bindra Stadium',
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi Intl. Cricket Stadium': 'Rajiv Gandhi International Stadium',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium', 'Feroz Shah Kotla': 'Arun Jaitley Stadium'
    }
    if 'venue' in df.columns:
        df['venue'] = df['venue'].replace(venue_mapping)

    print("Calculating dynamic rolling form (Last 5 Matches)...")
    df['is_wicket'] = df['player_out'].notnull().astype(int)
    
    match_batting = df.groupby(['date', 'match_id', 'batting_team']).agg(
        runs=('runs_total', 'sum'), balls=('ball', 'count'), wickets=('is_wicket', 'sum')
    ).reset_index().sort_values('date')
    
    match_bowling = df.groupby(['date', 'match_id', 'bowling_team']).agg(
        runs_c=('runs_total', 'sum'), balls_b=('ball', 'count'), wkts_t=('is_wicket', 'sum')
    ).reset_index().sort_values('date')

    match_batting['roll_runs'] = match_batting.groupby('batting_team')['runs'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    match_batting['roll_balls'] = match_batting.groupby('batting_team')['balls'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    match_batting['roll_wkts'] = match_batting.groupby('batting_team')['wickets'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    
    match_bowling['roll_runs_c'] = match_bowling.groupby('bowling_team')['runs_c'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    match_bowling['roll_balls_b'] = match_bowling.groupby('bowling_team')['balls_b'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
    match_bowling['roll_wkts_t'] = match_bowling.groupby('bowling_team')['wkts_t'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))

    match_batting['dyn_sr'] = ((match_batting['roll_runs'] / match_batting['roll_balls']) * 100).fillna(130.0)
    match_batting['dyn_bat_avg'] = (match_batting['roll_runs'] / match_batting['roll_wkts'].replace(0, 1)).fillna(25.0)
    
    match_bowling['dyn_econ'] = (match_bowling['roll_runs_c'] / (match_bowling['roll_balls_b'] / 6)).fillna(8.0)
    match_bowling['dyn_bowl_avg'] = (match_bowling['roll_runs_c'] / match_bowling['roll_wkts_t'].replace(0, 1)).fillna(28.0)

    latest_bat = match_batting.groupby('batting_team').tail(1).set_index('batting_team')[['dyn_sr', 'dyn_bat_avg']]
    latest_bowl = match_bowling.groupby('bowling_team').tail(1).set_index('bowling_team')[['dyn_econ', 'dyn_bowl_avg']]
    team_stats = pd.concat([latest_bat, latest_bowl], axis=1)
    team_stats.rename(columns={'dyn_sr': 'sr', 'dyn_bat_avg': 'bat_avg', 'dyn_econ': 'econ', 'dyn_bowl_avg': 'bowl_avg'}, inplace=True)
    team_stats.to_csv('team_stats.csv')

    print("Extracting match-level data & Head-to-Head...")
    matches = df[df['innings'] == 1].drop_duplicates(subset=['match_id']).copy()
    matches.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2'}, inplace=True)
    matches.dropna(subset=['match_won_by', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision'], inplace=True)

    matches = pd.merge(matches, match_batting[['match_id', 'batting_team', 'dyn_sr', 'dyn_bat_avg']], left_on=['match_id', 'team1'], right_on=['match_id', 'batting_team'], how='left').rename(columns={'dyn_sr': 'team1_sr', 'dyn_bat_avg': 'team1_bat_avg'}).drop(columns=['batting_team'])
    matches = pd.merge(matches, match_batting[['match_id', 'batting_team', 'dyn_sr', 'dyn_bat_avg']], left_on=['match_id', 'team2'], right_on=['match_id', 'batting_team'], how='left').rename(columns={'dyn_sr': 'team2_sr', 'dyn_bat_avg': 'team2_bat_avg'}).drop(columns=['batting_team'])
    matches = pd.merge(matches, match_bowling[['match_id', 'bowling_team', 'dyn_econ', 'dyn_bowl_avg']], left_on=['match_id', 'team1'], right_on=['match_id', 'bowling_team'], how='left').rename(columns={'dyn_econ': 'team1_econ', 'dyn_bowl_avg': 'team1_bowl_avg'}).drop(columns=['bowling_team'])
    matches = pd.merge(matches, match_bowling[['match_id', 'bowling_team', 'dyn_econ', 'dyn_bowl_avg']], left_on=['match_id', 'team2'], right_on=['match_id', 'bowling_team'], how='left').rename(columns={'dyn_econ': 'team2_econ', 'dyn_bowl_avg': 'team2_bowl_avg'}).drop(columns=['bowling_team'])

    # Head-to-Head Logic
    matches['team_A'] = matches.apply(lambda x: min(x['team1'], x['team2']), axis=1)
    matches['team_B'] = matches.apply(lambda x: max(x['team1'], x['team2']), axis=1)
    matches['matchup'] = matches['team_A'] + " vs " + matches['team_B']
    matches['team_A_won'] = (matches['match_won_by'] == matches['team_A']).astype(int)
    matches['team_A_h2h_win_pct'] = matches.groupby('matchup')['team_A_won'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.5)
    matches['team1_h2h_win_pct'] = matches.apply(lambda x: x['team_A_h2h_win_pct'] if x['team1'] == x['team_A'] else (1 - x['team_A_h2h_win_pct']), axis=1)

    latest_h2h = matches.drop_duplicates(subset=['matchup'], keep='last')
    latest_h2h[['team_A', 'team_B', 'team_A_h2h_win_pct']].to_csv('h2h_stats.csv', index=False)

    venue_home_team = {
        'M Chinnaswamy Stadium': 'Royal Challengers Bengaluru', 'Wankhede Stadium': 'Mumbai Indians',
        'MA Chidambaram Stadium': 'Chennai Super Kings', 'Eden Gardens': 'Kolkata Knight Riders',
        'Arun Jaitley Stadium': 'Delhi Capitals', 'Rajiv Gandhi International Stadium': 'Sunrisers Hyderabad',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Kings', 'Sawai Mansingh Stadium': 'Rajasthan Royals',
        'Narendra Modi Stadium': 'Gujarat Titans', 'Ekana Cricket Stadium': 'Lucknow Super Giants'
    }
    matches['venue_home_team'] = matches['venue'].map(venue_home_team)
    matches['team1_is_home'] = (matches['team1'] == matches['venue_home_team']).astype(int)
    matches['team2_is_home'] = (matches['team2'] == matches['venue_home_team']).astype(int)
    
    matches['toss_winner_is_team1'] = (matches['toss_winner'] == matches['team1']).astype(int)
    matches['toss_decision_bat'] = (matches['toss_decision'] == 'bat').astype(int)
    matches['target'] = (matches['match_won_by'] == matches['team1']).astype(int)

    features = [
        'team1_sr', 'team2_sr', 'team1_econ', 'team2_econ', 
        'team1_bat_avg', 'team2_bat_avg', 'team1_bowl_avg', 'team2_bowl_avg',
        'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat',
        'team1_h2h_win_pct'
    ]
    
    matches.dropna(subset=features, inplace=True)
    matches[features + ['target']].to_csv('ml_ready_data.csv', index=False)
    
    print("Pre-processing complete. Output saved to 'ml_ready_data.csv', 'team_stats.csv', and 'h2h_stats.csv'.")

if __name__ == "__main__":
    clean_and_prepare_data()