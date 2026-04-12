import pandas as pd

def clean_and_prepare_data():
    print("Loading raw data...")
    df = pd.read_csv('ipl_data.csv')

    # 1. Map legacy teams that evolved into current active franchises
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals', 
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad', 
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
    }
    for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
        if col in df.columns:
            df[col] = df[col].replace(team_mapping)

    # 2. STRICT FILTER: Only keep the 10 active IPL teams
    active_teams = [
        'Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
        'Royal Challengers Bengaluru', 'Delhi Capitals', 'Rajasthan Royals',
        'Sunrisers Hyderabad', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
    ]
    
    # Drop any row involving a defunct franchise
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

    print("Calculating historical metrics for active teams...")
    df['is_wicket'] = df['player_out'].notnull().astype(int)
    
    # Calculate Team Stats
    stats = pd.DataFrame()
    stats['sr'] = (df.groupby('batting_team')['runs_total'].sum() / df.groupby('batting_team')['ball'].count()) * 100
    stats['econ'] = df.groupby('bowling_team')['runs_total'].sum() / (df.groupby('bowling_team')['ball'].count() / 6)
    stats['bat_avg'] = df.groupby('batting_team')['runs_total'].sum() / df.groupby('batting_team')['is_wicket'].sum().replace(0, 1)
    stats['bowl_avg'] = df.groupby('bowling_team')['runs_total'].sum() / df.groupby('bowling_team')['is_wicket'].sum().replace(0, 1)

    print("Extracting match-level data...")
    matches = df[df['innings'] == 1].drop_duplicates(subset=['match_id']).copy()
    matches.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2'}, inplace=True)
    matches.dropna(subset=['match_won_by', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision'], inplace=True)

    # Map stats back to matches
    matches['team1_sr'] = matches['team1'].map(stats['sr'])
    matches['team2_sr'] = matches['team2'].map(stats['sr'])
    matches['team1_econ'] = matches['team1'].map(stats['econ'])
    matches['team2_econ'] = matches['team2'].map(stats['econ'])
    matches['team1_bat_avg'] = matches['team1'].map(stats['bat_avg'])
    matches['team2_bat_avg'] = matches['team2'].map(stats['bat_avg'])
    matches['team1_bowl_avg'] = matches['team1'].map(stats['bowl_avg'])
    matches['team2_bowl_avg'] = matches['team2'].map(stats['bowl_avg'])

    # Determine historical home flags for training data
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
        'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat', 'target'
    ]
    matches.dropna(subset=features, inplace=True)
    matches[features].to_csv('ml_ready_data.csv', index=False)
    stats.to_csv('team_stats.csv')
    print("Pre-processing complete. Filtered to 10 active teams.")

if __name__ == "__main__":
    clean_and_prepare_data()