from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- 1. Data Processing & Model Training ---
print("Loading and cleaning data...")
df = pd.read_csv('ipl_data.csv')

# Exhaustive Team Cleaning
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Pune Warriors': 'Rising Pune Supergiant',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}
for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
    df[col] = df[col].replace(team_mapping)

# Exhaustive Venue Cleaning
venue_mapping = {
    'M.Chinnaswamy Stadium': 'M Chinnaswamy Stadium',
    'M Chinnaswamy Stadium, Bengaluru': 'M Chinnaswamy Stadium',
    'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
    'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
    'Rajiv Gandhi Intl. Cricket Stadium': 'Rajiv Gandhi International Stadium',
    'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
    'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
    'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
    'Eden Gardens, Kolkata': 'Eden Gardens',
    'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
    'Feroz Shah Kotla': 'Arun Jaitley Stadium'
}
df['venue'] = df['venue'].replace(venue_mapping)

# Calculate Historical Team Stats
team_sr = (df.groupby('batting_team')['runs_total'].sum() / df.groupby('batting_team')['ball'].count()) * 100
team_econ = df.groupby('bowling_team')['runs_total'].sum() / (df.groupby('bowling_team')['ball'].count() / 6)

# Extract Match Data
matches = df[df['innings'] == 1].drop_duplicates(subset=['match_id']).copy()
matches.rename(columns={'batting_team': 'team1', 'bowling_team': 'team2'}, inplace=True)
matches.dropna(subset=['match_won_by', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision'], inplace=True)

# Map Stats
matches['team1_sr'] = matches['team1'].map(team_sr)
matches['team2_sr'] = matches['team2'].map(team_sr)
matches['team1_econ'] = matches['team1'].map(team_econ)
matches['team2_econ'] = matches['team2'].map(team_econ)

# Map Home Advantage
venue_home_team = {
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
    'Maharashtra Cricket Association Stadium': 'Rising Pune Supergiant'
}
matches['venue_home_team'] = matches['venue'].map(venue_home_team)
matches['team1_is_home'] = (matches['team1'] == matches['venue_home_team']).astype(int)
matches['team2_is_home'] = (matches['team2'] == matches['venue_home_team']).astype(int)

# Categorical Encodes
matches['toss_winner_is_team1'] = (matches['toss_winner'] == matches['team1']).astype(int)
matches['toss_decision_bat'] = (matches['toss_decision'] == 'bat').astype(int)
matches['target'] = (matches['match_won_by'] == matches['team1']).astype(int)

# Filter and Train
features = ['team1_sr', 'team2_sr', 'team1_econ', 'team2_econ', 'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat']
matches.dropna(subset=features, inplace=True)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(matches[features], matches['target'])
print("API Ready!")

# --- 2. API Endpoints ---
class MatchRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str

@app.get("/")
async def home(request: Request):
    # .unique() guarantees no duplicates are sent to the HTML dropdowns
    teams = sorted(matches['team1'].unique().tolist())
    venues = sorted(matches['venue'].unique().tolist())
    return templates.TemplateResponse("index.html", {"request": request, "teams": teams, "venues": venues})

@app.post("/predict")
async def predict(data: MatchRequest):
    t1, t2 = data.team1, data.team2
    if t1 == t2: 
        return {"error": "Teams must be different"}

    home_team = venue_home_team.get(data.venue, "Neutral")
    
    input_data = pd.DataFrame([{
        'team1_sr': team_sr.get(t1, 130), 
        'team2_sr': team_sr.get(t2, 130),
        'team1_econ': team_econ.get(t1, 8.0),
        'team2_econ': team_econ.get(t2, 8.0),
        'team1_is_home': 1 if t1 == home_team else 0,
        'team2_is_home': 1 if t2 == home_team else 0,
        'toss_winner_is_team1': 1 if data.toss_winner == t1 else 0,
        'toss_decision_bat': 1 if data.toss_decision == 'bat' else 0
    }])
    
    prediction = model.predict(input_data)[0]
    return {"winner": t1 if prediction == 1 else t2}