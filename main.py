from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = FastAPI()
templates = Jinja2Templates(directory="templates")

print("Loading pre-processed data...")
matches = pd.read_csv('ml_ready_data.csv')
stats = pd.read_csv('team_stats.csv', index_col=0)

features = [
    'team1_sr', 'team2_sr', 'team1_econ', 'team2_econ', 
    'team1_bat_avg', 'team2_bat_avg', 'team1_bowl_avg', 'team2_bowl_avg',
    'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat'
]

print("Training Gradient Boosting model...")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(matches[features], matches['target'])
print("API Ready!")

class MatchRequest(BaseModel):
    team1: str
    team2: str
    team1_venue_status: str # "home", "away", or "neutral"
    toss_winner: str
    toss_decision: str

@app.get("/")
async def home(request: Request):
    teams = sorted(stats.index.tolist())
    return templates.TemplateResponse("index.html", {"request": request, "teams": teams})

@app.post("/predict")
async def predict(data: MatchRequest):
    t1, t2 = data.team1, data.team2
    if t1 == t2: 
        return {"error": "Teams must be different"}

    t1_home = 1 if data.team1_venue_status == "home" else 0
    t2_home = 1 if data.team1_venue_status == "away" else 0

    input_data = pd.DataFrame([{
        'team1_sr': stats.loc[t1, 'sr'], 'team2_sr': stats.loc[t2, 'sr'],
        'team1_econ': stats.loc[t1, 'econ'], 'team2_econ': stats.loc[t2, 'econ'],
        'team1_bat_avg': stats.loc[t1, 'bat_avg'], 'team2_bat_avg': stats.loc[t2, 'bat_avg'],
        'team1_bowl_avg': stats.loc[t1, 'bowl_avg'], 'team2_bowl_avg': stats.loc[t2, 'bowl_avg'],
        'team1_is_home': t1_home, 'team2_is_home': t2_home,
        'toss_winner_is_team1': 1 if data.toss_winner == t1 else 0,
        'toss_decision_bat': 1 if data.toss_decision == 'bat' else 0
    }])
    
    # Get raw prediction and probability array
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0] 
    
    # probabilities[1] = Team 1 win chance, probabilities[0] = Team 2 win chance
    winner = t1 if prediction == 1 else t2
    win_prob = probabilities[1] if prediction == 1 else probabilities[0]
    
    return {
        "winner": winner,
        "probability": round(win_prob * 100, 1) # Round to 1 decimal place
    }