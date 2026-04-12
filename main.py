from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from xgboost import XGBClassifier
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

print("Loading pre-processed data...")
matches = pd.read_csv('ml_ready_data.csv')
stats = pd.read_csv('team_stats.csv', index_col=0)
h2h_df = pd.read_csv('h2h_stats.csv')

h2h_lookup = {}
for _, row in h2h_df.iterrows():
    pair = tuple(sorted([row['team_A'], row['team_B']]))
    h2h_lookup[pair] = row['team_A_h2h_win_pct']

features = [
    'team1_sr', 'team2_sr', 'team1_econ', 'team2_econ', 
    'team1_bat_avg', 'team2_bat_avg', 'team1_bowl_avg', 'team2_bowl_avg',
    'team1_is_home', 'team2_is_home', 'toss_winner_is_team1', 'toss_decision_bat',
    'team1_h2h_win_pct'
]

print("Training XGBoost model...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(matches[features], matches['target'])
print("API Ready!")

class MatchRequest(BaseModel):
    team1: str
    team2: str
    team1_venue_status: str 
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

    pair = tuple(sorted([t1, t2]))
    team_A_win_pct = h2h_lookup.get(pair, 0.5)
    t1_h2h = team_A_win_pct if t1 == pair[0] else (1 - team_A_win_pct)

    t1_home = 1 if data.team1_venue_status == "home" else 0
    t2_home = 1 if data.team1_venue_status == "away" else 0

    input_data = pd.DataFrame([{
        'team1_sr': stats.loc[t1, 'sr'], 'team2_sr': stats.loc[t2, 'sr'],
        'team1_econ': stats.loc[t1, 'econ'], 'team2_econ': stats.loc[t2, 'econ'],
        'team1_bat_avg': stats.loc[t1, 'bat_avg'], 'team2_bat_avg': stats.loc[t2, 'bat_avg'],
        'team1_bowl_avg': stats.loc[t1, 'bowl_avg'], 'team2_bowl_avg': stats.loc[t2, 'bowl_avg'],
        'team1_is_home': t1_home, 'team2_is_home': t2_home,
        'toss_winner_is_team1': 1 if data.toss_winner == t1 else 0,
        'toss_decision_bat': 1 if data.toss_decision == 'bat' else 0,
        'team1_h2h_win_pct': t1_h2h
    }])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0] 
    
    winner = t1 if prediction == 1 else t2
    win_prob = float(probabilities[1] if prediction == 1 else probabilities[0]) * 100
    
    return {
        "winner": winner,
        "probability": round(win_prob, 1),
        "team1_stats": {
            "sr": round(stats.loc[t1, 'sr'], 1),
            "bat_avg": round(stats.loc[t1, 'bat_avg'], 1),
            "econ": round(stats.loc[t1, 'econ'], 1),
            "h2h": round(t1_h2h * 100, 1)
        },
        "team2_stats": {
            "sr": round(stats.loc[t2, 'sr'], 1),
            "bat_avg": round(stats.loc[t2, 'bat_avg'], 1),
            "econ": round(stats.loc[t2, 'econ'], 1),
            "h2h": round((1 - t1_h2h) * 100, 1)
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)