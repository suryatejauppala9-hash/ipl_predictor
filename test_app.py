import urllib.request
import urllib.parse
import json
import ssl

base_url = "http://127.0.0.1:8000"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def test_endpoint(name, url, method="GET", json_data=None):
    try:
        req_url = base_url + url
        req = urllib.request.Request(req_url, method=method)
        if json_data:
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(json_data).encode('utf-8')
            req.data = data
        
        with urllib.request.urlopen(req, context=ctx) as res:
            print(f"[{name}] {method} {url} - Status: {res.status}")
    except urllib.error.HTTPError as e:
        print(f"[{name}] {method} {url} - Error: {e.code} {e.read().decode('utf-8')[:200]}")
    except Exception as e:
        print(f"[{name}] {method} {url} - FAILED: {str(e)}")

team1 = "Chennai Super Kings"
team2 = "Mumbai Indians"

team1_enc = urllib.parse.quote(team1)
team2_enc = urllib.parse.quote(team2)
venue_enc = urllib.parse.quote("Wankhede Stadium, Mumbai")

test_endpoint("Root", "/")
test_endpoint("Predict GET", f"/predict?team1={team1_enc}&team2={team2_enc}&venue={venue_enc}&toss_winner={team1_enc}&toss_decision=field")
test_endpoint("Squad", f"/squad/{team1_enc}")
test_endpoint("Playing 11", f"/playing11/{team1_enc}")
test_endpoint("Ideal XI", f"/ideal-xi/{team1_enc}?style=balanced")
test_endpoint("Fantasy XI", f"/fantasy-xi?team1={team1_enc}&team2={team2_enc}")
test_endpoint("Player Stats", f"/player-stats?player={urllib.parse.quote('MS Dhoni')}")
test_endpoint("Model Info", "/model-info")

sim_data = {
    "t1": team1,
    "t2": team2,
    "venue": "Wankhede Stadium, Mumbai",
    "sim_count": 10
}
test_endpoint("Simulate Stream", "/simulate-stream", method="POST", json_data=sim_data)

custom_sim_data = {
    "t1": team1,
    "t2": team2,
    "venue": "Wankhede Stadium, Mumbai",
    "t1_xi": ["MS Dhoni", "Ruturaj Gaikwad", "Ravindra Jadeja", "Deepak Chahar", "Moeen Ali", "Shivam Dube", "Ambati Rayudu", "Devon Conway", "Maheesh Theekshana", "Matheesha Pathirana", "Tushar Deshpande"],
    "t2_xi": ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Tilak Varma", "Hardik Pandya", "Tim David", "Jasprit Bumrah", "Piyush Chawla", "Gerald Coetzee", "Romario Shepherd", "Mohammad Nabi"],
    "sim_count": 10
}
test_endpoint("Simulate Custom", "/simulate-custom", method="POST", json_data=custom_sim_data)
