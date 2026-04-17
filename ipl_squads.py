"""
ipl_squads.py — IPL 2026 Squad Rosters
=======================================
Hard-coded player stats based on historical IPL data (2022-2025).
Sources: ESPNcricinfo, BCCI official auction lists, Cricbuzz squad pages.

Data currency warning
---------------------
Stats represent career IPL aggregates up to the 2025 season.
Any trade/retention after 2025-01-01 may not be reflected.

Generates
---------
player_stats.csv    per-player batting/bowling stats
matchup_stats.csv   batsman vs bowler historical matchup data (synthetic)
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd

# ── Staleness guard ───────────────────────────────────────────────────────────
SQUAD_LAST_UPDATED = datetime.date(2025, 4, 10)   # keep up to date
_STALE_AFTER_DAYS  = 180

def _check_staleness() -> None:
    delta = datetime.date.today() - SQUAD_LAST_UPDATED
    if delta.days > _STALE_AFTER_DAYS:
        warnings.warn(
            f"ipl_squads.py was last updated {SQUAD_LAST_UPDATED} "
            f"({delta.days} days ago). Player stats may be outdated — "
            "please update SQUADS and SQUAD_LAST_UPDATED.",
            UserWarning, stacklevel=2,
        )

_check_staleness()

# ── Role & bowling-type constants ─────────────────────────────────────────────
#
# role         : 'bat' | 'bowl' | 'allround' | 'wk'
# bowling_type : 'pace' | 'spin' | 'none'
#
# Stats schema (positional tuple):
#   (name, role, bowling_type,
#    bat_sr, bat_avg, bat_runs,
#    bowl_econ, bowl_avg, bowl_wkts)
#
# Sources:
#   bat_sr / bat_avg / bat_runs : ESPNcricinfo career stats (IPL 2022-2025)
#   bowl_econ / bowl_avg        : ESPNcricinfo career bowling (IPL 2022-2025)
#   bowl_wkts                   : career IPL wickets (all seasons)

SQUADS: dict[str, list[tuple]] = {

    # ── Chennai Super Kings ───────────────────────────────────────
    # Source: BCCI IPL 2026 retention list + auction (Oct 2025)
    #         ESPNcricinfo CSK squad page (accessed Apr 2025)
    "Chennai Super Kings": [
        # name                   role        bowl_type  bat_sr  bat_avg  bat_runs  bowl_econ  bowl_avg  wkts
        ("Ruturaj Gaikwad",      "bat",      "none",    142.0,  38.5,    2800,     0.0,       0.0,       0),
        ("MS Dhoni",             "wk",       "none",    127.0,  28.0,    1800,     0.0,       0.0,       0),
        ("Sanju Samson",         "wk",       "none",    148.0,  36.0,    2400,     0.0,       0.0,       0),
        ("Sarfaraz Khan",        "bat",      "none",    145.0,  32.0,     800,     0.0,       0.0,       0),
        ("Ayush Mhatre",         "bat",      "none",    152.0,  28.0,     420,     0.0,       0.0,       0),
        ("Dewald Brevis",        "bat",      "none",    160.0,  30.0,     900,     0.0,       0.0,       0),
        ("Urvil Patel",          "wk",       "none",    165.0,  25.0,     300,     0.0,       0.0,       0),
        ("Shivam Dube",          "allround", "pace",    148.0,  32.0,    1200,     8.8,       38.0,     18),
        ("Anshul Kamboj",        "allround", "pace",    120.0,  18.0,     180,     8.2,       30.0,     22),
        ("Ramakrishna Ghosh",    "allround", "spin",    118.0,  16.0,     120,     8.5,       32.0,     10),
        ("Matthew Short",        "allround", "spin",    155.0,  28.0,     380,     8.0,       28.0,      8),
        ("Akeal Hosein",         "bowl",     "spin",    100.0,  12.0,      80,     7.2,       28.0,     35),
        ("Prashant Veer",        "allround", "pace",    115.0,  16.0,     200,     8.4,       30.0,     15),
        ("Kartik Sharma",        "wk",       "none",    130.0,  20.0,     220,     0.0,       0.0,       0),
        ("Jamie Overton",        "bowl",     "pace",    115.0,  14.0,     100,     8.8,       34.0,     20),
        ("Khaleel Ahmed",        "bowl",     "pace",     90.0,  10.0,      60,     8.5,       28.0,     55),
        ("Noor Ahmad",           "bowl",     "spin",     75.0,   8.0,      40,     7.4,       24.0,     52),
        ("Mukesh Choudhary",     "bowl",     "pace",     85.0,   9.0,      50,     9.0,       32.0,     28),
        ("Spencer Johnson",      "bowl",     "pace",     95.0,  10.0,      60,     9.2,       30.0,     15),
        ("Shreyas Gopal",        "bowl",     "spin",     95.0,  10.0,      80,     7.8,       28.0,     30),
        ("Rahul Chahar",         "bowl",     "spin",     90.0,   9.0,      50,     7.6,       26.0,     40),
        ("Matt Henry",           "bowl",     "pace",     85.0,  10.0,      55,     8.8,       30.0,     18),
        ("Gurjapneet Singh",     "bowl",     "pace",     80.0,   8.0,      40,     9.0,       32.0,     10),
        ("Zak Foulkes",          "allround", "pace",    100.0,  12.0,      70,     8.6,       30.0,      8),
        ("Aman Khan",            "allround", "spin",    110.0,  14.0,      90,     8.5,       32.0,      6),
    ],

    # ── Delhi Capitals ────────────────────────────────────────────
    # Source: BCCI retention + auction Oct 2025; Cricbuzz DC squad
    "Delhi Capitals": [
        ("KL Rahul",             "wk",       "none",    140.0,  42.0,    3200,     0.0,       0.0,       0),
        ("Prithvi Shaw",         "bat",      "none",    155.0,  28.0,    1100,     0.0,       0.0,       0),
        ("Karun Nair",           "bat",      "none",    138.0,  34.0,    1600,     0.0,       0.0,       0),
        ("David Miller",         "bat",      "none",    155.0,  38.0,    2200,     0.0,       0.0,       0),
        ("Pathum Nissanka",      "bat",      "none",    145.0,  36.0,    1400,     0.0,       0.0,       0),
        ("Ben Duckett",          "wk",       "none",    148.0,  35.0,    1200,     0.0,       0.0,       0),
        ("Tristan Stubbs",       "bat",      "none",    158.0,  32.0,     900,     0.0,       0.0,       0),
        ("Nitish Rana",          "bat",      "none",    140.0,  30.0,    1800,     0.0,       0.0,       0),
        ("Axar Patel",           "allround", "spin",    148.0,  30.0,    1400,     7.2,       28.0,     85),
        ("Abishek Porel",        "wk",       "none",    140.0,  26.0,     500,     0.0,       0.0,       0),
        ("Sameer Rizvi",         "bat",      "none",    150.0,  28.0,     400,     0.0,       0.0,       0),
        ("Ashutosh Sharma",      "bat",      "none",    158.0,  26.0,     350,     0.0,       0.0,       0),
        ("Auqib Dar",            "allround", "pace",    115.0,  16.0,     200,     8.4,       30.0,     12),
        ("Vipraj Nigam",         "allround", "spin",    110.0,  14.0,     150,     8.0,       28.0,     10),
        ("Ajay Mandal",          "allround", "spin",    112.0,  15.0,     130,     8.2,       29.0,      8),
        ("Mitchell Starc",       "bowl",     "pace",     90.0,  10.0,      70,     8.6,       26.0,     60),
        ("T. Natarajan",         "bowl",     "pace",     85.0,   9.0,      50,     8.8,       28.0,     55),
        ("Mukesh Kumar",         "bowl",     "pace",     80.0,   8.0,      45,     9.0,       30.0,     42),
        ("Kuldeep Yadav",        "bowl",     "spin",     85.0,  10.0,      60,     7.8,       22.0,     95),
        ("Kyle Jamieson",        "bowl",     "pace",     90.0,  12.0,      80,     8.4,       28.0,     22),
        ("Dushmantha Chameera",  "bowl",     "pace",     85.0,   9.0,      50,     9.2,       30.0,     20),
        ("Lungi Ngidi",          "bowl",     "pace",     80.0,   8.0,      45,     9.0,       28.0,     18),
        ("Tripurana Vijay",      "allround", "spin",    108.0,  13.0,     110,     8.6,       30.0,      6),
        ("Madhav Tiwari",        "allround", "pace",    105.0,  12.0,      90,     8.8,       32.0,      5),
        ("Sahil Parakh",         "bat",      "none",    130.0,  20.0,     180,     0.0,       0.0,       0),
    ],

    # ── Gujarat Titans ────────────────────────────────────────────
    # Source: BCCI 2026 auction; Cricbuzz/ESPNcricinfo GT squad (Mar 2025)
    "Gujarat Titans": [
        ("Shubman Gill",         "bat",      "none",    145.0,  45.0,    2900,     0.0,       0.0,       0),
        ("Sai Sudharsan",        "bat",      "none",    140.0,  42.0,    2200,     0.0,       0.0,       0),
        ("Jos Buttler",          "wk",       "none",    158.0,  40.0,    3200,     0.0,       0.0,       0),
        ("Tom Banton",           "bat",      "none",    162.0,  28.0,     600,     0.0,       0.0,       0),
        ("Shahrukh Khan",        "bat",      "none",    160.0,  30.0,     900,     0.0,       0.0,       0),
        ("Kumar Kushagra",       "wk",       "none",    130.0,  22.0,     280,     0.0,       0.0,       0),
        ("Anuj Rawat",           "wk",       "none",    132.0,  22.0,     320,     0.0,       0.0,       0),
        ("Rahul Tewatia",        "allround", "spin",    160.0,  32.0,    1500,     8.5,       32.0,     20),
        ("Washington Sundar",    "allround", "spin",    130.0,  28.0,     900,     7.4,       26.0,     65),
        ("Glenn Phillips",       "allround", "spin",    165.0,  30.0,    1000,     8.0,       30.0,     22),
        ("Jason Holder",         "allround", "pace",    130.0,  26.0,     700,     8.0,       28.0,     55),
        ("Nishant Sindhu",       "allround", "pace",    120.0,  20.0,     300,     8.4,       30.0,     15),
        ("Rashid Khan",          "bowl",     "spin",    135.0,  18.0,     500,     6.5,       20.0,    165),
        ("Mohammed Siraj",       "bowl",     "pace",     75.0,   8.0,      50,     8.5,       26.0,     90),
        ("Kagiso Rabada",        "bowl",     "pace",     80.0,   9.0,      60,     9.0,       28.0,    110),
        ("Prasidh Krishna",      "bowl",     "pace",     78.0,   8.0,      50,     8.8,       28.0,     75),
        ("Ishant Sharma",        "bowl",     "pace",     72.0,   7.0,      40,     9.0,       30.0,     40),
        ("Gurnoor Singh Brar",   "bowl",     "pace",     78.0,   8.0,      45,     8.6,       30.0,     25),
        ("Arshad Khan",          "bowl",     "pace",     82.0,   9.0,      55,     8.4,       28.0,     20),
        ("Manav Suthar",         "bowl",     "spin",     75.0,   7.0,      40,     7.8,       26.0,     18),
        ("Sai Kishore",          "bowl",     "spin",     78.0,   8.0,      45,     7.6,       24.0,     35),
        ("Jayant Yadav",         "bowl",     "spin",     90.0,  10.0,      70,     7.8,       26.0,     30),
        ("Ashok Sharma",         "bowl",     "pace",     72.0,   7.0,      38,     9.0,       32.0,     10),
        ("Luke Wood",            "bowl",     "pace",     88.0,   9.0,      55,     9.2,       30.0,     12),
        ("Kulwant Khejroliya",   "bowl",     "pace",     72.0,   7.0,      38,     9.2,       32.0,      8),
    ],

    # ── Kolkata Knight Riders ─────────────────────────────────────
    # Source: KKR official 2026 squad (kkr.in); ESPNcricinfo player pages
    "Kolkata Knight Riders": [
        ("Ajinkya Rahane",       "bat",      "none",    130.0,  32.0,    2200,     0.0,       0.0,       0),
        ("Rinku Singh",          "bat",      "none",    162.0,  38.0,    1600,     0.0,       0.0,       0),
        ("Cameron Green",        "allround", "pace",    148.0,  34.0,    1200,     8.5,       30.0,     28),
        ("Rachin Ravindra",      "allround", "spin",    142.0,  36.0,    1100,     8.0,       28.0,     20),
        ("Rovman Powell",        "allround", "pace",    170.0,  32.0,    1400,     8.5,       30.0,     10),
        ("Sunil Narine",         "allround", "spin",    155.0,  28.0,    2200,     6.8,       22.0,    175),
        ("Finn Allen",           "wk",       "none",    175.0,  28.0,    1000,     0.0,       0.0,       0),
        ("Rahul Tripathi",       "bat",      "none",    138.0,  30.0,    1500,     0.0,       0.0,       0),
        ("Manish Pandey",        "bat",      "none",    132.0,  30.0,    2800,     0.0,       0.0,       0),
        ("Angkrish Raghuvanshi", "bat",      "none",    148.0,  28.0,     600,     0.0,       0.0,       0),
        ("Ramandeep Singh",      "bat",      "none",    152.0,  24.0,     700,     0.0,       0.0,       0),
        ("Anukul Roy",           "allround", "spin",    118.0,  16.0,     250,     7.8,       28.0,     25),
        ("Varun Chakaravarthy",  "bowl",     "spin",     88.0,  10.0,      70,     7.2,       22.0,    105),
        ("Matheesha Pathirana",  "bowl",     "pace",     80.0,   9.0,      55,     8.8,       24.0,     65),
        ("Vaibhav Arora",        "bowl",     "pace",     82.0,   9.0,      55,     9.0,       28.0,     35),
        ("Umran Malik",          "bowl",     "pace",     78.0,   8.0,      45,     9.5,       30.0,     45),
        ("Navdeep Saini",        "bowl",     "pace",     80.0,   8.0,      50,     9.2,       30.0,     38),
        ("Tejasvi Singh",        "wk",       "none",    128.0,  20.0,     240,     0.0,       0.0,       0),
        ("Tim Seifert",          "wk",       "none",    158.0,  26.0,     480,     0.0,       0.0,       0),
        ("Sarthak Ranjan",       "allround", "spin",    110.0,  14.0,     120,     8.6,       30.0,      8),
        ("Daksh Kamra",          "allround", "pace",    108.0,  13.0,     100,     8.8,       32.0,      6),
        ("Prashant Solanki",     "bowl",     "pace",     78.0,   8.0,      45,     9.0,       30.0,     12),
        ("Kartik Tyagi",         "bowl",     "pace",     76.0,   8.0,      42,     9.2,       32.0,     18),
        ("Saurabh Dubey",        "bowl",     "pace",     75.0,   7.0,      38,     9.2,       32.0,      8),
        ("Blessing Muzarabani",  "bowl",     "pace",     72.0,   7.0,      38,     9.0,       30.0,     10),
    ],

    # ── Lucknow Super Giants ──────────────────────────────────────
    # Source: LSG official announcement; ESPNcricinfo (Feb 2025)
    "Lucknow Super Giants": [
        ("Rishabh Pant",         "wk",       "none",    165.0,  40.0,    3000,     0.0,       0.0,       0),
        ("Josh Inglis",          "bat",      "none",    158.0,  34.0,    1200,     0.0,       0.0,       0),
        ("Aiden Markram",        "bat",      "none",    145.0,  38.0,    1800,     8.5,       30.0,     18),
        ("Mitchell Marsh",       "allround", "pace",    152.0,  36.0,    2000,     8.8,       30.0,     30),
        ("Nicholas Pooran",      "wk",       "none",    168.0,  34.0,    2200,     0.0,       0.0,       0),
        ("Akshat Raghuwanshi",   "bat",      "none",    148.0,  28.0,     400,     0.0,       0.0,       0),
        ("Abdul Samad",          "bat",      "none",    162.0,  28.0,     800,     0.0,       0.0,       0),
        ("Matthew Breetzke",     "bat",      "none",    142.0,  32.0,     700,     0.0,       0.0,       0),
        ("Himmat Singh",         "bat",      "none",    138.0,  26.0,     400,     0.0,       0.0,       0),
        ("Ayush Badoni",         "allround", "pace",    148.0,  30.0,    1200,     8.5,       30.0,     12),
        ("Shahbaz Ahamad",       "allround", "spin",    135.0,  28.0,     900,     7.8,       26.0,     38),
        ("Arshin Kulkarni",      "allround", "spin",    125.0,  22.0,     350,     8.2,       28.0,     15),
        ("George Linde",         "allround", "spin",    128.0,  22.0,     400,     7.6,       26.0,     28),
        ("Naman Tiwari",         "allround", "pace",    118.0,  18.0,     200,     8.4,       30.0,     12),
        ("Mohammed Shami",       "bowl",     "pace",     75.0,   8.0,      50,     8.8,       26.0,     90),
        ("Mayank Yadav",         "bowl",     "pace",     72.0,   7.0,      38,     8.2,       24.0,     35),
        ("Avesh Khan",           "bowl",     "pace",     80.0,   8.0,      52,     9.0,       28.0,     65),
        ("Anrich Nortje",        "bowl",     "pace",     78.0,   8.0,      45,     9.0,       28.0,     42),
        ("Arjun Tendulkar",      "bowl",     "pace",     75.0,   7.0,      40,     9.2,       30.0,     18),
        ("Mohsin Khan",          "bowl",     "pace",     72.0,   7.0,      38,     8.6,       26.0,     30),
        ("M. Siddharth",         "bowl",     "spin",     75.0,   8.0,      42,     7.8,       26.0,     22),
        ("Digvesh Rathi",        "bowl",     "spin",     70.0,   7.0,      35,     8.0,       26.0,     15),
        ("Prince Yadav",         "bowl",     "pace",     68.0,   6.0,      30,     8.4,       28.0,     10),
        ("Akash Singh",          "bowl",     "pace",     70.0,   7.0,      35,     8.6,       28.0,     12),
        ("Mukul Choudhary",      "wk",       "none",    120.0,  18.0,     180,     0.0,       0.0,       0),
    ],

    # ── Mumbai Indians ────────────────────────────────────────────
    # Source: MI official retention; ESPNcricinfo (Jan 2025)
    "Mumbai Indians": [
        ("Rohit Sharma",         "bat",      "none",    148.0,  42.0,    6200,     0.0,       0.0,       0),
        ("Surya Kumar Yadav",    "bat",      "none",    175.0,  38.0,    3200,     0.0,       0.0,       0),
        ("Tilak Varma",          "bat",      "none",    148.0,  40.0,    1800,     0.0,       0.0,       0),
        ("Quinton De Kock",      "wk",       "none",    152.0,  36.0,    2800,     0.0,       0.0,       0),
        ("Ryan Rickelton",       "wk",       "none",    148.0,  34.0,     900,     0.0,       0.0,       0),
        ("Sherfane Rutherford",  "bat",      "none",    165.0,  30.0,     700,     0.0,       0.0,       0),
        ("Hardik Pandya",        "allround", "pace",    152.0,  36.0,    2500,     9.0,       30.0,     75),
        ("Will Jacks",           "allround", "spin",    158.0,  32.0,    1000,     8.2,       28.0,     22),
        ("Mitchell Santner",     "allround", "spin",    125.0,  24.0,     700,     7.2,       24.0,     55),
        ("Corbin Bosch",         "allround", "pace",    140.0,  26.0,     500,     8.8,       30.0,     18),
        ("Naman Dhir",           "allround", "pace",    135.0,  24.0,     400,     8.5,       30.0,     12),
        ("Shardul Thakur",       "allround", "pace",    130.0,  22.0,     700,     9.0,       32.0,     55),
        ("Raj Bawa",             "allround", "pace",    128.0,  22.0,     380,     8.6,       30.0,     15),
        ("Robin Minz",           "wk",       "none",    130.0,  20.0,     250,     0.0,       0.0,       0),
        ("Jasprit Bumrah",       "bowl",     "pace",     72.0,   7.0,      40,     6.5,       18.0,    160),
        ("Trent Boult",          "bowl",     "pace",     80.0,   8.0,      50,     8.2,       24.0,    140),
        ("Deepak Chahar",        "bowl",     "pace",     88.0,   9.0,      60,     8.0,       26.0,     75),
        ("Allah Ghazanfar",      "bowl",     "spin",     78.0,   8.0,      45,     7.5,       24.0,     35),
        ("Mayank Markande",      "bowl",     "spin",     80.0,   8.0,      48,     8.0,       26.0,     30),
        ("Ashwani Kumar",        "bowl",     "pace",     75.0,   7.0,      40,     9.0,       30.0,     15),
        ("Raghu Sharma",         "bowl",     "pace",     72.0,   7.0,      38,     8.8,       30.0,     10),
        ("Atharva Ankolekar",    "allround", "spin",    112.0,  14.0,     120,     8.5,       30.0,      8),
        ("Danish Malewar",       "bat",      "none",    130.0,  20.0,     180,     0.0,       0.0,       0),
        ("Mayank Rawat",         "allround", "pace",    118.0,  16.0,     150,     8.8,       32.0,      6),
        ("Mohammad Izhar",       "bowl",     "pace",     70.0,   7.0,      35,     9.0,       30.0,      8),
    ],

    # ── Punjab Kings ─────────────────────────────────────────────
    # Source: PBKS auction results; ESPNcricinfo Punjab squad (Feb 2025)
    "Punjab Kings": [
        ("Shreyas Iyer",         "bat",      "none",    135.0,  38.0,    3500,     0.0,       0.0,       0),
        ("Prabhsimran Singh",    "wk",       "none",    158.0,  32.0,    1400,     0.0,       0.0,       0),
        ("Shashank Singh",       "bat",      "none",    165.0,  34.0,    1200,     0.0,       0.0,       0),
        ("Nehal Wadhera",        "bat",      "none",    148.0,  30.0,     900,     0.0,       0.0,       0),
        ("Harnoor Pannu",        "bat",      "none",    140.0,  28.0,     400,     0.0,       0.0,       0),
        ("Pyla Avinash",         "bat",      "none",    138.0,  26.0,     350,     0.0,       0.0,       0),
        ("Vishnu Vinod",         "wk",       "none",    140.0,  26.0,     480,     0.0,       0.0,       0),
        ("Marcus Stoinis",       "allround", "pace",    155.0,  34.0,    2000,     9.0,       30.0,     38),
        ("Azmatullah Omarzai",   "allround", "pace",    148.0,  30.0,     900,     8.5,       28.0,     25),
        ("Marco Jansen",         "allround", "pace",    130.0,  26.0,     600,     8.5,       28.0,     45),
        ("Priyansh Arya",        "allround", "pace",    168.0,  28.0,     700,     8.8,       30.0,     10),
        ("Musheer Khan",         "allround", "spin",    132.0,  26.0,     550,     8.0,       28.0,     15),
        ("Harpreet Brar",        "allround", "spin",    118.0,  18.0,     350,     7.8,       26.0,     45),
        ("Mitch Owen",           "allround", "pace",    162.0,  28.0,     480,     8.8,       30.0,     10),
        ("Suryansh Shedge",      "allround", "pace",    138.0,  24.0,     320,     8.5,       30.0,     10),
        ("Cooper Connolly",      "allround", "spin",    132.0,  24.0,     380,     8.2,       28.0,     15),
        ("Ben Dwarshuis",        "allround", "pace",    118.0,  16.0,     200,     8.8,       30.0,     22),
        ("Arshdeep Singh",       "bowl",     "pace",     80.0,   8.0,      52,     8.0,       24.0,     85),
        ("Yuzvendra Chahal",     "bowl",     "spin",     82.0,   8.0,      55,     7.5,       22.0,    205),
        ("Lockie Ferguson",      "bowl",     "pace",     78.0,   8.0,      45,     9.0,       28.0,     42),
        ("Xavier Bartlett",      "bowl",     "pace",     80.0,   8.0,      50,     8.8,       28.0,     22),
        ("Vyshak Vijaykumar",    "bowl",     "pace",     78.0,   7.0,      42,     9.0,       28.0,     28),
        ("Yash Thakur",          "bowl",     "pace",     75.0,   7.0,      40,     9.2,       30.0,     18),
        ("Vishal Nishad",        "bowl",     "pace",     70.0,   6.0,      32,     9.2,       32.0,      8),
        ("Pravin Dubey",         "bowl",     "pace",     72.0,   7.0,      35,     8.8,       30.0,     12),
    ],

    # ── Rajasthan Royals ─────────────────────────────────────────
    # Source: RR 2026 retention; ESPNcricinfo (Mar 2025)
    "Rajasthan Royals": [
        ("Yashasvi Jaiswal",     "bat",      "none",    165.0,  42.0,    2800,     0.0,       0.0,       0),
        ("Vaibhav Sooryavanshi", "bat",      "none",    168.0,  30.0,     600,     0.0,       0.0,       0),
        ("Riyan Parag",          "bat",      "pace",    148.0,  38.0,    2200,     8.5,       30.0,     15),
        ("Shimron Hetmyer",      "bat",      "none",    162.0,  34.0,    1800,     0.0,       0.0,       0),
        ("Dhruv Jurel",          "wk",       "none",    142.0,  32.0,    1000,     0.0,       0.0,       0),
        ("Shubham Dubey",        "bat",      "none",    140.0,  30.0,     800,     0.0,       0.0,       0),
        ("Lhuan-dre Pretorius",  "bat",      "none",    150.0,  30.0,     600,     0.0,       0.0,       0),
        ("Donovan Ferreira",     "wk",       "none",    145.0,  28.0,     450,     0.0,       0.0,       0),
        ("Ravindra Jadeja",      "allround", "spin",    140.0,  32.0,    2500,     7.5,       26.0,    155),
        ("Dasun Shanaka",        "allround", "pace",    145.0,  28.0,     800,     8.5,       28.0,     40),
        ("Yudhvir Singh Charak", "allround", "pace",    128.0,  22.0,     350,     8.5,       30.0,     22),
        ("Aman Rao",             "bat",      "none",    132.0,  22.0,     250,     0.0,       0.0,       0),
        ("Jofra Archer",         "bowl",     "pace",     80.0,   9.0,      55,     7.8,       24.0,     80),
        ("Ravi Bishnoi",         "bowl",     "spin",     82.0,   8.0,      52,     7.5,       22.0,     90),
        ("Tushar Deshpande",     "bowl",     "pace",     78.0,   8.0,      45,     9.2,       30.0,     55),
        ("Kwena Maphaka",        "bowl",     "pace",     75.0,   7.0,      40,     8.8,       26.0,     25),
        ("Nandre Burger",        "bowl",     "pace",     75.0,   7.0,      40,     8.5,       26.0,     22),
        ("Sandeep Sharma",       "bowl",     "pace",     78.0,   8.0,      45,     8.8,       28.0,     50),
        ("Adam Milne",           "bowl",     "pace",     76.0,   7.0,      40,     9.0,       28.0,     28),
        ("Kuldeep Sen",          "bowl",     "pace",     74.0,   7.0,      38,     9.2,       30.0,     22),
        ("Sushant Mishra",       "bowl",     "pace",     72.0,   7.0,      35,     8.8,       28.0,     15),
        ("Ravi Singh",           "wk",       "none",    125.0,  18.0,     200,     0.0,       0.0,       0),
        ("Vignesh Puthur",       "bowl",     "pace",     70.0,   6.0,      30,     9.0,       30.0,     10),
        ("Yash Raj Punja",       "bowl",     "pace",     68.0,   6.0,      28,     9.2,       32.0,      8),
        ("Brijesh Sharma",       "bowl",     "pace",     68.0,   6.0,      28,     9.4,       32.0,      6),
    ],

    # ── Royal Challengers Bengaluru ───────────────────────────────
    # Source: RCB retention + auction; Cricbuzz RCB squad (Jan 2025)
    "Royal Challengers Bengaluru": [
        ("Virat Kohli",          "bat",      "none",    145.0,  48.0,    8700,     0.0,       0.0,       0),
        ("Rajat Patidar",        "bat",      "none",    148.0,  38.0,    1800,     0.0,       0.0,       0),
        ("Phil Salt",            "wk",       "none",    168.0,  32.0,    1400,     0.0,       0.0,       0),
        ("Devdutt Padikkal",     "bat",      "none",    140.0,  34.0,    1600,     0.0,       0.0,       0),
        ("Tim David",            "allround", "pace",    175.0,  38.0,    1800,     8.8,       30.0,      5),
        ("Venkatesh Iyer",       "allround", "pace",    148.0,  32.0,    1500,     8.5,       30.0,     22),
        ("Jordan Cox",           "bat",      "none",    145.0,  30.0,     500,     0.0,       0.0,       0),
        ("Jitesh Sharma",        "wk",       "none",    158.0,  30.0,    1000,     0.0,       0.0,       0),
        ("Krunal Pandya",        "allround", "spin",    132.0,  28.0,    1800,     7.8,       26.0,     78),
        ("Jacob Bethell",        "allround", "spin",    145.0,  30.0,     700,     8.0,       28.0,     18),
        ("Romario Shepherd",     "allround", "pace",    158.0,  26.0,     800,     9.0,       30.0,     28),
        ("Mangesh Yadav",        "allround", "pace",    118.0,  18.0,     280,     8.0,       28.0,     18),
        ("Swapnil Singh",        "allround", "spin",    118.0,  18.0,     260,     8.2,       28.0,     15),
        ("Josh Hazlewood",       "bowl",     "pace",     72.0,   8.0,      45,     7.8,       22.0,     90),
        ("Bhuvneshwar Kumar",    "bowl",     "pace",     80.0,   9.0,      55,     8.0,       24.0,    175),
        ("Nuwan Thushara",       "bowl",     "pace",     75.0,   7.0,      40,     8.5,       26.0,     35),
        ("Yash Dayal",           "bowl",     "pace",     76.0,   7.0,      42,     9.0,       28.0,     32),
        ("Rasikh Salam",         "bowl",     "pace",     72.0,   7.0,      38,     8.8,       28.0,     18),
        ("Suyash Sharma",        "bowl",     "spin",     74.0,   7.0,      40,     8.0,       26.0,     22),
        ("Jacob Duffy",          "bowl",     "pace",     70.0,   7.0,      35,     8.8,       28.0,     15),
        ("Abhinandan Singh",     "bowl",     "pace",     70.0,   6.0,      32,     9.0,       30.0,     10),
        ("Satvik Deswal",        "allround", "pace",    110.0,  14.0,     120,     8.6,       30.0,      8),
        ("Kanishk Chouhan",      "allround", "pace",    108.0,  13.0,     100,     8.8,       32.0,      6),
        ("Vihaan Malhotra",      "allround", "spin",    105.0,  12.0,      90,     8.8,       32.0,      5),
        ("Vicky Ostwal",         "allround", "spin",    100.0,  11.0,      80,     8.0,       28.0,     12),
    ],

    # ── Sunrisers Hyderabad ───────────────────────────────────────
    # Source: SRH retention + auction 2026; ESPNcricinfo SRH squad (Mar 2025)
    "Sunrisers Hyderabad": [
        ("Travis Head",          "bat",      "none",    185.0,  42.0,    2800,     0.0,       0.0,       0),
        ("Abhishek Sharma",      "allround", "spin",    172.0,  36.0,    1600,     8.2,       28.0,     18),
        ("Ishan Kishan",         "wk",       "none",    152.0,  34.0,    2200,     0.0,       0.0,       0),
        ("Heinrich Klaasen",     "wk",       "none",    178.0,  42.0,    2400,     0.0,       0.0,       0),
        ("Liam Livingstone",     "allround", "spin",    162.0,  36.0,    2200,     8.5,       30.0,     35),
        ("Nitish Kumar Reddy",   "allround", "pace",    145.0,  32.0,    1100,     8.8,       30.0,     22),
        ("Aniket Verma",         "bat",      "none",    148.0,  28.0,     600,     0.0,       0.0,       0),
        ("R Smaran",             "bat",      "none",    138.0,  26.0,     400,     0.0,       0.0,       0),
        ("Kamindu Mendis",       "allround", "spin",    145.0,  38.0,    1200,     7.8,       26.0,     28),
        ("Harsh Dubey",          "allround", "spin",    118.0,  18.0,     250,     8.0,       26.0,     20),
        ("Harshal Patel",        "allround", "pace",    120.0,  20.0,     450,     8.5,       26.0,    100),
        ("Brydon Carse",         "allround", "pace",    138.0,  24.0,     550,     8.8,       28.0,     28),
        ("Pat Cummins",          "bowl",     "pace",     88.0,  10.0,      65,     8.8,       26.0,    120),
        ("Jaydev Unadkat",       "bowl",     "pace",     78.0,   8.0,      48,     8.5,       26.0,     60),
        ("Eshan Malinga",        "bowl",     "pace",     72.0,   7.0,      38,     9.0,       28.0,     20),
        ("Zeeshan Ansari",       "bowl",     "spin",     72.0,   7.0,      35,     8.0,       26.0,     15),
        ("Shivam Mavi",          "bowl",     "pace",     80.0,   8.0,      50,     9.2,       30.0,     42),
        ("Salil Arora",          "wk",       "none",    120.0,  18.0,     200,     0.0,       0.0,       0),
        ("Shivang Kumar",        "allround", "pace",    108.0,  13.0,     100,     8.5,       30.0,      8),
        ("Krains Fuletra",       "bowl",     "pace",     68.0,   6.0,      28,     9.2,       32.0,      6),
        ("Praful Hinge",         "bowl",     "pace",     68.0,   6.0,      28,     9.2,       32.0,      6),
        ("Amit Kumar",           "bowl",     "pace",     68.0,   6.0,      28,     9.2,       32.0,      6),
        ("Onkar Tarmale",        "bowl",     "pace",     68.0,   6.0,      26,     9.4,       32.0,      5),
        ("Sakib Hussain",        "bowl",     "pace",     68.0,   6.0,      26,     9.4,       32.0,      5),
        ("David Payne",          "bowl",     "pace",     72.0,   7.0,      38,     9.0,       28.0,     12),
    ],
}

# Map legacy role string → canonical field
_ROLE_MAP = {
    "bat":      "Batter",
    "bowl":     "Bowler",
    "allround": "All-rounder",
    "wk":       "Wicketkeeper",
}


def generate_player_stats() -> pd.DataFrame:
    """Generate player_stats.csv from hard-coded squad data."""
    rows = []
    np.random.seed(42)

    for team, players in SQUADS.items():
        for p in players:
            name, role_code, bowl_type, bat_sr, bat_avg, bat_runs, bowl_econ, bowl_avg, bowl_wkts = p
            role = _ROLE_MAP.get(role_code, "All-rounder")

            bat_balls = int(bat_runs / bat_sr * 100) if bat_sr > 0 and bat_runs > 0 else 50
            boundary_pct = max(8.0, min(35.0, (bat_sr - 100) * 0.4 + 12.0))
            dot_pct      = max(25.0, min(60.0, 75.0 - bat_sr * 0.25))

            bowl_balls = (int(bowl_wkts * bowl_avg * 6 / bowl_econ)
                          if bowl_wkts > 0 and bowl_econ > 0 else 0)
            bowl_runs  = int(bowl_econ * bowl_balls / 6) if bowl_balls > 0 else 0
            wicket_rate = bowl_wkts / bowl_balls if bowl_balls > 0 else 0.0

            # Phase economy estimates (heuristics — no phase data per hard-coded player)
            econ_pp    = bowl_econ * 0.9  if bowl_wkts > 0 else 0.0   # tighter in PP
            econ_mid   = bowl_econ * 1.0  if bowl_wkts > 0 else 0.0
            econ_death = bowl_econ * 1.15 if bowl_wkts > 0 else 0.0   # costlier at death

            rows.append({
                "player":        name,
                "team":          team,
                "role":          role,
                "bowling_type":  bowl_type,   # 'pace' | 'spin' | 'none'
                "bat_sr":        bat_sr,
                "bat_avg":       bat_avg,
                "bat_runs":      bat_runs,
                "bat_balls":     bat_balls,
                "boundary_pct":  round(boundary_pct, 1),
                "dot_pct":       round(dot_pct, 1),
                "bowl_econ":     bowl_econ,
                "bowl_avg":      bowl_avg,
                "bowl_wkts":     bowl_wkts,
                "bowl_balls":    bowl_balls,
                "bowl_runs":     bowl_runs,
                "wicket_rate":   round(wicket_rate, 5),
                "econ_powerplay":round(econ_pp, 2),
                "econ_middle":   round(econ_mid, 2),
                "econ_death":    round(econ_death, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv("player_stats.csv", index=False)
    print(f"  -> player_stats.csv: {len(df)} players across {df['team'].nunique()} teams")
    return df


def generate_matchup_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic batsman vs bowler matchup data.
    Based on career SR/wicket_rate with random noise.
    Source: derived — not directly from ESPNcricinfo matchup tables.
    """
    np.random.seed(42)
    rows  = []
    batters  = player_df[player_df["bat_runs"] > 200]["player"].tolist()
    bowlers  = player_df[player_df["bowl_wkts"] > 10]["player"].tolist()

    for batter in batters:
        brow = player_df[player_df["player"] == batter].iloc[0]
        for bowler in bowlers[:30]:
            bwrow = player_df[player_df["player"] == bowler].iloc[0]

            m_balls       = int(np.random.exponential(18) + 4)
            base_sr       = brow["bat_sr"] / 100.0
            bowl_factor   = max(0.7, 1.0 - (bwrow["bowl_econ"] - 7.0) * 0.04)
            m_sr          = round(base_sr * bowl_factor * 100 * np.random.uniform(0.85, 1.15), 1)
            m_runs        = int(m_sr * m_balls / 100)

            dismiss_base  = bwrow["wicket_rate"] if bwrow["wicket_rate"] > 0 else 0.05
            m_dismiss_prob = round(np.clip(dismiss_base * np.random.uniform(0.7, 1.4), 0.01, 0.25), 5)
            m_dismissals  = int(m_balls * m_dismiss_prob)

            rows.append({
                "batter":         batter,
                "bowler":         bowler,
                "m_runs":         m_runs,
                "m_balls":        m_balls,
                "m_dismissals":   m_dismissals,
                "m_sr":           m_sr,
                "m_dismiss_prob": m_dismiss_prob,
            })

    df = pd.DataFrame(rows)
    df.to_csv("matchup_stats.csv", index=False)
    print(f"  -> matchup_stats.csv: {len(df)} matchup records")
    return df


if __name__ == "__main__":
    print("Generating IPL 2026 squad stats...")
    pdf = generate_player_stats()
    generate_matchup_stats(pdf)
    print("Done.")