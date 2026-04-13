"""
ipl_squads.py
Hard-coded IPL 2026 squads with realistic player stats
based on historical IPL data (2022-2025).
Generates: player_stats.csv, matchup_stats.csv
"""

import pandas as pd
import numpy as np
import os

# ── IPL 2026 Squads (hard-coded from official announcements) ─────────────────
# Each player: name, team, role, bat_sr, bat_avg, bat_runs, bat_balls,
#              boundary_pct, dot_pct, bowl_econ, bowl_avg, bowl_wkts, bowl_balls, wicket_rate

SQUADS = {
    "Chennai Super Kings": [
        # name, role, bat_sr, bat_avg, bat_runs, bowl_econ, bowl_avg, bowl_wkts
        ("Ruturaj Gaikwad",   "Batter",     142.0, 38.5, 2800, 0.0,  0.0,   0),
        ("MS Dhoni",          "Wicketkeeper",127.0, 28.0, 1800, 0.0,  0.0,   0),
        ("Sanju Samson",      "Wicketkeeper",148.0, 36.0, 2400, 0.0,  0.0,   0),
        ("Sarfaraz Khan",     "Batter",     145.0, 32.0,  800, 0.0,  0.0,   0),
        ("Ayush Mhatre",      "Batter",     152.0, 28.0,  420, 0.0,  0.0,   0),
        ("Dewald Brevis",     "Batter",     160.0, 30.0,  900, 0.0,  0.0,   0),
        ("Urvil Patel",       "Wicketkeeper",165.0, 25.0,  300, 0.0,  0.0,   0),
        ("Shivam Dube",       "All-rounder",148.0, 32.0, 1200, 8.8,  38.0,  18),
        ("Anshul Kamboj",     "All-rounder",120.0, 18.0,  180, 8.2,  30.0,  22),
        ("Ramakrishna Ghosh", "All-rounder",118.0, 16.0,  120, 8.5,  32.0,  10),
        ("Matthew Short",     "All-rounder",155.0, 28.0,  380, 8.0,  28.0,   8),
        ("Akeal Hosein",      "Bowler",     100.0, 12.0,   80, 7.2,  28.0,  35),
        ("Prashant Veer",     "All-rounder",115.0, 16.0,  200, 8.4,  30.0,  15),
        ("Kartik Sharma",     "Wicketkeeper",130.0, 20.0,  220, 0.0,  0.0,   0),
        ("Jamie Overton",     "Bowler",     115.0, 14.0,  100, 8.8,  34.0,  20),
        ("Khaleel Ahmed",     "Bowler",      90.0, 10.0,   60, 8.5,  28.0,  55),
        ("Noor Ahmad",        "Bowler",      75.0,  8.0,   40, 7.4,  24.0,  52),
        ("Mukesh Choudhary",  "Bowler",      85.0,  9.0,   50, 9.0,  32.0,  28),
        ("Spencer Johnson",   "Bowler",      95.0, 10.0,   60, 9.2,  30.0,  15),
        ("Shreyas Gopal",     "Bowler",      95.0, 10.0,   80, 7.8,  28.0,  30),
        ("Rahul Chahar",      "Bowler",      90.0,  9.0,   50, 7.6,  26.0,  40),
        ("Matt Henry",        "Bowler",      85.0, 10.0,   55, 8.8,  30.0,  18),
        ("Gurjapneet Singh",  "Bowler",      80.0,  8.0,   40, 9.0,  32.0,  10),
        ("Zak Foulkes",       "All-rounder",100.0, 12.0,   70, 8.6,  30.0,   8),
        ("Aman Khan",         "All-rounder",110.0, 14.0,   90, 8.5,  32.0,   6),
    ],
    "Delhi Capitals": [
        ("KL Rahul",          "Wicketkeeper",140.0, 42.0, 3200, 0.0,  0.0,   0),
        ("Prithvi Shaw",      "Batter",     155.0, 28.0, 1100, 0.0,  0.0,   0),
        ("Karun Nair",        "Batter",     138.0, 34.0, 1600, 0.0,  0.0,   0),
        ("David Miller",      "Batter",     155.0, 38.0, 2200, 0.0,  0.0,   0),
        ("Pathum Nissanka",   "Batter",     145.0, 36.0, 1400, 0.0,  0.0,   0),
        ("Ben Duckett",       "Wicketkeeper",148.0, 35.0, 1200, 0.0,  0.0,   0),
        ("Tristan Stubbs",    "Batter",     158.0, 32.0,  900, 0.0,  0.0,   0),
        ("Nitish Rana",       "Batter",     140.0, 30.0, 1800, 0.0,  0.0,   0),
        ("Axar Patel",        "All-rounder",148.0, 30.0, 1400, 7.2,  28.0,  85),
        ("Abishek Porel",     "Wicketkeeper",140.0, 26.0,  500, 0.0,  0.0,   0),
        ("Sameer Rizvi",      "Batter",     150.0, 28.0,  400, 0.0,  0.0,   0),
        ("Ashutosh Sharma",   "Batter",     158.0, 26.0,  350, 0.0,  0.0,   0),
        ("Auqib Dar",         "All-rounder",115.0, 16.0,  200, 8.4,  30.0,  12),
        ("Vipraj Nigam",      "All-rounder",110.0, 14.0,  150, 8.0,  28.0,  10),
        ("Ajay Mandal",       "All-rounder",112.0, 15.0,  130, 8.2,  29.0,   8),
        ("Mitchell Starc",    "Bowler",      90.0, 10.0,   70, 8.6,  26.0,  60),
        ("T. Natarajan",      "Bowler",      85.0,  9.0,   50, 8.8,  28.0,  55),
        ("Mukesh Kumar",      "Bowler",      80.0,  8.0,   45, 9.0,  30.0,  42),
        ("Kuldeep Yadav",     "Bowler",      85.0, 10.0,   60, 7.8,  22.0,  95),
        ("Kyle Jamieson",     "Bowler",      90.0, 12.0,   80, 8.4,  28.0,  22),
        ("Dushmantha Chameera","Bowler",     85.0,  9.0,   50, 9.2,  30.0,  20),
        ("Lungi Ngidi",       "Bowler",      80.0,  8.0,   45, 9.0,  28.0,  18),
        ("Tripurana Vijay",   "All-rounder",108.0, 13.0,  110, 8.6,  30.0,   6),
        ("Madhav Tiwari",     "All-rounder",105.0, 12.0,   90, 8.8,  32.0,   5),
        ("Sahil Parakh",      "Batter",     130.0, 20.0,  180, 0.0,  0.0,   0),
    ],
    "Gujarat Titans": [
        ("Shubman Gill",      "Batter",     145.0, 45.0, 2900, 0.0,  0.0,   0),
        ("Sai Sudharsan",     "Batter",     140.0, 42.0, 2200, 0.0,  0.0,   0),
        ("Jos Buttler",       "Wicketkeeper",158.0, 40.0, 3200, 0.0,  0.0,   0),
        ("Tom Banton",        "Batter",     162.0, 28.0,  600, 0.0,  0.0,   0),
        ("Shahrukh Khan",     "Batter",     160.0, 30.0,  900, 0.0,  0.0,   0),
        ("Kumar Kushagra",    "Wicketkeeper",130.0, 22.0,  280, 0.0,  0.0,   0),
        ("Anuj Rawat",        "Wicketkeeper",132.0, 22.0,  320, 0.0,  0.0,   0),
        ("Rahul Tewatia",     "All-rounder",160.0, 32.0, 1500, 8.5,  32.0,  20),
        ("Washington Sundar", "All-rounder",130.0, 28.0,  900, 7.4,  26.0,  65),
        ("Glenn Phillips",    "All-rounder",165.0, 30.0, 1000, 8.0,  30.0,  22),
        ("Jason Holder",      "All-rounder",130.0, 26.0,  700, 8.0,  28.0,  55),
        ("Nishant Sindhu",    "All-rounder",120.0, 20.0,  300, 8.4,  30.0,  15),
        ("Rashid Khan",       "Bowler",     135.0, 18.0,  500, 6.5,  20.0, 165),
        ("Mohammed Siraj",    "Bowler",      75.0,  8.0,   50, 8.5,  26.0,  90),
        ("Kagiso Rabada",     "Bowler",      80.0,  9.0,   60, 9.0,  28.0, 110),
        ("Prasidh Krishna",   "Bowler",      78.0,  8.0,   50, 8.8,  28.0,  75),
        ("Ishant Sharma",     "Bowler",      72.0,  7.0,   40, 9.0,  30.0,  40),
        ("Gurnoor Singh Brar","Bowler",      78.0,  8.0,   45, 8.6,  30.0,  25),
        ("Arshad Khan",       "Bowler",      82.0,  9.0,   55, 8.4,  28.0,  20),
        ("Manav Suthar",      "Bowler",      75.0,  7.0,   40, 7.8,  26.0,  18),
        ("Sai Kishore",       "Bowler",      78.0,  8.0,   45, 7.6,  24.0,  35),
        ("Jayant Yadav",      "Bowler",      90.0, 10.0,   70, 7.8,  26.0,  30),
        ("Ashok Sharma",      "Bowler",      72.0,  7.0,   38, 9.0,  32.0,  10),
        ("Luke Wood",         "Bowler",      88.0,  9.0,   55, 9.2,  30.0,  12),
        ("Kulwant Khejroliya","Bowler",      72.0,  7.0,   38, 9.2,  32.0,   8),
    ],
    "Kolkata Knight Riders": [
        ("Ajinkya Rahane",    "Batter",     130.0, 32.0, 2200, 0.0,  0.0,   0),
        ("Rinku Singh",       "Batter",     162.0, 38.0, 1600, 0.0,  0.0,   0),
        ("Cameron Green",     "All-rounder",148.0, 34.0, 1200, 8.5,  30.0,  28),
        ("Rachin Ravindra",   "All-rounder",142.0, 36.0, 1100, 8.0,  28.0,  20),
        ("Rovman Powell",     "All-rounder",170.0, 32.0, 1400, 8.5,  30.0,  10),
        ("Sunil Narine",      "All-rounder",155.0, 28.0, 2200, 6.8,  22.0, 175),
        ("Finn Allen",        "Wicketkeeper",175.0, 28.0, 1000, 0.0,  0.0,   0),
        ("Rahul Tripathi",    "Batter",     138.0, 30.0, 1500, 0.0,  0.0,   0),
        ("Manish Pandey",     "Batter",     132.0, 30.0, 2800, 0.0,  0.0,   0),
        ("Angkrish Raghuvanshi","Batter",   148.0, 28.0,  600, 0.0,  0.0,   0),
        ("Ramandeep Singh",   "Batter",     152.0, 24.0,  700, 0.0,  0.0,   0),
        ("Anukul Roy",        "All-rounder",118.0, 16.0,  250, 7.8,  28.0,  25),
        ("Varun Chakaravarthy","Bowler",     88.0, 10.0,   70, 7.2,  22.0, 105),
        ("Matheesha Pathirana","Bowler",     80.0,  9.0,   55, 8.8,  24.0,  65),
        ("Vaibhav Arora",     "Bowler",      82.0,  9.0,   55, 9.0,  28.0,  35),
        ("Umran Malik",       "Bowler",      78.0,  8.0,   45, 9.5,  30.0,  45),
        ("Navdeep Saini",     "Bowler",      80.0,  8.0,   50, 9.2,  30.0,  38),
        ("Tejasvi Singh",     "Wicketkeeper",128.0, 20.0,  240, 0.0,  0.0,   0),
        ("Tim Seifert",       "Wicketkeeper",158.0, 26.0,  480, 0.0,  0.0,   0),
        ("Sarthak Ranjan",    "All-rounder",110.0, 14.0,  120, 8.6,  30.0,   8),
        ("Daksh Kamra",       "All-rounder",108.0, 13.0,  100, 8.8,  32.0,   6),
        ("Prashant Solanki",  "Bowler",      78.0,  8.0,   45, 9.0,  30.0,  12),
        ("Kartik Tyagi",      "Bowler",      76.0,  8.0,   42, 9.2,  32.0,  18),
        ("Saurabh Dubey",     "Bowler",      75.0,  7.0,   38, 9.2,  32.0,   8),
        ("Blessing Muzarabani","Bowler",     72.0,  7.0,   38, 9.0,  30.0,  10),
    ],
    "Lucknow Super Giants": [
        ("Rishabh Pant",      "Wicketkeeper",165.0, 40.0, 3000, 0.0,  0.0,   0),
        ("Josh Inglis",       "Batter",     158.0, 34.0, 1200, 0.0,  0.0,   0),
        ("Aiden Markram",     "Batter",     145.0, 38.0, 1800, 8.5,  30.0,  18),
        ("Mitchell Marsh",    "All-rounder",152.0, 36.0, 2000, 8.8,  30.0,  30),
        ("Nicholas Pooran",   "Wicketkeeper",168.0, 34.0, 2200, 0.0,  0.0,   0),
        ("Akshat Raghuwanshi","Batter",     148.0, 28.0,  400, 0.0,  0.0,   0),
        ("Abdul Samad",       "Batter",     162.0, 28.0,  800, 0.0,  0.0,   0),
        ("Matthew Breetzke",  "Batter",     142.0, 32.0,  700, 0.0,  0.0,   0),
        ("Himmat Singh",      "Batter",     138.0, 26.0,  400, 0.0,  0.0,   0),
        ("Ayush Badoni",      "All-rounder",148.0, 30.0, 1200, 8.5,  30.0,  12),
        ("Shahbaz Ahamad",    "All-rounder",135.0, 28.0,  900, 7.8,  26.0,  38),
        ("Arshin Kulkarni",   "All-rounder",125.0, 22.0,  350, 8.2,  28.0,  15),
        ("George Linde",      "All-rounder",128.0, 22.0,  400, 7.6,  26.0,  28),
        ("Naman Tiwari",      "All-rounder",118.0, 18.0,  200, 8.4,  30.0,  12),
        ("Mohammed Shami",    "Bowler",      75.0,  8.0,   50, 8.8,  26.0,  90),
        ("Mayank Yadav",      "Bowler",      72.0,  7.0,   38, 8.2,  24.0,  35),
        ("Avesh Khan",        "Bowler",      80.0,  8.0,   52, 9.0,  28.0,  65),
        ("Anrich Nortje",     "Bowler",      78.0,  8.0,   45, 9.0,  28.0,  42),
        ("Arjun Tendulkar",   "Bowler",      75.0,  7.0,   40, 9.2,  30.0,  18),
        ("Mohsin Khan",       "Bowler",      72.0,  7.0,   38, 8.6,  26.0,  30),
        ("M. Siddharth",      "Bowler",      75.0,  8.0,   42, 7.8,  26.0,  22),
        ("Digvesh Rathi",     "Bowler",      70.0,  7.0,   35, 8.0,  26.0,  15),
        ("Prince Yadav",      "Bowler",      68.0,  6.0,   30, 8.4,  28.0,  10),
        ("Akash Singh",       "Bowler",      70.0,  7.0,   35, 8.6,  28.0,  12),
        ("Mukul Choudhary",   "Wicketkeeper",120.0, 18.0,  180, 0.0,  0.0,   0),
    ],
    "Mumbai Indians": [
        ("Rohit Sharma",      "Batter",     148.0, 42.0, 6200, 0.0,  0.0,   0),
        ("Surya Kumar Yadav", "Batter",     175.0, 38.0, 3200, 0.0,  0.0,   0),
        ("Tilak Varma",       "Batter",     148.0, 40.0, 1800, 0.0,  0.0,   0),
        ("Quinton De Kock",   "Wicketkeeper",152.0, 36.0, 2800, 0.0,  0.0,   0),
        ("Ryan Rickelton",    "Wicketkeeper",148.0, 34.0,  900, 0.0,  0.0,   0),
        ("Sherfane Rutherford","Batter",    165.0, 30.0,  700, 0.0,  0.0,   0),
        ("Hardik Pandya",     "All-rounder",152.0, 36.0, 2500, 9.0,  30.0,  75),
        ("Will Jacks",        "All-rounder",158.0, 32.0, 1000, 8.2,  28.0,  22),
        ("Mitchell Santner",  "All-rounder",125.0, 24.0,  700, 7.2,  24.0,  55),
        ("Corbin Bosch",      "All-rounder",140.0, 26.0,  500, 8.8,  30.0,  18),
        ("Naman Dhir",        "All-rounder",135.0, 24.0,  400, 8.5,  30.0,  12),
        ("Shardul Thakur",    "All-rounder",130.0, 22.0,  700, 9.0,  32.0,  55),
        ("Raj Bawa",          "All-rounder",128.0, 22.0,  380, 8.6,  30.0,  15),
        ("Robin Minz",        "Wicketkeeper",130.0, 20.0,  250, 0.0,  0.0,   0),
        ("Jasprit Bumrah",    "Bowler",      72.0,  7.0,   40, 6.5,  18.0, 160),
        ("Trent Boult",       "Bowler",      80.0,  8.0,   50, 8.2,  24.0, 140),
        ("Deepak Chahar",     "Bowler",      88.0,  9.0,   60, 8.0,  26.0,  75),
        ("Allah Ghazanfar",   "Bowler",      78.0,  8.0,   45, 7.5,  24.0,  35),
        ("Mayank Markande",   "Bowler",      80.0,  8.0,   48, 8.0,  26.0,  30),
        ("Ashwani Kumar",     "Bowler",      75.0,  7.0,   40, 9.0,  30.0,  15),
        ("Raghu Sharma",      "Bowler",      72.0,  7.0,   38, 8.8,  30.0,  10),
        ("Atharva Ankolekar", "All-rounder",112.0, 14.0,  120, 8.5,  30.0,   8),
        ("Danish Malewar",    "Batter",     130.0, 20.0,  180, 0.0,  0.0,   0),
        ("Mayank Rawat",      "All-rounder",118.0, 16.0,  150, 8.8,  32.0,   6),
        ("Mohammad Izhar",    "Bowler",      70.0,  7.0,   35, 9.0,  30.0,   8),
    ],
    "Punjab Kings": [
        ("Shreyas Iyer",      "Batter",     135.0, 38.0, 3500, 0.0,  0.0,   0),
        ("Prabhsimran Singh", "Wicketkeeper",158.0, 32.0, 1400, 0.0,  0.0,   0),
        ("Shashank Singh",    "Batter",     165.0, 34.0, 1200, 0.0,  0.0,   0),
        ("Nehal Wadhera",     "Batter",     148.0, 30.0,  900, 0.0,  0.0,   0),
        ("Harnoor Pannu",     "Batter",     140.0, 28.0,  400, 0.0,  0.0,   0),
        ("Pyla Avinash",      "Batter",     138.0, 26.0,  350, 0.0,  0.0,   0),
        ("Vishnu Vinod",      "Wicketkeeper",140.0, 26.0,  480, 0.0,  0.0,   0),
        ("Marcus Stoinis",    "All-rounder",155.0, 34.0, 2000, 9.0,  30.0,  38),
        ("Azmatullah Omarzai","All-rounder",148.0, 30.0,  900, 8.5,  28.0,  25),
        ("Marco Jansen",      "All-rounder",130.0, 26.0,  600, 8.5,  28.0,  45),
        ("Priyansh Arya",     "All-rounder",168.0, 28.0,  700, 8.8,  30.0,  10),
        ("Musheer Khan",      "All-rounder",132.0, 26.0,  550, 8.0,  28.0,  15),
        ("Harpreet Brar",     "All-rounder",118.0, 18.0,  350, 7.8,  26.0,  45),
        ("Mitch Owen",        "All-rounder",162.0, 28.0,  480, 8.8,  30.0,  10),
        ("Suryansh Shedge",   "All-rounder",138.0, 24.0,  320, 8.5,  30.0,  10),
        ("Cooper Connolly",   "All-rounder",132.0, 24.0,  380, 8.2,  28.0,  15),
        ("Ben Dwarshuis",     "All-rounder",118.0, 16.0,  200, 8.8,  30.0,  22),
        ("Arshdeep Singh",    "Bowler",      80.0,  8.0,   52, 8.0,  24.0,  85),
        ("Yuzvendra Chahal",  "Bowler",      82.0,  8.0,   55, 7.5,  22.0, 205),
        ("Lockie Ferguson",   "Bowler",      78.0,  8.0,   45, 9.0,  28.0,  42),
        ("Xavier Bartlett",   "Bowler",      80.0,  8.0,   50, 8.8,  28.0,  22),
        ("Vyshak Vijaykumar", "Bowler",      78.0,  7.0,   42, 9.0,  28.0,  28),
        ("Yash Thakur",       "Bowler",      75.0,  7.0,   40, 9.2,  30.0,  18),
        ("Vishal Nishad",     "Bowler",      70.0,  6.0,   32, 9.2,  32.0,   8),
        ("Pravin Dubey",      "Bowler",      72.0,  7.0,   35, 8.8,  30.0,  12),
    ],
    "Rajasthan Royals": [
        ("Yashasvi Jaiswal",  "Batter",     165.0, 42.0, 2800, 0.0,  0.0,   0),
        ("Vaibhav Sooryavanshi","Batter",   168.0, 30.0,  600, 0.0,  0.0,   0),
        ("Riyan Parag",       "Batter",     148.0, 38.0, 2200, 8.5,  30.0,  15),
        ("Shimron Hetmyer",   "Batter",     162.0, 34.0, 1800, 0.0,  0.0,   0),
        ("Dhruv Jurel",       "Wicketkeeper",142.0, 32.0, 1000, 0.0,  0.0,   0),
        ("Shubham Dubey",     "Batter",     140.0, 30.0,  800, 0.0,  0.0,   0),
        ("Lhuan-dre Pretorius","Batter",    150.0, 30.0,  600, 0.0,  0.0,   0),
        ("Donovan Ferreira",  "Wicketkeeper",145.0, 28.0,  450, 0.0,  0.0,   0),
        ("Ravindra Jadeja",   "All-rounder",140.0, 32.0, 2500, 7.5,  26.0, 155),
        ("Dasun Shanaka",     "All-rounder",145.0, 28.0,  800, 8.5,  28.0,  40),
        ("Yudhvir Singh Charak","All-rounder",128.0,22.0,  350, 8.5,  30.0,  22),
        ("Aman Rao",          "Batter",     132.0, 22.0,  250, 0.0,  0.0,   0),
        ("Jofra Archer",      "Bowler",      80.0,  9.0,   55, 7.8,  24.0,  80),
        ("Ravi Bishnoi",      "Bowler",      82.0,  8.0,   52, 7.5,  22.0,  90),
        ("Tushar Deshpande",  "Bowler",      78.0,  8.0,   45, 9.2,  30.0,  55),
        ("Kwena Maphaka",     "Bowler",      75.0,  7.0,   40, 8.8,  26.0,  25),
        ("Nandre Burger",     "Bowler",      75.0,  7.0,   40, 8.5,  26.0,  22),
        ("Sandeep Sharma",    "Bowler",      78.0,  8.0,   45, 8.8,  28.0,  50),
        ("Adam Milne",        "Bowler",      76.0,  7.0,   40, 9.0,  28.0,  28),
        ("Kuldeep Sen",       "Bowler",      74.0,  7.0,   38, 9.2,  30.0,  22),
        ("Sushant Mishra",    "Bowler",      72.0,  7.0,   35, 8.8,  28.0,  15),
        ("Ravi Singh",        "Wicketkeeper",125.0, 18.0,  200, 0.0,  0.0,   0),
        ("Vignesh Puthur",    "Bowler",      70.0,  6.0,   30, 9.0,  30.0,  10),
        ("Yash Raj Punja",    "Bowler",      68.0,  6.0,   28, 9.2,  32.0,   8),
        ("Brijesh Sharma",    "Bowler",      68.0,  6.0,   28, 9.4,  32.0,   6),
    ],
    "Royal Challengers Bengaluru": [
        ("Virat Kohli",       "Batter",     145.0, 48.0, 8700, 0.0,  0.0,   0),
        ("Rajat Patidar",     "Batter",     148.0, 38.0, 1800, 0.0,  0.0,   0),
        ("Phil Salt",         "Wicketkeeper",168.0, 32.0, 1400, 0.0,  0.0,   0),
        ("Devdutt Padikkal",  "Batter",     140.0, 34.0, 1600, 0.0,  0.0,   0),
        ("Tim David",         "All-rounder",175.0, 38.0, 1800, 8.8,  30.0,   5),
        ("Venkatesh Iyer",    "All-rounder",148.0, 32.0, 1500, 8.5,  30.0,  22),
        ("Jordan Cox",        "Batter",     145.0, 30.0,  500, 0.0,  0.0,   0),
        ("Jitesh Sharma",     "Wicketkeeper",158.0, 30.0, 1000, 0.0,  0.0,   0),
        ("Krunal Pandya",     "All-rounder",132.0, 28.0, 1800, 7.8,  26.0,  78),
        ("Jacob Bethell",     "All-rounder",145.0, 30.0,  700, 8.0,  28.0,  18),
        ("Romario Shepherd",  "All-rounder",158.0, 26.0,  800, 9.0,  30.0,  28),
        ("Mangesh Yadav",     "All-rounder",118.0, 18.0,  280, 8.0,  28.0,  18),
        ("Swapnil Singh",     "All-rounder",118.0, 18.0,  260, 8.2,  28.0,  15),
        ("Josh Hazlewood",    "Bowler",      72.0,  8.0,   45, 7.8,  22.0,  90),
        ("Bhuvneshwar Kumar", "Bowler",      80.0,  9.0,   55, 8.0,  24.0, 175),
        ("Nuwan Thushara",    "Bowler",      75.0,  7.0,   40, 8.5,  26.0,  35),
        ("Yash Dayal",        "Bowler",      76.0,  7.0,   42, 9.0,  28.0,  32),
        ("Rasikh Salam",      "Bowler",      72.0,  7.0,   38, 8.8,  28.0,  18),
        ("Suyash Sharma",     "Bowler",      74.0,  7.0,   40, 8.0,  26.0,  22),
        ("Jacob Duffy",       "Bowler",      70.0,  7.0,   35, 8.8,  28.0,  15),
        ("Abhinandan Singh",  "Bowler",      70.0,  6.0,   32, 9.0,  30.0,  10),
        ("Satvik Deswal",     "All-rounder",110.0, 14.0,  120, 8.6,  30.0,   8),
        ("Kanishk Chouhan",   "All-rounder",108.0, 13.0,  100, 8.8,  32.0,   6),
        ("Vihaan Malhotra",   "All-rounder",105.0, 12.0,   90, 8.8,  32.0,   5),
        ("Vicky Ostwal",      "All-rounder",100.0, 11.0,   80, 8.0,  28.0,  12),
    ],
    "Sunrisers Hyderabad": [
        ("Travis Head",       "Batter",     185.0, 42.0, 2800, 0.0,  0.0,   0),
        ("Abhishek Sharma",   "All-rounder",172.0, 36.0, 1600, 8.2,  28.0,  18),
        ("Ishan Kishan",      "Wicketkeeper",152.0, 34.0, 2200, 0.0,  0.0,   0),
        ("Heinrich Klaasen",  "Wicketkeeper",178.0, 42.0, 2400, 0.0,  0.0,   0),
        ("Liam Livingstone",  "All-rounder",162.0, 36.0, 2200, 8.5,  30.0,  35),
        ("Nitish Kumar Reddy","All-rounder",145.0, 32.0, 1100, 8.8,  30.0,  22),
        ("Aniket Verma",      "Batter",     148.0, 28.0,  600, 0.0,  0.0,   0),
        ("R Smaran",          "Batter",     138.0, 26.0,  400, 0.0,  0.0,   0),
        ("Kamindu Mendis",    "All-rounder",145.0, 38.0, 1200, 7.8,  26.0,  28),
        ("Harsh Dubey",       "All-rounder",118.0, 18.0,  250, 8.0,  26.0,  20),
        ("Harshal Patel",     "All-rounder",120.0, 20.0,  450, 8.5,  26.0, 100),
        ("Brydon Carse",      "All-rounder",138.0, 24.0,  550, 8.8,  28.0,  28),
        ("Pat Cummins",       "Bowler",      88.0, 10.0,   65, 8.8,  26.0, 120),
        ("Jaydev Unadkat",    "Bowler",      78.0,  8.0,   48, 8.5,  26.0,  60),
        ("Eshan Malinga",     "Bowler",      72.0,  7.0,   38, 9.0,  28.0,  20),
        ("Zeeshan Ansari",    "Bowler",      72.0,  7.0,   35, 8.0,  26.0,  15),
        ("Shivam Mavi",       "Bowler",      80.0,  8.0,   50, 9.2,  30.0,  42),
        ("Salil Arora",       "Wicketkeeper",120.0, 18.0,  200, 0.0,  0.0,   0),
        ("Shivang Kumar",     "All-rounder",108.0, 13.0,  100, 8.5,  30.0,   8),
        ("Krains Fuletra",    "Bowler",      68.0,  6.0,   28, 9.2,  32.0,   6),
        ("Praful Hinge",      "Bowler",      68.0,  6.0,   28, 9.2,  32.0,   6),
        ("Amit Kumar",        "Bowler",      68.0,  6.0,   28, 9.2,  32.0,   6),
        ("Onkar Tarmale",     "Bowler",      68.0,  6.0,   26, 9.4,  32.0,   5),
        ("Sakib Hussain",     "Bowler",      68.0,  6.0,   26, 9.4,  32.0,   5),
        ("David Payne",       "Bowler",      72.0,  7.0,   38, 9.0,  28.0,  12),
    ],
}

def generate_player_stats():
    rows = []
    np.random.seed(42)
    for team, players in SQUADS.items():
        for p in players:
            name, role, bat_sr, bat_avg, bat_runs, bowl_econ, bowl_avg, bowl_wkts = p
            bat_balls = int(bat_runs / bat_sr * 100) if bat_sr > 0 and bat_runs > 0 else 50
            # boundary_pct: estimated from SR (higher SR = more boundaries)
            boundary_pct = max(8.0, min(35.0, (bat_sr - 100) * 0.4 + 12.0))
            dot_pct = max(25.0, min(60.0, 75.0 - bat_sr * 0.25))
            # bowl_balls: from wickets and average
            bowl_balls = int(bowl_wkts * bowl_avg * 6 / bowl_econ) if bowl_wkts > 0 and bowl_econ > 0 else 0
            bowl_runs = int(bowl_econ * bowl_balls / 6) if bowl_balls > 0 else 0
            wicket_rate = bowl_wkts / bowl_balls if bowl_balls > 0 else 0.0

            rows.append({
                "player": name, "team": team, "role": role,
                "bat_sr": bat_sr, "bat_avg": bat_avg,
                "bat_runs": bat_runs, "bat_balls": bat_balls,
                "boundary_pct": round(boundary_pct, 1),
                "dot_pct": round(dot_pct, 1),
                "bowl_econ": bowl_econ, "bowl_avg": bowl_avg,
                "bowl_wkts": bowl_wkts, "bowl_balls": bowl_balls,
                "bowl_runs": bowl_runs, "wicket_rate": round(wicket_rate, 5),
            })

    df = pd.DataFrame(rows)
    df.to_csv("player_stats.csv", index=False)
    print(f"  -> player_stats.csv: {len(df)} players across {df['team'].nunique()} teams")
    return df


def generate_matchup_stats(player_df):
    """Generate realistic batsman vs bowler matchup data."""
    np.random.seed(42)
    rows = []
    batters = player_df[player_df["bat_runs"] > 200]["player"].tolist()
    bowlers = player_df[player_df["bowl_wkts"] > 10]["player"].tolist()

    for batter in batters:
        brow = player_df[player_df["player"] == batter].iloc[0]
        for bowler in bowlers[:30]:  # limit combinations
            bwrow = player_df[player_df["player"] == bowler].iloc[0]
            m_balls = int(np.random.exponential(18) + 4)
            base_sr = brow["bat_sr"] / 100.0
            bowl_factor = max(0.7, 1.0 - (bwrow["bowl_econ"] - 7.0) * 0.04)
            m_sr = round(base_sr * bowl_factor * 100 * np.random.uniform(0.85, 1.15), 1)
            m_runs = int(m_sr * m_balls / 100)
            dismiss_base = bwrow["wicket_rate"] if bwrow["wicket_rate"] > 0 else 0.05
            m_dismiss_prob = round(np.clip(dismiss_base * np.random.uniform(0.7, 1.4), 0.01, 0.25), 5)
            m_dismissals = int(m_balls * m_dismiss_prob)
            rows.append({
                "batter": batter, "bowler": bowler,
                "m_runs": m_runs, "m_balls": m_balls,
                "m_dismissals": m_dismissals,
                "m_sr": m_sr,
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