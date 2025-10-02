# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 00:15:38 2025

@author: AzureRogue
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import re

# Load decklists and standardize player names 
# UPDATE THIS ONCE MELEE.GG API CAN BE USED
decklists = pd.read_excel("PT_EOE_data.xlsx", sheet_name="Simplified_Decklists", 
                          engine="openpyxl")
decklists['Player'] = decklists['Pilot'].apply(lambda x: " ".join(reversed(x.split(", "))) if ", " in x else x.strip())
decklists = decklists[['Player', 'Deck']]

# Compile match results from rounds 1-16
# UPDATE THIS ONCE MELEE.GG API CAN BE USED
match_data = []
for round_num in range(1,17):
    sheet_name = f"r{round_num}"
    df = pd.read_excel("PT_EOE_data.xlsx", sheet_name=sheet_name, 
                       engine="openpyxl", skiprows=1)
    df.columns = ['Player_A', 'vs', 'Player_B', 'Result']
    df = df.dropna(subset=['Player_A', 'Player_B', 'Result'])
    df = df[~df['Result'].str.contains('Draw|forfeited|awarded|assigned|bye', 
                                       case=False, na=False)]
    
    for _, row in df.iterrows():
        result = row['Result']
        winner_match = re.match(r"(.+?) won", result)
        if winner_match:
            winner = winner_match.group(1).strip()
            player_a = row['Player_A'].strip()
            player_b = row['Player_B'].strip()
            if winner == player_a:
                outcome = 1
            elif winner == player_b:
                outcome = 0
            else:
                continue
            match_data.append({'Player_A': player_a, 'Player_B': player_b, 
                               'Winner': winner, 'Outcome': outcome})

matches_df = pd.DataFrame(match_data)
# SOMETHING BELOW HERE IS WRONG - FIX LATER
# Merge deck info with match results
deck_map = dict(zip(decklists['Player'], decklists['Deck']))
matches_df['Deck_A'] = matches_df['Player_A'].map(deck_map)
matches_df['Deck_B'] = matches_df['Player_B'].map(deck_map)

# Prepare for modeling
players = pd.Index(pd.concat([matches_df['Player_A'], matches_df['Player_B']]).unique())
decks = pd.Index(pd.concat([matches_df['Deck_A'], matches_df['Deck_B']]).unique())