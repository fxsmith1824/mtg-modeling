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
# The code below is better used on the round results sheets to convert lastname, firstname to decklists format
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
            # It is easier to convert round result names into decklists format
            winner = " ".join(reversed(winner.split(", ")))
            player_a = " ".join(reversed(player_a.split(", ")))
            player_b = " ".join(reversed(player_b.split(", ")))
            match_data.append({'Player_A': player_a, 'Player_B': player_b, 
                               'Winner': winner, 'Outcome': outcome, 'Round': round_num})

matches_df = pd.DataFrame(match_data)

# Merge deck info with match results
deck_map = dict(zip(decklists['Player'], decklists['Deck']))
matches_df['Deck_A'] = matches_df['Player_A'].map(deck_map)
matches_df['Deck_B'] = matches_df['Player_B'].map(deck_map)

# Replace rounds 1-3 and 9-11 decks with "draft"
draft_rounds = [1, 2, 3, 9, 10, 11]
matches_df.loc[matches_df['Round'].isin(draft_rounds), 'Deck_A'] = 'DRAFT'
matches_df.loc[matches_df['Round'].isin(draft_rounds), 'Deck_B'] = 'DRAFT'

# Below was just a sanity check to make sure data looked reasonable due to
# variable explorer issue in Spyder 5.4.5
# matches_df.to_csv('test.csv', index=False)

# Split data
draft_df = matches_df[matches_df["Round"].isin(draft_rounds)].copy()
constructed_df = matches_df[~matches_df["Round"].isin(draft_rounds)].copy()

# Encode players and decks
players = pd.Index(pd.concat([matches_df["Player_A"], matches_df["Player_B"]]).unique())
player_to_idx = {name: i for i, name in enumerate(players)}
draft_df["player_A_idx"] = draft_df["Player_A"].map(player_to_idx)
draft_df["player_B_idx"] = draft_df["Player_B"].map(player_to_idx)

# Model 1: Draft rounds (player skill only)
with pm.Model() as draft_model:
    player_skill = pm.Normal("player_skill", mu=0, sigma=1, shape=len(players))
    skill_diff = player_skill[draft_df["player_A_idx"].values] - player_skill[draft_df["player_B_idx"].values]
    win_prob = pm.Deterministic("win_prob", pm.math.sigmoid(skill_diff))
    outcome = pm.Bernoulli("outcome", p=win_prob, observed=draft_df["Outcome"].values)
    draft_trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

# Extract posterior summaries
player_means = draft_trace.posterior["player_skill"].mean(dim=["chain", "draw"]).values
player_stds = draft_trace.posterior["player_skill"].std(dim=["chain", "draw"]).values
player_hdi = az.hdi(draft_trace, var_names=["player_skill"], hdi_prob=0.95)["player_skill"]

# Encode constructed data
decks = pd.Index(pd.concat([constructed_df["Deck_A"], constructed_df["Deck_B"]]).unique())
deck_to_idx = {name: i for i, name in enumerate(decks)}
constructed_df["player_A_idx"] = constructed_df["Player_A"].map(player_to_idx)
constructed_df["player_B_idx"] = constructed_df["Player_B"].map(player_to_idx)
constructed_df["deck_A_idx"] = constructed_df["Deck_A"].map(deck_to_idx)
constructed_df["deck_B_idx"] = constructed_df["Deck_B"].map(deck_to_idx)

# Model 2: Constructed rounds (player skill + deck strength)
with pm.Model() as constructed_model:
    player_skill = pm.Normal("player_skill", mu=player_means, sigma=player_stds, shape=len(players))
    deck_strength = pm.Normal("deck_strength", mu=0, sigma=1, shape=len(decks))
    skill_diff = (player_skill[constructed_df["player_A_idx"].values] + deck_strength[constructed_df["deck_A_idx"].values]) - \
                 (player_skill[constructed_df["player_B_idx"].values] + deck_strength[constructed_df["deck_B_idx"].values])
    win_prob = pm.Deterministic("win_prob", pm.math.sigmoid(skill_diff))
    outcome = pm.Bernoulli("outcome", p=win_prob, observed=constructed_df["Outcome"].values)
    constructed_trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

# Extract final summaries
player_summary = az.summary(constructed_trace, var_names=["player_skill"], round_to=4)
deck_summary = az.summary(constructed_trace, var_names=["deck_strength"], round_to=4)

# Add names
player_summary["Player"] = players
deck_summary["Deck"] = decks

# Save to CSV
player_summary.to_csv("player_skill_estimates.csv", index=False)
deck_summary.to_csv("deck_strength_estimates.csv", index=False)
