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
# decklists['Player'] = decklists['Pilot'].apply(lambda x: " ".join(reversed(x.split(", "))) if ", " in x else x.strip())
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

# Prepare for modeling
players = pd.Index(pd.concat([matches_df['Player_A'], matches_df['Player_B']]).unique())
decks = pd.Index(pd.concat([matches_df['Deck_A'], matches_df['Deck_B']]).unique())

player_to_idx = {name: i for i, name in enumerate(players)}
deck_to_idx = {name: i for i, name in enumerate(decks)}

matches_df["player_A_idx"] = matches_df["Player_A"].map(player_to_idx)
matches_df["player_B_idx"] = matches_df["Player_B"].map(player_to_idx)
matches_df["deck_A_idx"] = matches_df["Deck_A"].map(deck_to_idx)
matches_df["deck_B_idx"] = matches_df["Deck_B"].map(deck_to_idx)

# Build Bayesian model
with pm.Model() as model:
    player_skill = pm.Normal("player_skill", mu=0, sigma=1, shape=len(players))
    deck_strength = pm.Normal("deck_strength", mu=0, sigma=1, shape=len(decks))

    skill_diff = (player_skill[matches_df["player_A_idx"].values] + deck_strength[matches_df["deck_A_idx"].values]) - \
                 (player_skill[matches_df["player_B_idx"].values] + deck_strength[matches_df["deck_B_idx"].values])

    win_prob = pm.Deterministic("win_prob", pm.math.sigmoid(skill_diff))
    outcome = pm.Bernoulli("outcome", p=win_prob, observed=matches_df["Outcome"].values)

    trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True)

# Use the trace object from PyMC sampling
# Add coordinates to the trace for labeling
trace.posterior = trace.posterior.assign_coords({
    "player_skill_dim_0": list(players),
    "deck_strength_dim_0": list(decks)
})

# Rename dimensions for clarity
trace.posterior = trace.posterior.rename({
    "player_skill_dim_0": "player",
    "deck_strength_dim_0": "deck"
})

# Plot player skills with names
az.plot_forest(trace, var_names=["player_skill"], combined=True, figsize=(10, len(players) // 2))
plt.title("Posterior Distributions of Player Skills")
plt.tight_layout()
plt.savefig("player_skills_named.png")

# Plot deck strengths with names
az.plot_forest(trace, var_names=["deck_strength"], combined=True, figsize=(10, len(decks) // 2))
plt.title("Posterior Distributions of Deck Strengths")
plt.tight_layout()
plt.savefig("deck_strengths_named.png")

# Extract posterior mean estimates
player_means = trace.posterior["player_skill"].mean(dim=["chain", "draw"]).values
deck_means = trace.posterior["deck_strength"].mean(dim=["chain", "draw"]).values
player_stds = trace.posterior["player_skill"].std(dim=["chain", "draw"]).values
deck_stds = trace.posterior["deck_strength"].std(dim=["chain", "draw"]).values

# Create DataFrames
player_df = pd.DataFrame({"Player": list(players), "Skill_Estimate": player_means, "Skill_StD": player_stds})
deck_df = pd.DataFrame({"Deck": list(decks), "Strength_Estimate": deck_means, "Strength_StD": deck_stds})

# Save to CSV files
player_df.to_csv("player_skill_estimates.csv", index=False)
deck_df.to_csv("deck_strength_estimates.csv", index=False)

