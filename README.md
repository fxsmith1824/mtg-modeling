# mtg-modeling
First pass attempt at modeling MTG tournament data

So far this only takes in my manually formatted Pro Tour Edge of Eternities data and analyzes it.

My goals are to expand this using melee.gg API in the future.

For the current data, an initial model to estimate "player skill" (broadly) is trained on the 6 draft rounds. Then those player skill posterior distributions are used as the priors for modeling the remaining constructed Swiss rounds with both "player skill" and "deck strength" being estimated from the data.

The final estimates of deck strength broadly line up with some of the takeaways others have written about already (with some interesting variations as well, which I'll write up soon-ish). The player skill estimates are not meant to make any serious judgment of players (though the model does put 5 of the Top 8 players in the 8 highest "player skill" estimates) - it was more to try to more appropriately evaluate deck performance by also accounting for player skill. This might not be beneficial in the Pro Tour style setting where player skill has generally been filtered based on the qualification procedures but I wanted to try it anyways.