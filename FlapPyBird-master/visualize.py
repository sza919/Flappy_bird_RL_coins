import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the saved training data
with open("training_metrics.json", "r") as f:
    data = json.load(f)

episodes = np.arange(len(data["scores"]))
scores = np.array(data["scores"])
max_scores = np.maximum.accumulate(scores)  # Cumulative max scores

# Convert rolling mean scores (some NaNs at the start)
rolling_mean_scores = np.array(data["rolling_mean_scores"])
rolling_mean_scores[np.isnan(rolling_mean_scores)] = 0  # Replace NaNs with 0

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(episodes, scores, s=1, color="blue", alpha=0.7, label="Scores per Episode")
plt.plot(episodes, rolling_mean_scores, color="orange", label="Rolling Mean Score", linewidth=2)
plt.plot(episodes, max_scores, color="green", label="Max Score So Far", linewidth=2)

# Log scale for better visualization
plt.yscale("log")
plt.xlabel("Episode")
plt.ylabel("log(Score)")
plt.title("DQN Training Progress - Score Over Time (Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plot
plt.show()
