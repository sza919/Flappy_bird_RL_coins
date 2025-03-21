import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from collections import Counter

def visualize_training(json_file):
    # Load the saved training data
    with open(json_file + ".json", "r") as f:
        data = json.load(f)
    display_len = 1000
    episodes = np.arange(len(data["scores"]))[:display_len]
    scores = np.array(data["scores"])[:display_len]
    max_scores = np.array(data["max_scores"])[:display_len]
    print(len(data["scores"]), len(data["max_scores"]), len(data["rolling_mean_scores"]))
    # Convert rolling mean scores (some NaNs at the start)
    rolling_mean_scores = np.array(data["rolling_mean_scores"])[:display_len]
    rolling_mean_scores[np.isnan(rolling_mean_scores)] = 0  # Replace NaNs with 0
    coins_collected = np.array(data["coins_collected"])[:display_len]
    window_size = 100

    # If you want to keep the same length as the original array, use mode='same'
    rolling_avg_same_length = np.convolve(coins_collected, np.ones(window_size)/window_size, mode='same')
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(episodes, scores, s=1, color="blue", alpha=0.7, label="Scores at Episode")
    plt.plot(episodes, rolling_avg_same_length, color="red", label="Coins Collected")
    plt.plot(episodes, rolling_mean_scores, color="orange", label="Rolling Mean Score", linewidth=2)
    plt.plot(episodes, max_scores, color="green", label="Max Score So Far", linewidth=2)

    model_name = json_file
    if "training_metrics_" in model_name:
        model_name = model_name.replace("training_metrics_", "")
    else:
        model_name = ""
    # Log scale for better visualization
    plt.yscale("log")
    plt.xlabel("Episode")
    plt.ylabel("log(Score)")
    plt.title(f"DQN{model_name.capitalize()} Training Progress - Score Over Time (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Create output filename based on input JSON name
    output_base = "DQN" + model_name.capitalize()  + "_data" if model_name else "DQN_data"
    plt.savefig(f"{output_base}_plot.png", dpi=900, bbox_inches='tight')
    plt.close()

    print(f"Plot has been saved as: {output_base}_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize DQN training metrics from JSON file')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing training metrics')
    args = parser.parse_args()

    visualize_training(args.json_file)