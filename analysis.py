"""
analysis.py

This script provides visual analyses for the combined dataset of audio features. It includes functions
for plotting histograms of individual features and a heatmap to visualize the correlations among features.

Prerequisites:
- Ensure `combined_dataset.csv` exists by running the data cleanup process if needed.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to combined data file
combined_file = "combined_dataset.csv"

# Load the combined dataset
if os.path.exists(combined_file):
    combined_df = pd.read_csv(combined_file)
    print(f"Loaded combined dataset from {combined_file}")
else:
    raise FileNotFoundError(
        f"The file {combined_file} does not exist. Please run data_cleanup.py first to generate it.")

# Features to visualize
features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness', 'instrumentalness', 'liveness']


def plot_feature_distributions():
    """
    Plots histograms for each specified feature in the combined dataset.

    Features:
        - danceability, energy, tempo, valence, acousticness, instrumentalness, liveness

    Each histogram displays the frequency distribution of values within the feature.

    Returns:
        None
    """
    for feature in features:
        plt.figure(figsize=(6, 4))
        plt.hist(combined_df[feature].dropna(), bins=15, edgecolor='black')
        plt.title(f'Distribution of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')
        plt.show()


def plot_correlation_heatmap():
    """
    Plots a heatmap of the correlation matrix for the specified features in the combined dataset.

    The heatmap shows the correlation coefficients between each pair of features, indicating how
    closely related they are to each other.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = combined_df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap of Audio Features')
    plt.show()


# Execute the plotting functions
plot_feature_distributions()
plot_correlation_heatmap()
