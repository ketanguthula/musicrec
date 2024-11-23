"""
backend.py

This script provides backend functionalities for the recommendation system, including clustering, collaborative filtering, 
and external Spotify recommendations. It leverages the Spotify API to fetch data, processes it using various recommendation 
methods, and returns a combined set of recommendations for tracks.

Functions:
    - recommend_from_cluster: Get song recommendations based on cluster similarity.
    - recommend_collaborative: Get recommendations using collaborative filtering.
    - spotify_external_recommendations: Get external recommendations from Spotify based on seed tracks/artists/genres.
    - hybrid_recommendation: A hybrid recommendation function combining clustering, collaborative filtering, and external recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import utility

# Authenticate Spotify client
sp = utility.spotify_auth()

# Load combined dataset
combined_df = pd.read_csv("combined_dataset.csv")

# Define features for clustering and apply clustering if not already applied
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
combined_df = utility.apply_clustering(combined_df, features)

# Collaborative Filtering Setup
num_users = 100
interaction_matrix = np.random.randint(0, 2, (num_users, len(combined_df)))
interaction_df = pd.DataFrame(interaction_matrix, columns=combined_df['track_id'].values)
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(interaction_df.values.T)

def recommend_from_cluster(track_id, df):
    """
    Recommends songs based on cluster similarity.

    Parameters:
        - track_id (str): The track ID for which recommendations are needed.
        - df (DataFrame): The combined dataset with track features.

    Returns:
        DataFrame: Recommended songs from the same cluster as the provided track ID.
    """
    if 'cluster' not in df.columns:
        raise KeyError("DataFrame is missing required 'cluster' column for clustering.")
    cluster_id = df[df['track_id'] == track_id]['cluster'].values[0]
    recommendations = df[df['cluster'] == cluster_id].sample(5)
    return recommendations[['track_name', 'artist_name']]

def recommend_collaborative(track_id, df, n_recommendations=5):
    """
    Recommends songs based on collaborative filtering.

    Parameters:
        - track_id (str): The track ID for which collaborative recommendations are needed.
        - df (DataFrame): The combined dataset with track features.
        - n_recommendations (int): Number of recommendations to return.

    Returns:
        DataFrame: Recommended songs based on collaborative filtering.
    """
    if track_id not in interaction_df.columns:
        print("Track ID not found in interaction matrix for collaborative filtering.")
        return pd.DataFrame(columns=['track_name', 'artist_name'])
    track_index = np.where(interaction_df.columns == track_id)[0][0]
    distances, indices = model.kneighbors([interaction_df.iloc[:, track_index]], n_neighbors=n_recommendations + 1)
    recommended_indices = indices.flatten()[1:]
    recommended_track_ids = interaction_df.columns[recommended_indices].values
    return df[df['track_id'].isin(recommended_track_ids)][['track_name', 'artist_name']]

def spotify_external_recommendations(seed_tracks=None, seed_artists=None, seed_genres=['j-pop'], limit=20):
    """
    Fetches external recommendations from Spotify based on seed tracks, artists, or genres.

    Parameters:
        - seed_tracks (list): List of track IDs to use as seeds.
        - seed_artists (list): List of artist IDs to use as seeds.
        - seed_genres (list): List of genres to use as seeds.
        - limit (int): Maximum number of recommendations to retrieve.

    Returns:
        list of dict: List of recommended tracks with their metadata.
    """
    try:
        recommendations = sp.recommendations(seed_tracks=seed_tracks, seed_artists=seed_artists,
                                             seed_genres=seed_genres, limit=limit)
        return [{
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_name': track['artists'][0]['name'],
            'album_name': track['album']['name'],
            'release_date': track['album']['release_date'],
            'popularity': track['popularity']
        } for track in recommendations['tracks']]
    except Exception as e:
        print(f"Error fetching external recommendations: {e}")
        return []

def hybrid_recommendation(track_id, df, n_cluster_recs=5, n_collab_recs=5, n_external_recs=5):
    """
    Generates hybrid recommendations combining clustering, collaborative filtering, and external Spotify recommendations.

    Parameters:
        - track_id (str): The track ID for which recommendations are needed.
        - df (DataFrame): The combined dataset with track features.
        - n_cluster_recs (int): Number of cluster-based recommendations.
        - n_collab_recs (int): Number of collaborative recommendations.
        - n_external_recs (int): Number of external Spotify recommendations.

    Returns:
        DataFrame or str: Combined recommendations as a DataFrame or a message if no recommendations are available.
    """
    if track_id:
        # Check if track_id exists in combined_df
        if track_id not in combined_df['track_id'].values:
            # If not in combined_df, fetch audio features and save it
            print(f"Adding track '{track_id}' to the dataset.")
            external_audio_features = utility.get_audio_features([track_id], sp)
            utility.save_audio_features_to_csv(external_audio_features, "audio_features_dataset.csv")

            # Fetch track details from Spotify and save to jpop_dataset
            track_info = sp.track(track_id)
            track_data = {
                'track_id': track_id,
                'track_name': track_info['name'],
                'artist_name': track_info['artists'][0]['name'],
                'album_name': track_info['album']['name'],
                'release_date': track_info['album']['release_date'],
                'popularity': track_info['popularity']
            }
            utility.save_to_csv([track_data], "jpop_dataset.csv")

            # Run data cleaning to update combined dataset
            utility.data_clean()

    # Generate recommendations using different methods
    cluster_recs = recommend_from_cluster(track_id, df)
    collab_recs = recommend_collaborative(track_id, df, n_recommendations=n_collab_recs)
    print(cluster_recs)
    print(collab_recs)

    # Get external recommendations
    external_recs = spotify_external_recommendations(seed_tracks=[track_id])
    if external_recs:
        external_track_ids = [track['track_id'] for track in external_recs]
        utility.save_to_csv(external_recs, "jpop_dataset.csv")

        existing_audio_df = pd.read_csv("audio_features_dataset.csv")
        existing_audio_ids = set(existing_audio_df['track_id'])
        new_track_ids = [track_id for track_id in external_track_ids if track_id not in existing_audio_ids]

        if new_track_ids:
            external_audio_features = utility.get_audio_features(new_track_ids, sp)
            utility.save_audio_features_to_csv(external_audio_features, "audio_features_dataset.csv")

        utility.data_clean()
        external_recs_df = pd.DataFrame(external_recs)[['track_name', 'artist_name']]
    else:
        external_recs_df = pd.DataFrame(columns=['track_name', 'artist_name'])

    # Combine all recommendations
    combined_recs = pd.concat([cluster_recs, collab_recs, external_recs_df]).drop_duplicates()
    sample_size = min(len(combined_recs), n_cluster_recs + n_collab_recs + n_external_recs)
    utility.remove_comma("combined_dataset.csv")
    return combined_recs.sample(n=sample_size) if sample_size > 0 else "No recommendations available."
