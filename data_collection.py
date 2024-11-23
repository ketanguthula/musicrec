# data_collection.py
# This script collects metadata on J-Pop tracks and their audio features from Spotify and saves them to local CSV files.

import pandas as pd
from utility import spotify_auth, get_audio_features, save_to_csv

# Initialize Spotify client from utility
sp = spotify_auth()

# List of J-Pop artists for data collection; can be updated to include more artists.
jpop_artists = [
    "The Pillows", "Polkadot Stingray", "Fujifabric", "Tokyo Ska Paradise Orchestra",
    "Hoshino Gen", "Yorushika", "Suchmos", "Chai", "Sumika", "Takako Matsu",
]


def get_jpop_tracks_rotated(jpop_artists_subset, limit=50, num_batches=20):
    """
    Retrieves J-Pop tracks in multiple batches from Spotify's API based on the provided artist list.

    Parameters:
        jpop_artists_subset (list): List of J-Pop artist names to use in the search.
        limit (int): Number of tracks to retrieve per batch (default is 50).
        num_batches (int): Total number of batches to retrieve (default is 20).

    Returns:
        list of dict: Each dictionary contains track metadata, including 'track_id', 'track_name', 'artist_name',
                      'album_name', 'release_date', and 'popularity'.
    """
    all_tracks = []
    for i in range(num_batches):
        try:
            # Fetch a batch of tracks from Spotify
            results = sp.search(q="Japanese Pop", type="track", limit=limit, offset=i * limit)
            tracks = results['tracks']['items']

            # Extract relevant track details
            jpop_tracks = [{
                'track_id': track['id'],
                'track_name': track['name'],
                'artist_name': track['artists'][0]['name'],
                'album_name': track['album']['name'],
                'release_date': track['album']['release_date'],
                'popularity': track['popularity']
            } for track in tracks]
            all_tracks.extend(jpop_tracks)
        except Exception as e:
            print(f"Error on batch {i}: {e}")
    print("Number of tracks pulled:", len(all_tracks))
    return all_tracks


def collect_and_save_jpop_data():
    """
    Collects metadata for J-Pop tracks and saves it to 'jpop_dataset.csv' using the utility function.

    Parameters:
        None

    Returns:
        None
    """
    jpop_tracks = get_jpop_tracks_rotated(jpop_artists_subset=jpop_artists)
    save_to_csv(jpop_tracks, filename="jpop_dataset.csv")  # Saves unique tracks only
    print("J-Pop track data collection complete.")


def collect_and_save_audio_features():
    """
    Collects audio features for each track in 'jpop_dataset.csv' and saves them to 'audio_features_dataset.csv'.

    Parameters:
        None

    Returns:
        None
    """
    # Load track IDs from the saved J-Pop dataset
    jpop_df = pd.read_csv("jpop_dataset.csv")
    track_ids = jpop_df['track_id'].tolist()

    # Fetch audio features and save them
    audio_features = get_audio_features(track_ids, sp)
    save_to_csv(audio_features, filename="audio_features_dataset.csv")  # Append new features
    print("Audio features data collection complete.")


# Main script execution
if __name__ == "__main__":
    collect_and_save_jpop_data()
    collect_and_save_audio_features()

