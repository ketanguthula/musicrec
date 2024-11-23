import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def spotify_auth():
    """
    Authenticates and initializes the Spotify client.

    Returns:
        spotipy.Spotify: Authenticated Spotify client instance.
    """
    os.environ['SPOTIPY_CLIENT_ID'] = '346ccc38266c4bde953ddd073f7a021b'
    os.environ['SPOTIPY_CLIENT_SECRET'] = '94b9c47f8aef41f69c9fbe407596ad90'
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def get_audio_features(track_ids, sp):
    """
    Retrieves audio features from Spotify for a list of track IDs.

    Parameters:
        track_ids (list): List of track IDs to retrieve features for.
        sp (spotipy.Spotify): Authenticated Spotify client.

    Returns:
        list of dict: Each dictionary contains audio features like danceability, energy, tempo, etc., for a track.
    """
    all_track_features = []
    for i in range(0, len(track_ids), 100):
        audio_features = sp.audio_features(track_ids[i:i + 100])
        all_track_features.extend([{
            'track_id': features['id'],
            'danceability': features['danceability'],
            'energy': features['energy'],
            'key': features['key'],
            'loudness': features['loudness'],
            'mode': features['mode'],
            'speechiness': features['speechiness'],
            'acousticness': features['acousticness'],
            'instrumentalness': features['instrumentalness'],
            'liveness': features['liveness'],
            'valence': features['valence'],
            'tempo': features['tempo']
        } for features in audio_features if features])
    return all_track_features

def save_audio_features_to_csv(audio_features, filename="audio_features_dataset.csv"):
    """
    Saves new audio features to CSV, appending without duplicating existing tracks.

    Parameters:
        audio_features (list of dict): List of audio features for tracks.
        filename (str): CSV filename to save data to (default is 'audio_features_dataset.csv').

    Returns:
        None
    """
    existing_df = load_dataset(filename)
    existing_ids = set(existing_df['track_id'])
    new_features = [feature for feature in audio_features if feature['track_id'] not in existing_ids]
    new_features_df = pd.DataFrame(new_features)

    if not new_features_df.empty:
        with open(filename, 'a') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.write("\n")

        new_features_df.to_csv(filename, mode='a', index=False, header=False)
        print(f"Added {len(new_features)} new audio features to {filename}")
    else:
        print("No new audio features to add.")

def data_clean():
    """
    Cleans and aligns data between jpop_dataset and audio_features_dataset, then merges into combined_dataset.

    Returns:
        None
    """
    jpop_df = pd.read_csv("jpop_dataset.csv").drop_duplicates(subset='track_id')
    audio_df = pd.read_csv("audio_features_dataset.csv").drop_duplicates(subset='track_id')
    common_track_ids = set(jpop_df['track_id']).intersection(audio_df['track_id'])
    jpop_df_filtered = jpop_df[jpop_df['track_id'].isin(common_track_ids)]
    audio_df_filtered = audio_df[audio_df['track_id'].isin(common_track_ids)]
    jpop_df_filtered.to_csv("jpop_dataset.csv", index=False)
    audio_df_filtered.to_csv("audio_features_dataset.csv", index=False)
    remove_comma("audio_features_dataset.csv")
    remove_comma("combined_dataset.csv")
    combined_df = pd.merge(jpop_df_filtered, audio_df_filtered, on='track_id', how='inner')
    if 'track_name_x' in combined_df.columns and 'track_name_y' in combined_df.columns:
        combined_df = combined_df.rename(columns={'track_name_x': 'track_name'}).drop(columns=['track_name_y'])
    combined_df.to_csv("combined_dataset.csv", index=False)

def clean_csv_trailing_commas(file_path):
    """
    Cleans trailing commas from a CSV file by reading and saving it back in the correct format.

    Parameters:
        file_path (str): The path to the CSV file to be cleaned.

    Returns:
        None
    """
    df = pd.read_csv(file_path, engine='python')
    df.to_csv(file_path, index=False)

def remove_comma(filepath):
    """
    Removes trailing commas from each line in a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        None
    """
    with open(filepath, 'r') as fp:
        lines = [line.rstrip(',\n') for line in fp]
    with open(filepath, 'w') as fp:
        for line in lines:
            fp.write(line + '\n')

def load_dataset(filename):
    """
    Loads a dataset from CSV or initializes an empty DataFrame if the file doesn't exist.

    Parameters:
        filename (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: Loaded DataFrame or an empty DataFrame if the file doesn't exist.
    """
    return pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()

def save_to_csv(tracks, filename="jpop_dataset.csv"):
    """
    Saves unique tracks to a CSV file, appending without duplicating existing tracks.

    Parameters:
        tracks (list of dict): List of track data dictionaries.
        filename (str): CSV filename to save data to (default is 'jpop_dataset.csv').

    Returns:
        None
    """
    existing_df = load_dataset(filename)
    new_tracks_df = pd.DataFrame(tracks).drop_duplicates(subset="track_id")
    combined_df = pd.concat([existing_df, new_tracks_df]).drop_duplicates(subset="track_id")
    combined_df.to_csv(filename, index=False)

def apply_clustering(df, features):
    """
    Applies clustering to a DataFrame based on selected features, handling missing values as needed.

    Parameters:
        df (pd.DataFrame): DataFrame containing the feature columns.
        features (list): List of feature column names to use for clustering.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'cluster' column.
    """
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])

    if 'cluster' not in df.columns:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        kmeans = KMeans(n_clusters=min(10, len(df)), random_state=42)
        df['cluster'] = kmeans.fit_predict(df[features])
        print("Clustering applied to DataFrame.")
    else:
        print("Clustering already applied.")

    return df

def append_new(array, sp):
    """
    Appends new tracks to 'jpop_dataset.csv' and their audio features to 'audio_features_dataset.csv'.

    Parameters:
        array (list of dict): List of new track data dictionaries.
        sp (spotipy.Spotify): Authenticated Spotify client.

    Returns:
        None
    """
    save_to_csv(array, "jpop_dataset.csv")
    df = pd.read_csv("jpop_dataset.csv")
    num_rows = len(df)
    num_ext = len(array)
    n = num_rows - num_ext
    jdf = pd.read_csv("jpop_dataset.csv", skiprows=range(1, n), nrows=n + 1)
    track_ids = jdf['track_id'].tolist()
    audio_features = get_audio_features(track_ids, sp)
    save_to_csv(audio_features, filename="audio_features_dataset.csv")
    data_clean()

remove_comma("combined_dataset.csv")