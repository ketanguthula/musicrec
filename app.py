import gradio as gr
import pandas as pd
import utility
from backend import hybrid_recommendation
import spotipy

# Authenticate Spotify
sp = utility.spotify_auth()

# Load combined dataset
combined_df = pd.read_csv("combined_dataset.csv")

# Features for clustering
features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']

# Ensure clustering is applied
utility.apply_clustering(combined_df, features)


def get_track_id_from_name(track_name):
    try:
        results = sp.search(q=f"track:{track_name}", type="track", limit=1)
        if results['tracks']['items']:
            return results['tracks']['items'][0]['id']
        else:
            return None
    except Exception as e:
        return f"Error fetching track ID: {e}"

def get_artist_id_from_name(artist_name):
    try:
        results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        if results['artists']['items']:
            return results['artists']['items'][0]['id']
        else:
            return None
    except Exception as e:
        return f"Error fetching artist ID: {e}"


def rec(input_text, input_type):
    global combined_df

    if input_type == 'Track':
        track_id = get_track_id_from_name(input_text)
        if track_id:
            recommendations = hybrid_recommendation(track_id, combined_df)
            return recommendations
        else:
            return "Track not found on Spotify."

    elif input_type == 'Artist':
        artist_id = get_artist_id_from_name(input_text)
        if artist_id:
            top_tracks = sp.artist_top_tracks(artist_id, country='US')
            if top_tracks['tracks']:
                top_track_id = top_tracks['tracks'][0]['id']  # Use the first top track's ID
                top_track_name = top_tracks['tracks'][0]['name']

                # Check if top_track_id exists in combined_df
                if top_track_id not in combined_df['track_id'].values:
                    # If not in combined_df, fetch audio features and save it
                    external_audio_features = utility.get_audio_features([top_track_id], sp)
                    utility.save_audio_features_to_csv(external_audio_features, "audio_features_dataset.csv")
                    utility.save_to_csv([{
                        'track_id': top_track_id,
                        'track_name': top_track_name,
                        'artist_name': input_text,
                        'album_name': top_tracks['tracks'][0]['album']['name'],
                        'release_date': top_tracks['tracks'][0]['album']['release_date'],
                        'popularity': top_tracks['tracks'][0]['popularity']
                    }], "jpop_dataset.csv")

                    # Run data cleaning to update combined dataset
                    utility.data_clean()

                    # Reload combined_df to reflect the new track
                    combined_df = pd.read_csv("combined_dataset.csv")
                    print("Track added to dataset and dataset cleaned.")

                # Call hybrid_recommendation after reloading updated combined_df
                recommendations = hybrid_recommendation(top_track_id, combined_df)
                return f"Using top track '{top_track_name}' by {input_text} for recommendations.\n\n{recommendations}"
            else:
                return "No top tracks found for this artist on Spotify."
        else:
            return "Artist not found on Spotify."


# Create Gradio interface
input_text = gr.Textbox(label="Enter Track or Artist Name")
input_type = gr.Radio(["Track", "Artist"], label="Select Input Type")

# Define output as a Gradio text component
output = gr.Textbox(label="Recommendations")

# Launch Gradio app
gr.Interface(fn=rec, inputs=[input_text, input_type], outputs=output, title="Spotify Track & Artist Recommendations",
             description="Get Spotify track or artist-based recommendations by entering a track or artist name and selecting the input type.").launch()
