# api.py
# This is the "ML Engine" server for the CNN-based project.

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Load the Classified Database at startup ---
print("Loading classified song database...")
try:
    # These are the feature columns our API expects from Node.js
    # Must match the order used in MODEL_FEATURES from Phase 2
    FEATURE_COLS = ["danceability", "energy", "valence", "tempo"]
    
    # Load the database created in Phase 2
    song_db = pd.read_csv('classified_song_database.csv')
    
    # Ensure all required columns exist
    for col in FEATURE_COLS + ['predicted_mood']:
        if col not in song_db.columns:
            raise Exception(f"Database is missing required column: {col}")
    
    # Drop any rows with missing data to be safe
    song_db = song_db.dropna(subset=FEATURE_COLS + ['predicted_mood'])
    
    print("...Database loaded successfully.")
    print(f"Total songs in library: {len(song_db)}")

except FileNotFoundError:
    print("Error: 'classified_song_database.csv' not found.")
    print("Please run Phase 2 (classify_database.py) and copy the file here.")
    exit()
except Exception as e:
    print(f"Error loading database: {e}")
    exit()


# --- 2. Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Allow cross-origin requests

# --- 3. Define the Recommendation Endpoint ---
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    
    try:
        # Get data from Node.js API
        mood_vote = data['mood_vote']
        avg_vector = data['average_feature_vector'] # This is a list, e.g., [0.5, 0.6, 0.7, 120.0]
        
        # --- Step 1: Filter by Mood ---
        # (This is the "mood-based pre-filtering" from your report)
        mood_filtered_songs = song_db[song_db['predicted_mood'] == mood_vote]
        
        if mood_filtered_songs.empty:
            print(f"No songs found for mood '{mood_vote}'. Using fallback.")
            # Fallback: If no songs match the mood, use the whole database
            mood_filtered_songs = song_db
        
        # --- Step 2: Calculate Cosine Similarity ---
        # (This is the content-based filtering from your report)
        
        # Get the feature vectors for all songs in the mood subset
        song_features = mood_filtered_songs[FEATURE_COLS].values
        
        # The average_feature_vector from Node.js
        group_vector = np.array(avg_vector).reshape(1, -1)
        
        # Calculate similarity between the group vector and all songs
        similarity_scores = cosine_similarity(group_vector, song_features)
        
        # --- Step 3: Get Top N Songs ---
        
        # Get the indices of the top 20 most similar songs
        # We use [0] because similarity_scores is a 2D array [1, num_songs]
        top_indices = similarity_scores[0].argsort()[-20:][::-1]
        
        # Get the full song data for these top indices
        top_songs = mood_filtered_songs.iloc[top_indices]
        
        # Convert to JSON and return
        return jsonify({
            'recommendations': top_songs.to_dict(orient='records')
        })

    except KeyError as e:
        return jsonify({'error': f"Missing key in request: {e}"}), 400
    except Exception as e:
        print(f"Error in /recommend: {e}")
        return jsonify({'error': str(e)}), 500

# --- 4. Define a simple /predict endpoint (for consistency) ---
# Our new Node.js logic doesn't use this, but we keep it in case.
# The *real* prediction was already done offline in Phase 2.
@app.route('/predict', methods=['POST'])
def predict():
    # This endpoint is no longer used by the main logic,
    # but we can provide a dummy or error response.
    return jsonify({'message': 'This endpoint is not used. Prediction is done in Phase 2.'}), 404


# --- 5. Run the App ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5001
    app.run(port=5001, debug=True)