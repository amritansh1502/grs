# classify_database.py

import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
# --- REPLACE THIS with the path to your REAL Kaggle CSV ---
# e.g., KAGGLE_DATASET_PATH = '~/Downloads/spotify_tracks_dataset.csv'
KAGGLE_DATASET_PATH = 'database.csv' # Set to None to use simulated data

# These are the feature columns your proxy model was trained on
MODEL_FEATURES = ["danceability", "energy", "valence", "tempo"]

# These are the files you created in Phase 1
MODEL_PATH = 'mood_from_features_model.joblib'
SCALER_PATH = 'feature_scaler.joblib'
ENCODER_PATH = 'mood_label_encoder.joblib'

# The name of the final file our API will use
OUTPUT_DATABASE_PATH = 'classified_song_database.csv'


def load_kaggle_dataset():
    """
    Loads your big Kaggle dataset.
    We simulate it if no path is provided.
    """
    print(f"Step 1: Loading dataset...")
    if KAGGLE_DATASET_PATH:
        try:
            df = pd.read_csv(KAGGLE_DATASET_PATH)
            # You might need to rename columns to match MODEL_FEATURES
            # e.g., if your CSV has 'track_danceability', rename it.
        except FileNotFoundError:
            print(f"Error: Could not find {KAGGLE_DATASET_PATH}")
            print("Using simulated data instead.")
            return create_simulated_kaggle_data(1000)
    else:
        print("KAGGLE_DATASET_PATH is not set. Using simulated data.")
        df = create_simulated_kaggle_data(1000)

    # Basic cleaning: Drop rows with missing features
    df = df.dropna(subset=MODEL_FEATURES + ['track_name'])
    return df

def create_simulated_kaggle_data(num_samples=1000):
    """
    Simulates a large Kaggle dataset with the columns we need.
    """
    data = {
        'track_name': [f"Song {i}" for i in range(num_samples)],
        'artist_name': [f"Artist {i % 100}" for i in range(num_samples)],
        'danceability': np.random.rand(num_samples),
        'energy': np.random.rand(num_samples),
        'valence': np.random.rand(num_samples),
        'tempo': np.random.uniform(80, 180, size=num_samples),
        'other_feature_1': np.random.rand(num_samples), # To simulate a big file
        'other_feature_2': np.random.rand(num_samples)
    }
    return pd.DataFrame(data)

def classify_database(df):
    """
    Uses the trained proxy model to predict the mood for every
    song in the DataFrame.
    """
    print("Step 2: Loading trained models (model, scaler, encoder)...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not load model files. {e}")
        print("Please run 'train_models.py' (Phase 1) first.")
        return

    print("Step 3: Preparing and scaling dataset features...")
    # Select only the features the model was trained on
    X_features = df[MODEL_FEATURES]
    
    # Scale the features using the *same scaler* from training
    X_scaled = scaler.transform(X_features)

    print("Step 4: Predicting moods for all songs...")
    # Predict the numeric labels (e.g., 0, 1, 2, 3)
    predicted_labels_numeric = model.predict(X_scaled)
    
    # Use the encoder to turn numbers back into text
    # e.g., [0, 1, 2] -> ["Calm", "Energetic", "Happy"]
    predicted_moods = encoder.inverse_transform(predicted_labels_numeric)
    
    # Add the moods as a new column to our database
    df['predicted_mood'] = predicted_moods
    
    print("...Mood prediction complete.")
    
    print("Step 5: Saving final classified database...")
    # Select only the columns our API will need, to keep it small
    final_columns = ['track_name', 'artist_name'] + MODEL_FEATURES + ['predicted_mood']
    
    # Ensure all required columns exist before trying to save
    final_df = df[[col for col in final_columns if col in df.columns]]
    
    # Save to CSV
    final_df.to_csv(OUTPUT_DATABASE_PATH, index=False)
    
    print(f"\n--- Phase 2 Complete! ---")
    print(f"Successfully created '{OUTPUT_DATABASE_PATH}'")
    print("Final database preview:")
    print(final_df.head())


# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Load data
    main_df = load_kaggle_dataset()
    
    # 2. Classify and save
    if main_df is not None:
        classify_database(main_df)