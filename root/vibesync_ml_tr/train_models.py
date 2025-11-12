# train_models.py

import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- PART 1: SIMULATE DATA ---
# In a real project, you would load your Kaggle datasets here.
# We simulate data so the script is runnable for everyone.

def create_simulated_data(num_samples=50):
    """
    Creates a simulated dataset of audio, features, and moods.
    This replaces the "Music Emotion Recognition" dataset 
    and the Spotify API calls for this step.
    """
    print(f"Step 1: Simulating {num_samples} samples of audio + features...")
    data = []
    sample_rate = 22050
    moods = ["Happy", "Sad", "Energetic", "Calm"]
    
    for i in range(num_samples):
        mood = np.random.choice(moods)
        
        # 1. Simulate a 3-second audio clip
        # (This is just noise to make librosa work)
        audio_signal = np.random.uniform(-0.5, 0.5, size=(sample_rate * 3,))
        
        # 2. Simulate the feature vector (what Spotify gives you)
        feature_vector = {
            'danceability': np.random.rand(),
            'energy': np.random.rand(),
            'valence': np.random.rand(),
            'tempo': np.random.uniform(80, 180)
        }
        
        data.append({
            "audio_signal": audio_signal,
            "features": feature_vector,
            "mood": mood
        })
        
    return data, sample_rate

# --- PART 2: STAGE 1 - TRAIN THE CNN (Audio -> Mood) ---
# This implements the CNN from your report [cite: 64, 104]

def train_cnn_model(data, sample_rate):
    """
    Trains the Deep Learning CNN on Mel-spectrograms [cite: 65] to predict mood.
    """
    print("\n--- STAGE 1: Training Deep Learning CNN on Audio ---")

    # 1. Preprocess: Create spectrograms and labels
    X_spectrograms = []
    y_moods = []

    for item in data:
        # Create Mel-spectrogram (an "image" of the audio)
        spectrogram = librosa.feature.melspectrogram(y=item["audio_signal"], sr=sample_rate, n_mels=64)
        # Resize all spectrograms to a fixed size (e.g., 64x128)
        # We'll just pad/truncate this example
        fixed_shape_spec = np.zeros((64, 128))
        shape_to_use = min(spectrogram.shape[1], 128)
        fixed_shape_spec[:, :shape_to_use] = spectrogram[:, :shape_to_use]

        X_spectrograms.append(fixed_shape_spec)
        y_moods.append(item["mood"])

    # Reshape for CNN (samples, height, width, channels)
    X = np.array(X_spectrograms).reshape(len(X_spectrograms), 64, 128, 1)

    # Encode labels ("Happy" -> 0, "Sad" -> 1, etc.)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_moods)
    y_categorical = to_categorical(y_encoded)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # 2. Build the CNN Model (a simple version)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 128, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(y_categorical.shape[1], activation='softmax') # Output layer
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train the model
    print("Training CNN... (This is fast on simulated data)")
    model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=0)

    # 4. Evaluate the model on test set
    print("Evaluating CNN on test set...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"CNN Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    target_names = label_encoder.classes_
    print("CNN Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))

    # Save the CNN model (optional, not used by our API)
    # model.save('cnn_audio_model.keras')
    print("...CNN Training and Evaluation Complete.")

# --- PART 3: STAGE 2 - TRAIN THE "PROXY" MODEL (Features -> Mood) ---
# This is the "bridge" model. It learns to imitate the CNN,
# but using only the simple audio features.

def train_proxy_model(data):
    """
    Trains a simpler Random Forest model to predict mood from
    *audio features* (danceability, energy, etc.).
    """
    print("\n--- STAGE 2: Training 'Proxy' Model (Features -> Mood) ---")
    
    # 1. Preprocess: Create feature (X) and label (y) set
    feature_list = []
    y_moods = []
    
    for item in data:
        feature_list.append(list(item["features"].values()))
        y_moods.append(item["mood"])
        
    X = pd.DataFrame(feature_list, columns=["danceability", "energy", "valence", "tempo"])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_moods)
    
    # 2. Scale features and split data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Train the Random Forest model
    print("Training Random Forest 'Proxy' model...")
    proxy_model = RandomForestClassifier(n_estimators=50, random_state=42)
    proxy_model.fit(X_train, y_train)
    
    # 4. Test its accuracy and other metrics
    y_pred = proxy_model.predict(X_test)
    print(f"...Proxy Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Classification report
    target_names = label_encoder.classes_
    print("Proxy Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 5. SAVE THE FINAL MODEL
    # This is the *only* model our live API needs!
    joblib.dump(proxy_model, 'mood_from_features_model.joblib')
    # We also save the scaler and label encoder, as we'll need them
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(label_encoder, 'mood_label_encoder.joblib')
    
    print("... 'mood_from_features_model.joblib' (and helpers) saved!")
    

# --- Main execution ---
if __name__ == "__main__":
    
    # Step 1: Get Data
    simulated_data, sample_rate = create_simulated_data(num_samples=100)
    
    # Step 2: Train the big CNN (as described in your report)
    train_cnn_model(simulated_data, sample_rate)
    
    # Step 3: Train the "Proxy" model (the one our API will use)
    train_proxy_model(simulated_data)
    
    print("\n--- ML Training Complete! ---")
    print("You now have 'mood_from_features_model.joblib', which is ready for the next phase.")