import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- PART 1: SIMULATE DATA (MODIFIED) ---
# We now create *patterns* for the models to learn.

def create_simulated_data(num_samples=500): # Increased samples
    """
    Creates a *patterned* simulated dataset.
    The features and audio are now *correlated* with the mood.
    """
    print(f"Step 1: Simulating {num_samples} samples with *patterns*...")
    data = []
    sample_rate = 22050
    moods = ["Happy", "Sad", "Energetic", "Calm"]
    
    # Define rules for our simulation
    # (Mood: [audio_freq, (dance_min, dance_max), (energy_min, energy_max), (valence_min, valence_max), (tempo_min, tempo_max)])
    rules = {
        "Happy":     [660, (0.6, 1.0), (0.6, 1.0), (0.7, 1.0), (100, 160)],
        "Sad":       [220, (0.1, 0.4), (0.1, 0.4), (0.0, 0.3), ( 60,  90)],
        "Energetic": [880, (0.7, 1.0), (0.8, 1.0), (0.5, 0.9), (120, 180)],
        "Calm":      [440, (0.2, 0.5), (0.2, 0.5), (0.4, 0.7), ( 70, 110)]
    }

    for i in range(num_samples):
        mood = np.random.choice(moods)
        rule = rules[mood]
        
        # 1. Simulate audio based on mood (sine wave at a specific frequency)
        # This gives the CNN a clear pattern to learn.
        duration = 3
        t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
        audio_signal = 0.5 * np.sin(2 * np.pi * rule[0] * t)
        # Add a little noise so it's not *too* perfect
        audio_signal += np.random.uniform(-0.05, 0.05, size=audio_signal.shape)

        # 2. Simulate features based on mood
        # This gives the Random Forest a clear pattern to learn.
        feature_vector = {
            'danceability': np.random.uniform(rule[1][0], rule[1][1]),
            'energy':       np.random.uniform(rule[2][0], rule[2][1]),
            'valence':      np.random.uniform(rule[3][0], rule[3][1]),
            'tempo':        np.random.uniform(rule[4][0], rule[4][1])
        }
        
        data.append({
            "audio_signal": audio_signal,
            "features": feature_vector,
            "mood": mood
        })
        
    return data, sample_rate

# --- PART 2: STAGE 1 - TRAIN THE CNN (Audio -> Mood) ---
# (This section is unchanged, but will now work)

def train_cnn_model(data, sample_rate):
    """
    Trains the Deep Learning CNN on Mel-spectrograms to predict mood.
    """
    print("\n--- STAGE 1: Training Deep Learning CNN on Audio ---")

    # 1. Preprocess: Create spectrograms and labels
    X_spectrograms = []
    y_moods = []

    for item in data:
        # Create Mel-spectrogram (an "image" of the audio)
        # 
        spectrogram = librosa.feature.melspectrogram(y=item["audio_signal"], sr=sample_rate, n_mels=64)
        
        # Resize all spectrograms to a fixed size (e.g., 64x128)
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
    print("Training CNN... (This will be fast)")
    # Increased epochs to 10 for better convergence
    model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0, validation_split=0.1)

    # 4. Evaluate the model on test set
    print("Evaluating CNN on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    target_names = label_encoder.classes_
    print("CNN Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))
    print("CNN Confusion Matrix:")
    print(confusion_matrix(y_test_classes, y_pred_classes))

    print("...CNN Training and Evaluation Complete.")

# --- PART 3: STAGE 2 - TRAIN THE "PROXY" MODEL (Features -> Mood) ---
# (This section is unchanged, but will now work)

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
    proxy_model = RandomForestClassifier(n_estimators=100, random_state=42) # Increased n_estimators
    proxy_model.fit(X_train, y_train)
    
    # 4. Test its accuracy and other metrics
    y_pred = proxy_model.predict(X_test)
    print(f"...Proxy Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Classification report
    target_names = label_encoder.classes_
    print("Proxy Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 5. SAVE THE FINAL MODEL
    joblib.dump(proxy_model, 'mood_from_features_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(label_encoder, 'mood_label_encoder.joblib')
    
    print("... 'mood_from_features_model.joblib' (and helpers) saved!")
    

# --- Main execution ---
if __name__ == "__main__":
    
    # Step 1: Get Data
    # Use 500 samples to give the models enough data
    simulated_data, sample_rate = create_simulated_data(num_samples=500)
    
    # Step 2: Train the big CNN (This will now have high accuracy)
    train_cnn_model(simulated_data, sample_rate)
    
    # Step 3: Train the "Proxy" model (This will also have high accuracy)
    train_proxy_model(simulated_data)
    
    print("\n--- ML Training Complete! ---")
    print("You now have 'mood_from_features_model.joblib', which is ready for the next phase.")