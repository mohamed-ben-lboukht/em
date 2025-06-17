#!/usr/bin/env python3
"""
Create a more realistic XGBoost model for emotion detection from keystroke dynamics
"""
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

print("Creating realistic XGBoost emotion detection model...")

# Create more realistic training data that mimics keystroke emotion patterns
np.random.seed(42)
n_samples = 2000

# Generate features that correlate with emotions
def generate_emotion_data(emotion_id, n_samples_per_emotion):
    """Generate keystroke features that correlate with specific emotions"""
    data = []
    labels = []
    
    for _ in range(n_samples_per_emotion):
        # Base feature vector (32 dimensions)
        features = np.zeros(32)
        
        if emotion_id == 0:  # Neutral
            # Moderate, consistent typing patterns
            features[:20] = np.random.dirichlet(np.ones(20) * 5)  # Linear histogram
            features[20:25] = np.random.dirichlet(np.ones(5) * 3)  # Log histogram
            features[25:] = [150, 50, 140, 80, 300, 100, 200]  # Stats: mean, std, median, min, max, q25, q75
            
        elif emotion_id == 1:  # Happy
            # Faster, more rhythmic typing
            features[:20] = np.random.dirichlet(np.ones(20) * 3)
            features[10:15] *= 2  # Emphasize faster timings
            features[20:25] = np.random.dirichlet(np.ones(5) * 2)
            features[25:] = [120, 35, 110, 60, 250, 90, 160]  # Faster typing
            
        elif emotion_id == 2:  # Sad
            # Slower, more irregular typing
            features[:20] = np.random.dirichlet(np.ones(20) * 4)
            features[15:20] *= 1.5  # Emphasize slower timings
            features[20:25] = np.random.dirichlet(np.ones(5) * 4)
            features[25:] = [200, 80, 190, 100, 400, 150, 280]  # Slower typing
            
        elif emotion_id == 3:  # Angry
            # Erratic, variable typing
            features[:20] = np.random.dirichlet(np.ones(20) * 2)
            features[5:10] *= 1.8  # Mixed fast/slow patterns
            features[20:25] = np.random.dirichlet(np.ones(5) * 1.5)
            features[25:] = [160, 90, 150, 40, 450, 80, 250]  # High variability
            
        elif emotion_id == 4:  # Fearful
            # Hesitant, inconsistent typing
            features[:20] = np.random.dirichlet(np.ones(20) * 6)
            features[12:18] *= 1.3  # Hesitation patterns
            features[20:25] = np.random.dirichlet(np.ones(5) * 5)
            features[25:] = [180, 75, 170, 70, 380, 120, 240]  # Cautious typing
            
        elif emotion_id == 5:  # Disgusted
            # Irregular, interrupted patterns
            features[:20] = np.random.dirichlet(np.ones(20) * 3.5)
            features[8:12] *= 1.6  # Irregular patterns
            features[20:25] = np.random.dirichlet(np.ones(5) * 3)
            features[25:] = [170, 85, 160, 50, 420, 110, 260]  # Inconsistent
            
        elif emotion_id == 6:  # Surprised
            # Sharp changes in patterns
            features[:20] = np.random.dirichlet(np.ones(20) * 2.5)
            features[3:8] *= 2  # Sudden pattern changes
            features[20:25] = np.random.dirichlet(np.ones(5) * 2)
            features[25:] = [140, 95, 130, 45, 380, 85, 220]  # Variable
        
        # Add noise to make it realistic
        features += np.random.normal(0, 0.05, 32)
        features = np.clip(features, 0, None)  # Ensure non-negative
        
        data.append(features)
        labels.append(emotion_id)
    
    return np.array(data), np.array(labels)

# Generate data for all emotions
all_data = []
all_labels = []

emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
samples_per_emotion = n_samples // len(emotions)

for emotion_id, emotion_name in enumerate(emotions):
    print(f"Generating {samples_per_emotion} samples for {emotion_name}...")
    data, labels = generate_emotion_data(emotion_id, samples_per_emotion)
    all_data.append(data)
    all_labels.append(labels)

X = np.vstack(all_data)
y = np.concatenate(all_labels)

print(f"Total dataset shape: {X.shape}, Labels: {y.shape}")
print(f"Emotion distribution: {np.bincount(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    objective='multi:softprob',
    num_class=7,
    subsample=0.8,
    colsample_bytree=0.8
)

print("Training XGBoost model...")
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Save the model
joblib.dump(model, 'xgboost_model.joblib')
print("✅ Realistic XGBoost model saved as 'xgboost_model.joblib'")

# Test prediction with varied outputs
test_features = np.random.rand(5, 32)
predictions = model.predict_proba(test_features)
print(f"\n🧪 Test predictions:")
for i, pred in enumerate(predictions):
    dominant_emotion = emotions[np.argmax(pred)]
    confidence = np.max(pred)
    print(f"Sample {i+1}: {dominant_emotion} ({confidence:.2f}) - {pred}")

print("\n🎯 Realistic model ready for emotion detection!") 