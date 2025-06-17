#!/usr/bin/env python3
"""
Create a dummy XGBoost model for testing the emotion detection system
"""
import xgboost as xgb
import numpy as np
import joblib
from sklearn.datasets import make_classification

# Create dummy training data
# 32 features (as expected by our system)
# 7 classes (for 7 emotions)
X, y = make_classification(
    n_samples=1000,
    n_features=32,  # Our PP/PR/RP/RR interval features
    n_classes=7,    # 7 emotions
    n_informative=20,
    n_redundant=5,
    n_clusters_per_class=1,
    random_state=42
)

print("Creating dummy XGBoost model...")
print(f"Training data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Create and train XGBoost classifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    objective='multi:softprob',  # For probability output
    num_class=7
)

# Train the model
model.fit(X, y)

# Save the model
joblib.dump(model, 'xgboost_model.joblib')
print("✅ Dummy XGBoost model saved as 'xgboost_model.joblib'")

# Test prediction
test_features = np.random.rand(1, 32)
prediction = model.predict_proba(test_features)
print(f"Test prediction shape: {prediction.shape}")
print(f"Test prediction: {prediction[0]}")

print("\n🎯 Model is ready for emotion detection!") 