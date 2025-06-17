from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import time
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_tiger_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the trained model
try:
    model = joblib.load('xgboost_model.joblib')
    logger.info("XGBoost model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

class RobustScaler:
    def __init__(self):
        self.means_ = None
        self.scales_ = None

    def fit_transform(self, X):
        self.means_ = np.mean(X, axis=0)
        self.scales_ = np.std(X, axis=0)
        self.scales_[self.scales_ == 0] = 1e-10  # Éviter division par zéro
        return (X - self.means_) / self.scales_

    def transform(self, X):
        if self.means_ is None or self.scales_ is None:
            # Si pas encore entrainé, utiliser des valeurs par défaut
            return X
        return (X - self.means_) / self.scales_

    def inverse_transform(self, X):
        if self.means_ is None or self.scales_ is None:
            return X
        return X * self.scales_ + self.means_

def extract_32_features_for_model(keystroke_data):
    """Extract 32 features for XGBoost model with proper normalization"""
    try:
        # Combine all timing data
        all_timings = []
        pp_timings = keystroke_data.get('pp', [])
        pr_timings = keystroke_data.get('pr', [])
        rp_timings = keystroke_data.get('rp', [])
        rr_timings = keystroke_data.get('rr', [])
        
        # Collect all timings
        for timings in [pp_timings, pr_timings, rp_timings, rr_timings]:
            if timings:
                all_timings.extend(timings[-15:])  # Last 15 for each category
        
        if not all_timings or len(all_timings) < 2:
            # Return default neutral features
            return np.array([0.5, 0.3, 0.15, 0.05] * 8).reshape(1, -1)
        
        timings = np.array(all_timings, dtype=float)
        timings = timings[~np.isnan(timings)]
        
        if len(timings) < 2:
            return np.array([0.5, 0.3, 0.15, 0.05] * 8).reshape(1, -1)
        
        # Calculate intervals
        intervals = np.diff(np.sort(timings))
        
        # 32 features for XGBoost model
        features = []
        
        # 1. Speed distributions (20 features - linear bins)
        speed_bins = [0, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 280, 320, 360, 400, 500, 700, 1000, np.inf]
        hist, _ = np.histogram(intervals, bins=speed_bins)
        hist_norm = hist / len(intervals) if len(intervals) > 0 else np.zeros(20)
        features.extend(hist_norm)
        
        # 2. Log-scale distributions (5 features)
        log_bins = np.logspace(np.log10(10), np.log10(2000), 6)
        hist_log, _ = np.histogram(intervals, bins=log_bins)
        hist_log_norm = hist_log / len(intervals) if len(intervals) > 0 else np.zeros(5)
        features.extend(hist_log_norm)
        
        # 3. Statistical features (7 features)
        stats = [
            np.mean(intervals),
            np.std(intervals) if len(intervals) > 1 else 0,
            np.median(intervals),
            np.min(intervals),
            np.max(intervals),
            np.percentile(intervals, 25),
            np.percentile(intervals, 75)
        ]
        features.extend(stats)
        
        return np.array(features[:32]).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Error extracting 32 features: {e}")
        return np.array([0.5, 0.3, 0.15, 0.05] * 8).reshape(1, -1)

def analyze_emotion_from_patterns(pp_timings, pr_timings, rp_timings, rr_timings):
    """Analyze emotions based on immediate keystroke patterns - FAST and RESPONSIVE"""
    
    # Combine all timing data
    all_timings = []
    if pp_timings: all_timings.extend(pp_timings[-10:])  # Only last 10 events for speed
    if pr_timings: all_timings.extend(pr_timings[-10:])
    if rp_timings: all_timings.extend(rp_timings[-10:])
    if rr_timings: all_timings.extend(rr_timings[-10:])
    
    if not all_timings:
        return {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
    
    timings = np.array(all_timings)
    timings = timings[~np.isnan(timings)]
    
    if len(timings) < 2:
        return {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
    
    # Calculate real-time emotion indicators
    intervals = np.diff(np.sort(timings))
    
    # Emotion detection based on typing patterns
    avg_interval = np.mean(intervals)
    std_interval = np.std(intervals) if len(intervals) > 1 else 0
    variability = std_interval / avg_interval if avg_interval > 0 else 0
    
    # Fast typing bursts
    fast_count = np.sum(intervals < 80)
    slow_count = np.sum(intervals > 300)
    
    # Initialize emotion scores
    emotions = {emotion: 0.0 for emotion in EMOTIONS}
    
    # Dynamic emotion detection
    if avg_interval < 100 and variability < 0.3:  # Fast, consistent
        emotions['happy'] = 0.6 + np.random.normal(0, 0.1)
        emotions['surprised'] = 0.2 + np.random.normal(0, 0.05)
        emotions['neutral'] = 0.2 + np.random.normal(0, 0.05)
    
    elif avg_interval > 250 and variability < 0.2:  # Slow, consistent
        emotions['sad'] = 0.5 + np.random.normal(0, 0.1)
        emotions['fearful'] = 0.2 + np.random.normal(0, 0.05)
        emotions['neutral'] = 0.3 + np.random.normal(0, 0.05)
    
    elif variability > 0.8:  # High variability
        emotions['angry'] = 0.6 + np.random.normal(0, 0.1)
        emotions['surprised'] = 0.2 + np.random.normal(0, 0.05)
        emotions['neutral'] = 0.2 + np.random.normal(0, 0.05)
    
    elif fast_count > len(intervals) * 0.7:  # Mostly fast
        emotions['surprised'] = 0.5 + np.random.normal(0, 0.1)
        emotions['happy'] = 0.3 + np.random.normal(0, 0.05)
        emotions['neutral'] = 0.2 + np.random.normal(0, 0.05)
    
    elif slow_count > len(intervals) * 0.6:  # Mostly slow
        emotions['fearful'] = 0.4 + np.random.normal(0, 0.1)
        emotions['sad'] = 0.3 + np.random.normal(0, 0.05)
        emotions['neutral'] = 0.3 + np.random.normal(0, 0.05)
    
    elif 150 < avg_interval < 200 and variability > 0.4:  # Moderate with some irregularity
        emotions['disgusted'] = 0.4 + np.random.normal(0, 0.1)
        emotions['angry'] = 0.2 + np.random.normal(0, 0.05)
        emotions['neutral'] = 0.4 + np.random.normal(0, 0.05)
    
    else:  # Default to neutral with slight variations
        emotions['neutral'] = 0.6 + np.random.normal(0, 0.1)
        emotions['happy'] = 0.15 + np.random.normal(0, 0.03)
        emotions['sad'] = 0.1 + np.random.normal(0, 0.02)
        emotions['surprised'] = 0.1 + np.random.normal(0, 0.02)
        emotions['angry'] = 0.05 + np.random.normal(0, 0.01)
    
    # Normalize and ensure positive values
    total = sum(max(0, score) for score in emotions.values())
    if total > 0:
        emotions = {emotion: max(0, score) / total for emotion, score in emotions.items()}
    else:
        emotions = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
    
    return emotions

class HybridEmotionDetector:
    def __init__(self):
        self.last_emotions = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
        self.emotion_momentum = 0.2  # Reduced for faster response
        self.scaler = RobustScaler()
        self.scaler_trained = False
        self.feature_buffer = deque(maxlen=50)  # Buffer for training scaler
        
    def train_scaler_if_needed(self, features):
        """Train the scaler with accumulated features"""
        if not self.scaler_trained and len(self.feature_buffer) >= 20:
            # Train scaler with buffered features
            training_data = np.vstack(list(self.feature_buffer))
            self.scaler.fit_transform(training_data)
            self.scaler_trained = True
            logger.info("📊 RobustScaler trained with {} samples".format(len(self.feature_buffer)))
        
    def predict_emotion_hybrid(self, keystroke_data):
        """Hybrid prediction: Fast rule-based + XGBoost with normalization"""
        try:
            # 1. Fast prediction for real-time responsiveness
            fast_emotions = analyze_emotion_from_patterns(
                keystroke_data.get('pp', []),
                keystroke_data.get('pr', []),
                keystroke_data.get('rp', []),
                keystroke_data.get('rr', [])
            )
            
            # 2. XGBoost prediction with normalization (if model available)
            model_emotions = fast_emotions.copy()  # Fallback
            
            if model is not None:
                try:
                    # Extract 32 features for model
                    features = extract_32_features_for_model(keystroke_data)
                    
                    # Add to buffer for scaler training
                    self.feature_buffer.append(features[0])
                    self.train_scaler_if_needed(features)
                    
                    # Apply normalization if scaler is trained
                    if self.scaler_trained:
                        features_normalized = self.scaler.transform(features)
                    else:
                        features_normalized = features
                    
                    # Get model prediction
                    prediction_proba = model.predict_proba(features_normalized)[0]
                    model_emotions = {emotion: float(prob) for emotion, prob in zip(EMOTIONS, prediction_proba)}
                    
                except Exception as model_error:
                    logger.warning(f"Model prediction failed, using fast method: {model_error}")
            
            # 3. Blend fast and model predictions (70% fast, 30% model for responsiveness)
            blended_emotions = {}
            for emotion in EMOTIONS:
                fast_score = fast_emotions[emotion] * 0.7
                model_score = model_emotions[emotion] * 0.3
                blended_emotions[emotion] = fast_score + model_score
            
            # 4. Apply momentum for smooth transitions
            final_emotions = {}
            for emotion in EMOTIONS:
                momentum_value = self.last_emotions[emotion] * self.emotion_momentum
                current_value = blended_emotions[emotion] * (1 - self.emotion_momentum)
                final_emotions[emotion] = momentum_value + current_value
            
            # 5. Normalize
            total = sum(final_emotions.values())
            if total > 0:
                final_emotions = {emotion: score / total for emotion, score in final_emotions.items()}
            
            # Update last emotions
            self.last_emotions = final_emotions.copy()
            
            return final_emotions
            
        except Exception as e:
            logger.error(f"Hybrid emotion prediction error: {e}")
            return {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}

# Global hybrid emotion detector
emotion_detector = HybridEmotionDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'HybridEmotionDetector',
        'features': 'Fast + XGBoost with RobustScaler',
        'scaler_trained': emotion_detector.scaler_trained,
        'response_time': 'Ultra-fast',
        'timestamp': time.time()
    })

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'status': 'Connected to My Tiger - Hybrid Fast Emotion Detection'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('keystroke_data')
def handle_keystroke_data(data):
    """Handle incoming keystroke data with hybrid ultra-fast emotion detection"""
    try:
        features = data.get('features', {})
        
        if not features:
            return
        
        # Hybrid emotion prediction (fast + normalized model)
        emotions = emotion_detector.predict_emotion_hybrid(features)
        
        # Emit immediately for responsiveness
        emit('emotion_prediction', {
            'emotions': emotions,
            'timestamp': time.time(),
            'mode': 'hybrid-real-time',
            'scaler_trained': emotion_detector.scaler_trained,
            'features_summary': {
                'pp_count': len(features.get('pp', [])),
                'pr_count': len(features.get('pr', [])),
                'rp_count': len(features.get('rp', [])),
                'rr_count': len(features.get('rr', []))
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing keystroke data: {e}")
        emit('error', {'message': str(e)})

@socketio.on('reset_buffer')
def handle_reset_buffer():
    """Reset the emotion state and scaler"""
    emotion_detector.last_emotions = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
    emotion_detector.feature_buffer.clear()
    emotion_detector.scaler_trained = False
    emotion_detector.scaler = RobustScaler()
    emit('buffer_reset', {'status': 'Emotion state and scaler reset'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8000) 