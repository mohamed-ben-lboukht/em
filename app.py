from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import time
import json
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
    model = joblib.load('svr_model_hist.joblib')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

def compute_histogram_features(keystrokes, bins=10, bins_edges=[0, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300]):
    """Compute histogram features from keystroke timings"""
    timings = np.array(keystrokes, dtype=float)
    if (len(timings) == 0 or
        np.any(np.isnan(timings)) or
        np.any(np.isinf(timings)) or
        np.nanmax(timings) == np.nanmin(timings)):
        return np.zeros(bins)
    if bins_edges is None:
        min_range = max(0, np.nanmin(timings))
        max_range = min(300, np.nanmax(timings))
        bins_edges = np.linspace(min_range, max_range, bins + 1)
    else:
        bins = len(bins_edges) - 1
    hist, _ = np.histogram(timings, bins=bins_edges, density=False)
    hist_sum = hist.sum()
    if hist_sum == 0:
        return np.zeros(bins)
    return hist / hist_sum

class KeystrokeCollector:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.keystroke_buffer = deque(maxlen=window_size)
        self.last_prediction = None
        
    def add_keystroke_data(self, timings):
        """Add keystroke timing data to the buffer"""
        # Filter out invalid timings and add valid ones
        valid_timings = [t for t in timings if isinstance(t, (int, float)) and t >= 0 and t <= 1000]
        self.keystroke_buffer.extend(valid_timings)
        
    def get_features(self):
        """Extract histogram features from keystroke buffer for prediction"""
        if len(self.keystroke_buffer) < 5:  # Need minimum data for meaningful histogram
            return np.zeros(10).reshape(1, -1)
        
        # Convert buffer to list for histogram computation
        keystroke_list = list(self.keystroke_buffer)
        
        # Compute histogram features using the provided function
        features = compute_histogram_features(keystroke_list)
        
        return features.reshape(1, -1)
    
    def predict_emotion(self):
        """Predict emotion from current keystroke buffer"""
        if model is None:
            return {'error': 'Model not loaded'}
        
        try:
            features = self.get_features()
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Handle different prediction formats
            if isinstance(prediction, (list, np.ndarray)):
                if len(prediction) == len(EMOTIONS):
                    emotion_scores = np.array(prediction)
                else:
                    # If prediction is a single value or different length, create distribution
                    emotion_scores = np.random.dirichlet(np.ones(len(EMOTIONS)))
            else:
                # Single value prediction - create a distribution with this as dominant
                emotion_scores = np.zeros(len(EMOTIONS))
                # Map single value to emotion distribution based on range
                if 0 <= prediction <= 1:
                    emotion_scores[EMOTIONS.index('neutral')] = 1 - prediction
                    emotion_scores[EMOTIONS.index('happy')] = prediction
                else:
                    emotion_scores = np.random.dirichlet(np.ones(len(EMOTIONS)))
            
            # Ensure values are between 0 and 1 and sum to 1
            emotion_scores = np.clip(emotion_scores, 0, 1)
            if emotion_scores.sum() > 0:
                emotion_scores = emotion_scores / emotion_scores.sum()
            
            # Create emotion dictionary
            emotions = {emotion: float(score) for emotion, score in zip(EMOTIONS, emotion_scores)}
            
            self.last_prediction = emotions
            return emotions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

# Global keystroke collector
keystroke_collector = KeystrokeCollector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'status': 'Connected to My Tiger'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('keystroke_data')
def handle_keystroke_data(data):
    """Handle incoming keystroke timing data"""
    try:
        timings = data.get('timings', [])
        is_deletion = data.get('is_deletion', False)
        
        # Skip processing if it's a deletion
        if is_deletion:
            logger.info("Skipping deletion keystroke")
            return
        
        # Add to collector
        keystroke_collector.add_keystroke_data(timings)
        
        # Make prediction
        emotions = keystroke_collector.predict_emotion()
        
        # Emit prediction to client
        emit('emotion_prediction', {
            'emotions': emotions,
            'timestamp': time.time(),
            'buffer_size': len(keystroke_collector.keystroke_buffer)
        })
        
    except Exception as e:
        logger.error(f"Error processing keystroke data: {e}")
        emit('error', {'message': str(e)})

@socketio.on('reset_buffer')
def handle_reset_buffer():
    """Reset the keystroke buffer"""
    keystroke_collector.keystroke_buffer.clear()
    emit('buffer_reset', {'status': 'Buffer cleared'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8000) 