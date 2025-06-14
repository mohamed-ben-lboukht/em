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
    model = joblib.load('svr_model_hist.joblib')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

def compute_histogram_features(keystrokes, bins=10, bins_edges=[0, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300]):
    """Optimized histogram computation with better error handling"""
    try:
        timings = np.array(keystrokes, dtype=float)
        
        # Enhanced validation
        if (len(timings) == 0 or
            np.any(np.isnan(timings)) or
            np.any(np.isinf(timings)) or
            np.any(timings < 0) or
            np.any(timings > 5000)):
            return np.zeros(bins)
            
        # Remove outliers
        q1, q3 = np.percentile(timings, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        timings = timings[(timings >= lower_bound) & (timings <= upper_bound)]
        
        if len(timings) == 0:
            return np.zeros(bins)
            
        if bins_edges is None:
            min_range = max(0, np.min(timings))
            max_range = min(300, np.max(timings))
            bins_edges = np.linspace(min_range, max_range, bins + 1)
        else:
            bins = len(bins_edges) - 1
            
        hist, _ = np.histogram(timings, bins=bins_edges, density=False)
        hist_sum = hist.sum()
        
        if hist_sum == 0:
            return np.zeros(bins)
            
        return hist / hist_sum
    except Exception as e:
        logger.error(f"Error in histogram computation: {e}")
        return np.zeros(bins)

class SimpleKeystrokeCollector:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.keystroke_buffer = deque(maxlen=window_size)
        self.last_prediction = None
        self.min_data_points = 3
        
    def add_keystroke_data(self, timings):
        """Add keystroke timing data to the buffer"""
        if not timings:
            return
            
        # Filter valid timings
        valid_timings = []
        for t in timings:
            if isinstance(t, (int, float)) and 0 < t < 2000:
                valid_timings.append(t)
        
        if valid_timings:
            self.keystroke_buffer.extend(valid_timings)
        
    def get_features(self):
        """Extract histogram features from keystroke buffer"""
        if len(self.keystroke_buffer) < self.min_data_points:
            return np.zeros(10).reshape(1, -1)
        
        keystroke_list = list(self.keystroke_buffer)
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
            
            # Process prediction
            emotions = self.process_prediction(prediction)
            self.last_prediction = emotions
            
            return emotions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def process_prediction(self, prediction):
        """Process prediction into emotion distribution"""
        try:
            # Handle different prediction formats
            if isinstance(prediction, (list, np.ndarray)):
                if len(prediction) == len(EMOTIONS):
                    emotion_scores = np.array(prediction)
                else:
                    emotion_scores = self.map_single_to_distribution(prediction[0] if len(prediction) > 0 else 0)
            else:
                emotion_scores = self.map_single_to_distribution(prediction)
            
            # Ensure valid probability distribution
            emotion_scores = np.clip(emotion_scores, 0, 1)
            if emotion_scores.sum() > 0:
                emotion_scores = emotion_scores / emotion_scores.sum()
            else:
                emotion_scores = np.zeros(len(EMOTIONS))
                emotion_scores[0] = 1.0  # Default to neutral
            
            # Create emotion dictionary
            emotions = {emotion: float(score) for emotion, score in zip(EMOTIONS, emotion_scores)}
            return emotions
            
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            return {emotion: 1.0 if emotion == 'neutral' else 0.0 for emotion in EMOTIONS}
    
    def map_single_to_distribution(self, value):
        """Map single prediction value to emotion distribution"""
        try:
            # Normalize value to 0-1 range
            normalized_value = max(0, min(1, (value + 1) / 2)) if isinstance(value, (int, float)) else 0.5
            
            # Create distribution based on value
            emotion_scores = np.zeros(len(EMOTIONS))
            
            if normalized_value < 0.2:
                emotion_scores[EMOTIONS.index('sad')] = 0.7
                emotion_scores[EMOTIONS.index('neutral')] = 0.3
            elif normalized_value < 0.4:
                emotion_scores[EMOTIONS.index('fearful')] = 0.6
                emotion_scores[EMOTIONS.index('neutral')] = 0.4
            elif normalized_value < 0.6:
                emotion_scores[EMOTIONS.index('neutral')] = 1.0
            elif normalized_value < 0.8:
                emotion_scores[EMOTIONS.index('happy')] = 0.7
                emotion_scores[EMOTIONS.index('neutral')] = 0.3
            else:
                emotion_scores[EMOTIONS.index('surprised')] = 0.6
                emotion_scores[EMOTIONS.index('happy')] = 0.4
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Error mapping single value: {e}")
            emotion_scores = np.zeros(len(EMOTIONS))
            emotion_scores[0] = 1.0  # Default to neutral
            return emotion_scores

# Global keystroke collector
keystroke_collector = SimpleKeystrokeCollector()

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
        
        if not timings:
            return
        
        # Add to collector
        keystroke_collector.add_keystroke_data(timings)
        
        # Make prediction
        emotions = keystroke_collector.predict_emotion()
        
        if 'error' not in emotions:
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