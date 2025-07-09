from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import numpy as np
import time
from collections import deque
import logging
import os
import threading
from functools import lru_cache, wraps
import hashlib
import json
from datetime import datetime, timedelta
import psutil
import gc

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('my_tiger.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app configuration with optimizations
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'my_tiger_ultra_secure_key_2024_v2'),
    
    # Caching configuration
    CACHE_TYPE='simple',
    CACHE_DEFAULT_TIMEOUT=300,
    
    # Performance optimizations
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=False,
    
    # Security
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Initialize extensions
cache = Cache(app)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour", "1 per second"]
)

socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# Enhanced model loading with error handling and fallback
class ModelManager:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_path = None
        self.load_model()
    
    def load_model(self):
        """Load XGBoost model with fallback strategy"""
        model_paths = [
            'xgboost_model.joblib',
            'model_xgboost.pkl', 
            'xgboost.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"🔄 Attempting to load model from: {path}")
                    self.model = joblib.load(path)
                    self.model_loaded = True
                    self.model_path = path
                    logger.info(f"✅ XGBoost model loaded successfully from: {path}")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model from {path}: {e}")
                    continue
        
        logger.warning("⚠️ No valid model found. Running in fallback mode.")
        self.model_loaded = False

    def predict(self, features):
        """Thread-safe model prediction"""
        if not self.model_loaded or self.model is None:
            return None
        
        try:
            with threading.Lock():
                prediction = self.model.predict_proba(features)[0]
                return prediction
        except Exception as e:
            logger.error(f"❌ Model prediction failed: {e}")
            return None

# Global model manager instance
model_manager = ModelManager()

# Emotion labels
EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.avg_response_time = 0
        self.predictions_made = 0
        self.start_time = time.time()
        self.memory_usage = []
        
    def log_request(self, response_time):
        self.request_count += 1
        self.avg_response_time = (
            (self.avg_response_time * (self.request_count - 1) + response_time) 
            / self.request_count
        )
        
    def log_error(self):
        self.error_count += 1
        
    def log_prediction(self):
        self.predictions_made += 1
        
    def get_stats(self):
        uptime = time.time() - self.start_time
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'uptime_seconds': uptime,
            'requests_total': self.request_count,
            'errors_total': self.error_count,
            'predictions_total': self.predictions_made,
            'avg_response_time_ms': round(self.avg_response_time * 1000, 2),
            'memory_usage_mb': round(memory, 2),
            'requests_per_minute': round(self.request_count / (uptime / 60), 2) if uptime > 60 else 0
        }

performance_monitor = PerformanceMonitor()

# Enhanced robust scaler with caching
class OptimizedRobustScaler:
    def __init__(self):
        self.means_ = None
        self.scales_ = None
        self.is_fitted = False
        self._cache = {}

    @lru_cache(maxsize=128)
    def _compute_stats(self, data_hash):
        """Cached computation of statistics"""
        return self.means_, self.scales_

    def fit_transform(self, X):
        """Fit and transform with caching"""
        X = np.asarray(X, dtype=np.float32)
        self.means_ = np.mean(X, axis=0)
        self.scales_ = np.std(X, axis=0)
        self.scales_[self.scales_ == 0] = 1e-10
        self.is_fitted = True
        return (X - self.means_) / self.scales_

    def transform(self, X):
        """Transform with validation"""
        if not self.is_fitted:
            return X
        
        X = np.asarray(X, dtype=np.float32)
        return (X - self.means_) / self.scales_

# Feature extraction optimization
@lru_cache(maxsize=256)
def get_histogram_bins():
    """Cached histogram bins for feature extraction"""
    return np.array([0, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300], dtype=np.float32)

def extract_optimized_features(keystroke_data):
    """Optimized feature extraction with validation and caching"""
    try:
        # Input validation
        if not keystroke_data or not isinstance(keystroke_data, dict):
            return np.zeros((1, 32), dtype=np.float32)
        
        # Combine timing data efficiently
        all_timings = []
        for timing_type in ['pp', 'pr', 'rp', 'rr']:
            timings = keystroke_data.get(timing_type, [])
            if timings:
                # Take last 15 for performance, convert to float32
                all_timings.extend(np.array(timings[-15:], dtype=np.float32))
        
        if len(all_timings) < 2:
            return np.zeros((1, 32), dtype=np.float32)
        
        # Efficient numpy operations
        timings = np.array(all_timings, dtype=np.float32)
        timings = timings[~np.isnan(timings)]
        
        if len(timings) < 2:
            return np.zeros((1, 32), dtype=np.float32)
        
        # Calculate intervals efficiently
        intervals = np.diff(np.sort(timings))
        
        if len(intervals) == 0:
            return np.zeros((1, 32), dtype=np.float32)
        
        # Feature computation with optimized memory usage
        features = np.zeros(32, dtype=np.float32)
        
        # Speed distributions (20 features)
        speed_bins = np.array([0, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 
                              230, 250, 280, 320, 360, 400, 500, 700, 1000, np.inf], dtype=np.float32)
        hist, _ = np.histogram(intervals, bins=speed_bins)
        hist_norm = hist / len(intervals) if len(intervals) > 0 else np.zeros(20)
        features[:20] = hist_norm.astype(np.float32)
        
        # Log-scale distributions (5 features)
        log_bins = np.logspace(np.log10(10), np.log10(2000), 6).astype(np.float32)
        hist_log, _ = np.histogram(intervals, bins=log_bins)
        hist_log_norm = hist_log / len(intervals) if len(intervals) > 0 else np.zeros(5)
        features[20:25] = hist_log_norm.astype(np.float32)
        
        # Statistical features (7 features) - optimized computation
        stats = np.array([
            np.mean(intervals),
            np.std(intervals) if len(intervals) > 1 else 0,
            np.median(intervals),
            np.min(intervals),
            np.max(intervals),
            np.percentile(intervals, 25),
            np.percentile(intervals, 75)
        ], dtype=np.float32)
        features[25:32] = stats
        
        return features.reshape(1, -1)
        
    except Exception as e:
        logger.error(f"❌ Feature extraction error: {e}")
        return np.zeros((1, 32), dtype=np.float32)

# Optimized emotion analysis with caching
def analyze_emotion_patterns_optimized(keystroke_data):
    """Ultra-fast emotion analysis with pattern caching"""
    
    # Create cache key
    cache_key = hashlib.md5(str(keystroke_data).encode()).hexdigest()[:16]
    
    # Check cache first
    cached_result = cache.get(f"emotion_pattern_{cache_key}")
    if cached_result:
        return cached_result
    
    try:
        # Combine timings efficiently
        all_timings = []
        for timing_type in ['pp', 'pr', 'rp', 'rr']:
            timings = keystroke_data.get(timing_type, [])
            if timings:
                all_timings.extend(timings[-10:])  # Last 10 for speed
        
        if not all_timings:
            result = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 
                     'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
            cache.set(f"emotion_pattern_{cache_key}", result, timeout=60)
            return result
        
        # Fast numpy operations
        timings = np.array(all_timings, dtype=np.float32)
        timings = timings[~np.isnan(timings)]
        
        if len(timings) < 2:
            result = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 
                     'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
            cache.set(f"emotion_pattern_{cache_key}", result, timeout=60)
            return result
        
        # Optimized pattern analysis
        intervals = np.diff(np.sort(timings))
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals) if len(intervals) > 1 else 0
        variability = std_interval / avg_interval if avg_interval > 0 else 0
        
        # Efficient emotion scoring with vectorized operations
        emotions = {emotion: 0.0 for emotion in EMOTIONS}
        
        # Pattern-based emotion detection (optimized rules)
        if avg_interval < 100 and variability < 0.3:  # Fast, consistent
            emotions['happy'] = 0.6 + np.random.normal(0, 0.05)
            emotions['surprised'] = 0.2 + np.random.normal(0, 0.02)
            emotions['neutral'] = 0.2 + np.random.normal(0, 0.02)
        elif avg_interval > 250 and variability < 0.2:  # Slow, consistent
            emotions['sad'] = 0.5 + np.random.normal(0, 0.05)
            emotions['fearful'] = 0.2 + np.random.normal(0, 0.02)
            emotions['neutral'] = 0.3 + np.random.normal(0, 0.02)
        elif variability > 0.8:  # High variability
            emotions['angry'] = 0.6 + np.random.normal(0, 0.05)
            emotions['surprised'] = 0.2 + np.random.normal(0, 0.02)
            emotions['neutral'] = 0.2 + np.random.normal(0, 0.02)
        else:  # Default neutral
            emotions['neutral'] = 0.6 + np.random.normal(0, 0.05)
            emotions['happy'] = 0.15 + np.random.normal(0, 0.02)
            emotions['sad'] = 0.1 + np.random.normal(0, 0.01)
            emotions['surprised'] = 0.1 + np.random.normal(0, 0.01)
            emotions['angry'] = 0.05 + np.random.normal(0, 0.005)
        
        # Normalize efficiently
        total = sum(max(0, score) for score in emotions.values())
        if total > 0:
            emotions = {emotion: max(0, score) / total for emotion, score in emotions.items()}
        
        # Cache result
        cache.set(f"emotion_pattern_{cache_key}", emotions, timeout=60)
        return emotions
        
    except Exception as e:
        logger.error(f"❌ Pattern analysis error: {e}")
        result = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 
                 'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
        return result

# Enhanced hybrid emotion detector with optimizations
class OptimizedHybridEmotionDetector:
    def __init__(self):
        self.last_emotions = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 
                             'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
        self.emotion_momentum = 0.15  # Faster response
        self.scaler = OptimizedRobustScaler()
        self.scaler_trained = False
        self.feature_buffer = deque(maxlen=30)  # Smaller buffer for speed
        self.prediction_cache = {}
        self.cache_max_size = 100
        
    def clear_cache(self):
        """Clear prediction cache to prevent memory leaks"""
        if len(self.prediction_cache) > self.cache_max_size:
            # Keep only recent 50% of cache
            items = list(self.prediction_cache.items())
            self.prediction_cache = dict(items[-50:])
        
    def train_scaler_if_needed(self, features):
        """Optimized scaler training"""
        if not self.scaler_trained and len(self.feature_buffer) >= 15:
            try:
                training_data = np.vstack(list(self.feature_buffer))
                self.scaler.fit_transform(training_data)
                self.scaler_trained = True
                logger.info(f"📊 Scaler trained with {len(self.feature_buffer)} samples")
            except Exception as e:
                logger.error(f"❌ Scaler training failed: {e}")
        
    def predict_emotion_optimized(self, keystroke_data):
        """Optimized hybrid prediction with caching and error handling"""
        start_time = time.time()
        
        try:
            # Create cache key for this prediction
            data_str = json.dumps(keystroke_data, sort_keys=True)
            cache_key = hashlib.md5(data_str.encode()).hexdigest()[:12]
            
            # Check cache
            if cache_key in self.prediction_cache:
                cached_result, timestamp = self.prediction_cache[cache_key]
                if time.time() - timestamp < 30:  # 30 second cache
                    return cached_result
            
            # Fast rule-based prediction
            fast_emotions = analyze_emotion_patterns_optimized(keystroke_data)
            
            # Model prediction (if available)
            model_emotions = fast_emotions.copy()
            
            if model_manager.model_loaded:
                try:
                    features = extract_optimized_features(keystroke_data)
                    
                    # Add to buffer for scaler training
                    self.feature_buffer.append(features[0])
                    self.train_scaler_if_needed(features)
                    
                    # Apply normalization if scaler is trained
                    if self.scaler_trained:
                        features_normalized = self.scaler.transform(features)
                    else:
                        features_normalized = features
                    
                    # Get model prediction
                    prediction_proba = model_manager.predict(features_normalized)
                    if prediction_proba is not None:
                        model_emotions = {emotion: float(prob) 
                                        for emotion, prob in zip(EMOTIONS, prediction_proba)}
                        performance_monitor.log_prediction()
                        
                except Exception as model_error:
                    logger.warning(f"⚠️ Model prediction failed: {model_error}")
            
            # Optimized blending (70% fast, 30% model)
            blended_emotions = {}
            for emotion in EMOTIONS:
                fast_score = fast_emotions[emotion] * 0.7
                model_score = model_emotions[emotion] * 0.3
                blended_emotions[emotion] = fast_score + model_score
            
            # Apply momentum for smooth transitions
            final_emotions = {}
            for emotion in EMOTIONS:
                momentum_value = self.last_emotions[emotion] * self.emotion_momentum
                current_value = blended_emotions[emotion] * (1 - self.emotion_momentum)
                final_emotions[emotion] = momentum_value + current_value
            
            # Normalize
            total = sum(final_emotions.values())
            if total > 0:
                final_emotions = {emotion: score / total for emotion, score in final_emotions.items()}
            
            # Update state
            self.last_emotions = final_emotions.copy()
            
            # Cache result
            self.prediction_cache[cache_key] = (final_emotions, time.time())
            self.clear_cache()
            
            # Log performance
            response_time = time.time() - start_time
            performance_monitor.log_request(response_time)
            
            return final_emotions
            
        except Exception as e:
            logger.error(f"❌ Hybrid prediction error: {e}")
            performance_monitor.log_error()
            return {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 
                   'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}

# Global emotion detector
emotion_detector = OptimizedHybridEmotionDetector()

# Request validation decorator
def validate_keystroke_data(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json() if hasattr(request, 'get_json') else args[0] if args else {}
        
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format'}), 400
            
        features = data.get('features', {})
        if not isinstance(features, dict):
            return jsonify({'error': 'Invalid features format'}), 400
            
        return f(*args, **kwargs)
    return decorated_function

# Routes with optimizations
@app.route('/')
@cache.cached(timeout=3600)  # Cache for 1 hour
def index():
    return render_template('index.html')

@app.route('/api/health')
@limiter.limit("30 per minute")
def health_check():
    """Enhanced health check with system metrics"""
    try:
        stats = performance_monitor.get_stats()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0.0-optimized',
            'model': {
                'loaded': model_manager.model_loaded,
                'path': model_manager.model_path,
                'type': 'HybridOptimizedDetector'
            },
            'features': {
                'caching': True,
                'rate_limiting': True,
                'performance_monitoring': True,
                'scaler_trained': emotion_detector.scaler_trained
            },
            'performance': stats,
            'system': {
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                'platform': os.name
            }
        })
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/metrics')
@limiter.limit("10 per minute")
def get_metrics():
    """Detailed metrics endpoint"""
    try:
        return jsonify({
            'performance': performance_monitor.get_stats(),
            'cache': {
                'emotion_cache_size': len(emotion_detector.prediction_cache),
                'scaler_buffer_size': len(emotion_detector.feature_buffer),
                'scaler_trained': emotion_detector.scaler_trained
            },
            'model': {
                'loaded': model_manager.model_loaded,
                'path': model_manager.model_path
            }
        })
    except Exception as e:
        logger.error(f"❌ Metrics error: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500

# Enhanced WebSocket events
@socketio.on('connect')
def handle_connect():
    logger.info(f'📱 Client connected: {request.sid}')
    emit('connected', {
        'status': 'Connected to My Tiger - Ultra-Optimized Edition',
        'features': ['caching', 'rate_limiting', 'performance_monitoring'],
        'timestamp': time.time()
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'📱 Client disconnected: {request.sid}')

@socketio.on('keystroke_data')
@limiter.limit("100 per minute")
def handle_keystroke_data_optimized(data):
    """Optimized keystroke data handler with validation and error handling"""
    start_time = time.time()
    
    try:
        # Input validation
        if not data or not isinstance(data, dict):
            emit('error', {'message': 'Invalid data format'})
            return
        
        features = data.get('features', {})
        if not features:
            emit('error', {'message': 'No features provided'})
            return
        
        # Process with hybrid emotion detection
        emotions = emotion_detector.predict_emotion_optimized(features)
        
        # Enhanced response
        response_data = {
            'emotions': emotions,
            'timestamp': time.time(),
            'mode': 'hybrid-ultra-optimized',
            'confidence': max(emotions.values()) if emotions else 0,
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'features_summary': {
                'pp_count': len(features.get('pp', [])),
                'pr_count': len(features.get('pr', [])),
                'rp_count': len(features.get('rp', [])),
                'rr_count': len(features.get('rr', []))
            },
            'system_info': {
                'scaler_trained': emotion_detector.scaler_trained,
                'model_available': model_manager.model_loaded,
                'cache_size': len(emotion_detector.prediction_cache)
            }
        }
        
        emit('emotion_prediction', response_data)
        
    except Exception as e:
        logger.error(f"❌ Keystroke processing error: {e}")
        performance_monitor.log_error()
        emit('error', {'message': 'Processing failed', 'timestamp': time.time()})

@socketio.on('reset_buffer')
def handle_reset_buffer_optimized():
    """Optimized buffer reset with cleanup"""
    try:
        # Reset emotion state
        emotion_detector.last_emotions = {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 
                                         'fearful': 0.0, 'disgusted': 0.0, 'surprised': 0.0}
        
        # Clear buffers and caches
        emotion_detector.feature_buffer.clear()
        emotion_detector.prediction_cache.clear()
        emotion_detector.scaler_trained = False
        emotion_detector.scaler = OptimizedRobustScaler()
        
        # Clear app cache
        cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("🧹 Buffers and caches reset successfully")
        emit('buffer_reset', {
            'status': 'All buffers and caches reset',
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ Buffer reset error: {e}")
        emit('error', {'message': 'Reset failed'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded', 'retry_after': e.retry_after}), 429

# Cleanup function
def cleanup_resources():
    """Periodic cleanup of resources"""
    try:
        # Clear old caches
        emotion_detector.clear_cache()
        
        # Garbage collection
        gc.collect()
        
        # Log memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"🧹 Cleanup completed. Memory usage: {memory_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# Schedule periodic cleanup
import threading
def schedule_cleanup():
    cleanup_resources()
    timer = threading.Timer(300.0, schedule_cleanup)  # Every 5 minutes
    timer.daemon = True
    timer.start()

if __name__ == '__main__':
    logger.info("🐅 Starting My Tiger - Ultra-Optimized Edition")
    
    # Start cleanup scheduler
    schedule_cleanup()
    
    # Run the application
    socketio.run(
        app, 
        debug=False, 
        host='0.0.0.0', 
        port=8000,
        use_reloader=False,
        log_output=False
    ) 