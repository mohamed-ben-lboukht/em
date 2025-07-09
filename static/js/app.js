// My Tiger - Ultra-Optimized Emotion Detection
class MyTigerApp {
    constructor() {
        // Core properties with optimized data structures
        this.socket = io();
        this.keystrokeBuffer = new CircularBuffer(200); // Efficient circular buffer
        this.emotionHistory = new CircularBuffer(100); // Track emotion changes over time
        
        // Performance optimizations
        this.isProcessing = false;
        this.lastProcessTime = 0;
        this.processThrottle = 150; // Minimum ms between processing
        this.batchSize = 10; // Process in batches
        
        // Animation optimization
        this.animationFrame = null;
        this.pendingAnimations = new Map();
        
        // Emotion configuration
        this.emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'];
        this.emotionIcons = {
            'neutral': 'far fa-face-meh',
            'happy': 'far fa-face-smile',
            'sad': 'far fa-face-frown',
            'angry': 'far fa-face-angry',
            'fearful': 'far fa-face-dizzy',
            'disgusted': 'far fa-face-grimace',
            'surprised': 'far fa-face-surprise'
        };
        
        this.emotionDescriptions = {
            'neutral': 'A balanced emotional state',
            'happy': 'Positive and joyful feelings',
            'sad': 'Melancholy or downcast emotions',
            'angry': 'Frustration or irritation detected',
            'fearful': 'Anxiety or concern present',
            'disgusted': 'Discomfort or aversion felt',
            'surprised': 'Unexpected or startled reactions'
        };
        
        // Statistics tracking
        this.stats = {
            keystrokesCount: 0,
            avgTypingSpeed: 0,
            sessionStart: Date.now(),
            predictions: 0
        };
        
        // Initialize the application
        this.init();
    }

    async init() {
        this.showLoading();
        
        // Pre-cache DOM elements for performance
        this.cacheElements();
        
        // Initialize components
        await this.setupEventListeners();
        await this.initializeSocketEvents();
        this.initializePerformanceMonitor();
        
        // Hide loading screen after setup
        setTimeout(() => {
            this.hideLoading();
            this.startStatsUpdater();
        }, 1200);
    }

    cacheElements() {
        // Cache frequently accessed DOM elements
        this.elements = {
            typingArea: document.getElementById('typingArea'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            dominantEmotion: document.getElementById('dominantEmotion'),
            emotionDescription: document.getElementById('emotionDescription'),
            dominantIcon: document.getElementById('dominantIcon'),
            emotionCircle: document.getElementById('emotionCircle'),
            helpPanel: document.getElementById('helpPanel'),
            
            // Emotion bars (pre-cache for performance)
            emotionBars: new Map(),
            emotionValues: new Map(),
            emotionGlows: new Map()
        };
        
        // Cache emotion bar elements
        this.emotions.forEach(emotion => {
            this.elements.emotionBars.set(emotion, 
                document.querySelector(`.bar-fill-modern[data-emotion="${emotion}"]`));
            this.elements.emotionValues.set(emotion, 
                document.querySelector(`.bar-value[data-emotion="${emotion}"]`));
            this.elements.emotionGlows.set(emotion, 
                document.querySelector(`.bar-glow[data-emotion="${emotion}"]`));
        });
    }

    showLoading() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'flex';
        }
    }

    hideLoading() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.classList.add('hidden');
            setTimeout(() => {
                this.elements.loadingOverlay.style.display = 'none';
            }, 500);
        }
    }

    async setupEventListeners() {
        // Optimized keystroke tracking with passive listeners
        this.elements.typingArea.addEventListener('keydown', this.handleKeyDown.bind(this), { passive: true });
        this.elements.typingArea.addEventListener('keyup', this.handleKeyUp.bind(this), { passive: true });

        // Button events with debouncing
        this.attachButtonListener('clearBtn', this.debounce(this.clearText.bind(this), 200));
        this.attachButtonListener('resetBtn', this.debounce(this.resetAnalysis.bind(this), 300));
        this.attachButtonListener('helpBtn', this.toggleHelp.bind(this));
        this.attachButtonListener('closeHelp', this.closeHelp.bind(this));
        this.attachButtonListener('exportBtn', this.exportData.bind(this));

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this), { passive: false });
        
        // Add resize listener for responsive updates
        window.addEventListener('resize', this.debounce(this.handleResize.bind(this), 250), { passive: true });
    }

    attachButtonListener(id, handler) {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', handler);
        }
    }

    handleKeyDown(event) {
        if (this.shouldSkipKey(event.key)) return;

        const now = performance.now();
        this.stats.keystrokesCount++;
        
        this.keystrokeBuffer.push({
            type: 'press',
            key: event.key,
            timestamp: now
        });

        // Throttled processing for performance
        this.throttledProcess();
    }

    handleKeyUp(event) {
        if (this.shouldSkipKey(event.key)) return;

        const now = performance.now();
        
        this.keystrokeBuffer.push({
            type: 'release',
            key: event.key,
            timestamp: now
        });
    }

    shouldSkipKey(key) {
        // Optimized with Set for O(1) lookup
        const skipKeys = new Set(['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab', 'Escape']);
        return skipKeys.has(key) || key.startsWith('Arrow') || key.startsWith('F');
    }

    throttledProcess() {
        const now = performance.now();
        
        if (this.isProcessing || (now - this.lastProcessTime) < this.processThrottle) {
            return;
        }
        
        if (this.keystrokeBuffer.size() >= 8) { // Minimum events for analysis
            this.lastProcessTime = now;
            this.processKeystrokeFeatures();
        }
    }

    processKeystrokeFeatures() {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        
        // Process in next tick to avoid blocking UI
        setTimeout(() => {
            try {
                const features = this.calculateKeystrokeFeatures();
                
                if (this.hasValidFeatures(features)) {
                    this.socket.emit('keystroke_data', {
                        features: features,
                        timestamp: Date.now(),
                        stats: this.stats
                    });
                    this.stats.predictions++;
                }
            } catch (error) {
                console.error('Feature processing error:', error);
            } finally {
                this.isProcessing = false;
            }
        }, 0);
    }

    calculateKeystrokeFeatures() {
        const features = {
            pp: [], // Press-to-Press
            pr: [], // Press-to-Release  
            rp: [], // Release-to-Press
            rr: []  // Release-to-Release
        };
        
        const events = this.keystrokeBuffer.getLastN(50); // Get recent events
        
        for (let i = 1; i < events.length; i++) {
            const current = events[i];
            const previous = events[i - 1];
            
            const timeDiff = current.timestamp - previous.timestamp;
            
            // Optimized timing validation
            if (timeDiff >= 10 && timeDiff <= 2000) {
                const key = `${previous.type}-${current.type}`;
                
                switch (key) {
                    case 'press-press':
                        features.pp.push(timeDiff);
                        break;
                    case 'press-release':
                        if (current.key === previous.key) {
                            features.pr.push(timeDiff);
                        }
                        break;
                    case 'release-press':
                        features.rp.push(timeDiff);
                        break;
                    case 'release-release':
                        features.rr.push(timeDiff);
                        break;
                }
            }
        }
        
        return features;
    }

    hasValidFeatures(features) {
        return features.pp.length > 0 || features.pr.length > 0 || 
               features.rp.length > 0 || features.rr.length > 0;
    }

    clearText() {
        this.elements.typingArea.value = '';
        this.keystrokeBuffer.clear();
        this.stats.keystrokesCount = 0;
    }

    resetAnalysis() {
        this.keystrokeBuffer.clear();
        this.emotionHistory.clear();
        this.stats = {
            keystrokesCount: 0,
            avgTypingSpeed: 0,
            sessionStart: Date.now(),
            predictions: 0
        };
        
        // Efficient batch reset
        this.batchUpdateEmotions(this.emotions.reduce((acc, emotion) => {
            acc[emotion] = 0;
            return acc;
        }, {}));
        
        this.updateDominantEmotion('neutral', 0);
        this.socket.emit('reset_buffer');
    }

    toggleHelp() {
        this.elements.helpPanel?.classList.toggle('open');
    }

    closeHelp() {
        this.elements.helpPanel?.classList.remove('open');
    }

    exportData() {
        const exportData = {
            metadata: {
                app: "My Tiger - Optimized Emotion Detection",
                timestamp: new Date().toISOString(),
                session_duration: Date.now() - this.stats.sessionStart,
                total_keystrokes: this.stats.keystrokesCount,
                total_predictions: this.stats.predictions
            },
            emotions: this.emotionHistory.toArray(),
            stats: this.stats
        };
        
        this.downloadJSON(exportData, `my_tiger_session_${Date.now()}.json`);
    }

    downloadJSON(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            const shortcuts = {
                'l': this.clearText.bind(this),
                'r': this.resetAnalysis.bind(this),
                '/': this.toggleHelp.bind(this),
                'e': this.exportData.bind(this)
            };
            
            if (shortcuts[event.key]) {
                event.preventDefault();
                shortcuts[event.key]();
            }
        }
        
        if (event.key === 'Escape') {
            this.closeHelp();
        }
    }

    handleResize() {
        // Responsive adjustments if needed
        this.cacheElements(); // Re-cache elements if layout changed
    }

    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('✅ Connected to My Tiger server');
        });

        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
        });

        this.socket.on('emotion_prediction', (data) => {
            this.updateEmotionDisplay(data.emotions);
            this.emotionHistory.push({
                emotions: data.emotions,
                timestamp: Date.now(),
                mode: data.mode || 'hybrid'
            });
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
        });
    }

    updateEmotionDisplay(emotions) {
        if (!emotions || typeof emotions !== 'object') return;

        // Find dominant emotion efficiently
        const [dominantEmotion, maxValue] = Object.entries(emotions)
            .reduce((max, [emotion, value]) => value > max[1] ? [emotion, value] : max, ['neutral', 0]);
        
        // Batch update emotions with optimized animations
        this.batchUpdateEmotions(emotions);
        this.updateDominantEmotion(dominantEmotion, maxValue);
    }

    batchUpdateEmotions(emotions) {
        // Cancel previous animation frame
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        
        // Schedule batch update
        this.animationFrame = requestAnimationFrame(() => {
            this.emotions.forEach(emotion => {
                const value = emotions[emotion] || 0;
                const percentage = Math.round(value * 100);
                
                const barFill = this.elements.emotionBars.get(emotion);
                const barValue = this.elements.emotionValues.get(emotion);
                
                if (barFill && barValue) {
                    // Use transform for better performance
                    barFill.style.transform = `scaleX(${percentage / 100})`;
                    barFill.style.transformOrigin = 'left';
                    barValue.textContent = `${percentage}%`;
                    
                    // Add glow effect for significant emotions
                    if (percentage > 15) {
                        this.triggerGlowEffect(emotion);
                    }
                }
            });
        });
    }

    triggerGlowEffect(emotion) {
        const barGlow = this.elements.emotionGlows.get(emotion);
        if (barGlow && !barGlow.classList.contains('active')) {
            barGlow.classList.add('active');
            setTimeout(() => {
                barGlow.classList.remove('active');
            }, 800);
        }
    }

    updateDominantEmotion(emotion, confidence) {
        const elements = this.elements;
        
        // Batch DOM updates
        requestAnimationFrame(() => {
            if (elements.dominantEmotion) {
                elements.dominantEmotion.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
            }

            if (elements.emotionDescription) {
                elements.emotionDescription.textContent = this.emotionDescriptions[emotion] || 'Analyzing...';
            }

            if (elements.dominantIcon) {
                const iconClass = this.emotionIcons[emotion] || 'far fa-face-meh';
                elements.dominantIcon.innerHTML = `<i class="${iconClass}"></i>`;
            }

            if (elements.emotionCircle) {
                elements.emotionCircle.setAttribute('data-emotion', emotion);
            }
        });
    }

    initializePerformanceMonitor() {
        // Optional performance monitoring
        if (process.env.NODE_ENV === 'development') {
            setInterval(() => {
                console.log('📊 Performance Stats:', {
                    bufferSize: this.keystrokeBuffer.size(),
                    historySize: this.emotionHistory.size(),
                    keystrokesPerMin: Math.round(this.stats.keystrokesCount / ((Date.now() - this.stats.sessionStart) / 60000)),
                    predictions: this.stats.predictions
                });
            }, 30000);
        }
    }

    startStatsUpdater() {
        // Update statistics every 5 seconds
        setInterval(() => {
            const sessionTime = (Date.now() - this.stats.sessionStart) / 1000;
            this.stats.avgTypingSpeed = this.stats.keystrokesCount / sessionTime * 60; // KPM
        }, 5000);
    }

    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Efficient Circular Buffer Implementation
class CircularBuffer {
    constructor(size) {
        this.size = size;
        this.buffer = new Array(size);
        this.start = 0;
        this.count = 0;
    }

    push(item) {
        this.buffer[(this.start + this.count) % this.size] = item;
        if (this.count < this.size) {
            this.count++;
        } else {
            this.start = (this.start + 1) % this.size;
        }
    }

    getLastN(n) {
        const result = [];
        const actualN = Math.min(n, this.count);
        
        for (let i = 0; i < actualN; i++) {
            const index = (this.start + this.count - actualN + i) % this.size;
            result.push(this.buffer[index]);
        }
        
        return result;
    }

    toArray() {
        const result = [];
        for (let i = 0; i < this.count; i++) {
            result.push(this.buffer[(this.start + i) % this.size]);
        }
        return result;
    }

    clear() {
        this.start = 0;
        this.count = 0;
    }

    size() {
        return this.count;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.myTiger = new MyTigerApp();
});

// Add service worker for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => console.log('SW registered'))
            .catch(error => console.log('SW registration failed'));
    });
} 