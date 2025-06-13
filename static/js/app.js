// My Tiger - Simple Emotion Detection
class MyTigerApp {
    constructor() {
        // Core properties
        this.socket = io();
        this.keystrokeBuffer = [];
        this.lastKeystrokeTime = null;
        
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
        
        // Initialize the application
        this.init();
    }

    async init() {
        // Show loading screen
        this.showLoading();
        
        // Initialize components
        await this.setupEventListeners();
        await this.initializeSocketEvents();
        
        // Hide loading screen after setup
        setTimeout(() => {
            this.hideLoading();
        }, 1500);
    }

    showLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('hidden');
            setTimeout(() => {
                loadingOverlay.style.display = 'none';
            }, 500);
        }
    }

    async setupEventListeners() {
        const typingArea = document.getElementById('typingArea');
        
        // Keystroke tracking
        typingArea.addEventListener('keydown', (e) => this.handleKeyDown(e));
        typingArea.addEventListener('keyup', (e) => this.handleKeyUp(e));

        // Button events
        document.getElementById('clearBtn')?.addEventListener('click', () => this.clearText());
        document.getElementById('resetBtn')?.addEventListener('click', () => this.resetAnalysis());
        document.getElementById('helpBtn')?.addEventListener('click', () => this.toggleHelp());
        document.getElementById('closeHelp')?.addEventListener('click', () => this.closeHelp());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }

    handleKeyDown(event) {
        if (this.shouldSkipKey(event.key)) return;

        const now = performance.now();
        
        // Calculate timing features
        const timingSinceLastKey = this.lastKeystrokeTime ? now - this.lastKeystrokeTime : 0;
        
        this.keystrokeBuffer.push({
            type: 'keydown',
            timestamp: now,
            timingSinceLastKey: timingSinceLastKey
        });

        this.lastKeystrokeTime = now;
        
        // Process data for real-time feel
        if (this.keystrokeBuffer.length >= 3) {
            this.processKeystrokeData();
        }
    }

    handleKeyUp(event) {
        if (this.shouldSkipKey(event.key)) return;

        const now = performance.now();
        const timingSinceLastKey = this.lastKeystrokeTime ? now - this.lastKeystrokeTime : 0;
        
        this.keystrokeBuffer.push({
            type: 'keyup',
            timestamp: now,
            timingSinceLastKey: timingSinceLastKey
        });

        this.lastKeystrokeTime = now;
        
        // Keep buffer manageable
        if (this.keystrokeBuffer.length > 50) {
            this.keystrokeBuffer = this.keystrokeBuffer.slice(-30);
        }
    }

    shouldSkipKey(key) {
        const skipKeys = ['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab', 'Escape'];
        return skipKeys.includes(key) || key.startsWith('Arrow') || key.startsWith('F');
    }

    processKeystrokeData() {
        // Calculate timing features
        const timings = this.calculateTimingFeatures();
        
        if (timings.length > 0) {
            // Send to server
            this.socket.emit('keystroke_data', {
                timings: timings,
                timestamp: Date.now()
            });
        }
    }

    calculateTimingFeatures() {
        const timings = [];
        
        for (let i = 1; i < this.keystrokeBuffer.length; i++) {
            const current = this.keystrokeBuffer[i];
            const previous = this.keystrokeBuffer[i - 1];
            
            if (current.type === 'keydown' && previous.type === 'keydown') {
                const dwellTime = current.timestamp - previous.timestamp;
                if (dwellTime > 0 && dwellTime < 2000) { // Filter reasonable timings
                    timings.push(dwellTime);
                }
            }
        }
        
        return timings;
    }

    clearText() {
        document.getElementById('typingArea').value = '';
        this.keystrokeBuffer = [];
        this.lastKeystrokeTime = null;
    }

    resetAnalysis() {
        this.keystrokeBuffer = [];
        
        // Reset emotion display with animation
        this.emotions.forEach(emotion => {
            const barFill = document.querySelector(`.bar-fill-modern[data-emotion="${emotion}"]`);
            const barValue = document.querySelector(`.bar-value[data-emotion="${emotion}"]`);
            
            if (barFill && barValue) {
                barFill.style.width = '0%';
                barValue.textContent = '0%';
            }
        });
        
        // Reset dominant emotion
        this.updateDominantEmotion('neutral', 0);
    }

    toggleHelp() {
        const helpPanel = document.getElementById('helpPanel');
        if (helpPanel) {
            helpPanel.classList.toggle('open');
        }
    }

    closeHelp() {
        const helpPanel = document.getElementById('helpPanel');
        if (helpPanel) {
            helpPanel.classList.remove('open');
        }
    }

    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 'l':
                    event.preventDefault();
                    this.clearText();
                    break;
                case 'r':
                    event.preventDefault();
                    this.resetAnalysis();
                    break;
                case '/':
                    event.preventDefault();
                    this.toggleHelp();
                    break;
            }
        }
        
        if (event.key === 'Escape') {
            this.closeHelp();
        }
    }

    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to My Tiger server');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });

        this.socket.on('emotion_prediction', (data) => {
            this.updateEmotionDisplay(data.emotions);
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
        });
    }

    updateEmotionDisplay(emotions) {
        if (!emotions || typeof emotions !== 'object') return;

        // Find dominant emotion
        let dominantEmotion = 'neutral';
        let maxValue = 0;
        
        Object.entries(emotions).forEach(([emotion, value]) => {
            if (value > maxValue) {
                maxValue = value;
                dominantEmotion = emotion;
            }
        });
        
        // Update emotion bars with smooth animations
        this.emotions.forEach(emotion => {
            const value = emotions[emotion] || 0;
            const percentage = Math.round(value * 100);
            
            const barFill = document.querySelector(`.bar-fill-modern[data-emotion="${emotion}"]`);
            const barValue = document.querySelector(`.bar-value[data-emotion="${emotion}"]`);
            const barGlow = document.querySelector(`.bar-glow[data-emotion="${emotion}"]`);
            
            if (barFill && barValue) {
                // Animate bar fill
                setTimeout(() => {
                    barFill.style.width = `${percentage}%`;
                    barValue.textContent = `${percentage}%`;
                    
                    // Trigger glow effect for active emotions
                    if (percentage > 10 && barGlow) {
                        barGlow.style.left = '100%';
                        setTimeout(() => {
                            barGlow.style.left = '-100%';
                        }, 800);
                    }
                }, Math.random() * 100); // Stagger animations
            }
        });

        // Update dominant emotion
        this.updateDominantEmotion(dominantEmotion, maxValue);
    }

    updateDominantEmotion(emotion, confidence) {
        const emotionName = document.getElementById('dominantEmotion');
        const emotionDescription = document.getElementById('emotionDescription');
        const emotionIcon = document.getElementById('dominantIcon');
        const emotionCircle = document.getElementById('emotionCircle');

        if (emotionName) {
            emotionName.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        }

        if (emotionDescription) {
            emotionDescription.textContent = this.emotionDescriptions[emotion] || 'Analyzing your emotional state...';
        }

        if (emotionIcon) {
            const iconClass = this.emotionIcons[emotion] || 'far fa-face-meh';
            emotionIcon.innerHTML = `<i class="${iconClass}"></i>`;
        }

        // Update emotion circle styling
        if (emotionCircle) {
            emotionCircle.setAttribute('data-emotion', emotion);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.myTiger = new MyTigerApp();
}); 