// Emotion Detection App
class EmotionDetector {
    constructor() {
        this.socket = io();
        this.keystrokeBuffer = [];
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
        
        this.setupEventListeners();
        this.initializeSocketEvents();
    }

    setupEventListeners() {
        const typingArea = document.getElementById('typingArea');
        
        // Keystroke event listeners
        typingArea.addEventListener('keydown', (e) => this.handleKeyDown(e));
        typingArea.addEventListener('keyup', (e) => this.handleKeyUp(e));
        typingArea.addEventListener('input', () => this.updateStats());

        // Button event listeners
        document.getElementById('clearTextBtn').addEventListener('click', () => this.clearText());
        document.getElementById('resetBufferBtn').addEventListener('click', () => this.resetBuffer());
        
        // Info panel toggle
        document.getElementById('infoToggle').addEventListener('click', () => this.toggleInfoPanel());
    }

    handleKeyDown(event) {
        // Skip special keys (delete, backspace, etc.)
        if (this.shouldSkipKey(event.key)) {
            return;
        }

        const timestamp = performance.now();
        this.keystrokeBuffer.push({
            type: 'keydown',
            key: event.key,
            timestamp: timestamp,
            keyCode: event.keyCode
        });

        this.updateBufferDisplay();
        this.processKeystrokeData();
    }

    handleKeyUp(event) {
        // Skip special keys
        if (this.shouldSkipKey(event.key)) {
            return;
        }

        const timestamp = performance.now();
        this.keystrokeBuffer.push({
            type: 'keyup',
            key: event.key,
            timestamp: timestamp,
            keyCode: event.keyCode
        });

        this.updateBufferDisplay();
        this.processKeystrokeData();
    }

    shouldSkipKey(key) {
        // Skip deletion keys and special keys
        const skipKeys = ['Backspace', 'Delete', 'Tab', 'Escape', 'Enter', 'Shift', 'Control', 'Alt', 'Meta'];
        return skipKeys.includes(key) || key.startsWith('Arrow') || key.startsWith('F');
    }

    processKeystrokeData() {
        // Send keystroke data when we have enough samples
        if (this.keystrokeBuffer.length >= 10) {
            this.socket.emit('keystroke_data', {
                keystrokes: [...this.keystrokeBuffer],
                timestamp: Date.now()
            });
            
            // Keep only recent keystrokes to avoid memory issues
            if (this.keystrokeBuffer.length > 100) {
                this.keystrokeBuffer = this.keystrokeBuffer.slice(-50);
            }
        }
    }

    updateStats() {
        const typingArea = document.getElementById('typingArea');
        const text = typingArea.value;
        
        document.getElementById('wordCount').textContent = text.trim() ? text.trim().split(/\s+/).length : 0;
        document.getElementById('charCount').textContent = text.length;
    }

    updateBufferDisplay() {
        document.getElementById('bufferSize').textContent = this.keystrokeBuffer.length;
    }

    clearText() {
        document.getElementById('typingArea').value = '';
        this.updateStats();
    }

    resetBuffer() {
        this.keystrokeBuffer = [];
        this.updateBufferDisplay();
        
        // Reset emotion display
        this.emotions.forEach(emotion => {
            document.querySelector(`.bar-fill[data-emotion="${emotion}"]`).style.width = '0%';
            document.querySelector(`.emotion-value[data-emotion="${emotion}"]`).textContent = '0%';
        });
        
        // Reset dominant emotion
        document.getElementById('dominantEmotionName').textContent = 'Neutral';
        document.getElementById('dominantEmotionScore').textContent = '0%';
        document.getElementById('dominantEmotionIcon').innerHTML = '<i class="far fa-face-meh"></i>';
    }

    toggleInfoPanel() {
        const panel = document.getElementById('infoPanel');
        panel.classList.toggle('open');
    }

    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
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
        if (!emotions || typeof emotions !== 'object') {
            return;
        }

        // Update emotion bars
        this.emotions.forEach(emotion => {
            const value = emotions[emotion] || 0;
            const percentage = Math.round(value * 100);
            
            const barFill = document.querySelector(`.bar-fill[data-emotion="${emotion}"]`);
            const emotionValue = document.querySelector(`.emotion-value[data-emotion="${emotion}"]`);
            
            if (barFill && emotionValue) {
                barFill.style.width = `${percentage}%`;
                emotionValue.textContent = `${percentage}%`;
            }
        });

        // Find dominant emotion
        let dominantEmotion = 'neutral';
        let maxValue = 0;
        
        Object.entries(emotions).forEach(([emotion, value]) => {
            if (value > maxValue) {
                maxValue = value;
                dominantEmotion = emotion;
            }
        });

        // Update dominant emotion display
        const percentage = Math.round(maxValue * 100);
        document.getElementById('dominantEmotionName').textContent = 
            dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1);
        document.getElementById('dominantEmotionScore').textContent = `${percentage}%`;
        
        const iconClass = this.emotionIcons[dominantEmotion] || 'far fa-face-meh';
        document.getElementById('dominantEmotionIcon').innerHTML = `<i class="${iconClass}"></i>`;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EmotionDetector();
}); 