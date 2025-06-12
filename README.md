# 🐅 My Tiger - Real-time Emotion Detection from Keystroke Dynamics

**My Tiger** is a web application that detects emotions in real-time by analyzing your typing patterns. Using machine learning, it provides continuous emotional state monitoring based on keystroke dynamics without analyzing your content.

![My Tiger Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.3+-red) ![Machine Learning](https://img.shields.io/badge/ML-Keystroke_Dynamics-orange)

## 🌟 Features

### Real-time Emotion Detection
- **Continuous Analysis**: Emotions detected as you type
- **Seven Emotions**: Detects neutral, happy, sad, angry, fearful, disgusted, surprised
- **Privacy-First**: Only timing patterns analyzed, never your content
- **Instant Feedback**: Real-time emotion display and analysis

### Professional Interface
- **Modern Design**: Clean, intuitive web interface
- **Responsive Layout**: Works on desktop and mobile devices
- **Visual Feedback**: Interactive emotion bars and dominant emotion display
- **Data Export**: Download your emotion analysis data

### Keystroke Dynamics Analysis
- **Timing Patterns**: Analyzes press-release timings between keystrokes
- **Machine Learning**: Advanced SVR model for emotion prediction
- **Histogram Features**: Uses custom feature extraction for accuracy
- **Buffer Management**: Intelligent data collection and processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Your trained model file: `svr_model_hist.joblib`

### Installation

1. **Clone or download** the project files to your directory
2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python3 run.py
   ```
4. **Open your browser** and go to: http://localhost:8000

### Alternative: Using Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

### Using the Start Script
```bash
chmod +x start.sh
./start.sh
```

## 📊 How It Works

1. **Type Naturally**: Start typing in the text area
2. **Keystroke Capture**: The app captures timing between keystrokes
3. **Feature Extraction**: Histogram features computed from timing patterns
4. **Emotion Prediction**: Machine learning model predicts emotional state
5. **Real-time Display**: Results shown immediately with visual feedback

## 🔧 Technical Details

### Model Requirements
- **Input**: Histogram features from keystroke timings
- **Output**: 7-emotion probability distribution
- **Format**: Scikit-learn SVR model saved as `svr_model_hist.joblib`

### Keystroke Data Collection
- Press-press intervals
- Press-release (dwell) times
- Release-press intervals
- Automatic deletion filtering

### Feature Computation
```python
def compute_histogram_features(keystrokes, bins_edges=[0, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 300]):
    # Normalizes keystroke timings into histogram bins
    # Returns probability distribution across timing ranges
```

## 🤝 Contributing to Research

Help improve emotion detection research! 

**Contribute your data**: https://bit.ly/keystroke-emotion

- **Anonymous**: Only timing patterns shared, never content
- **Privacy Protected**: No personal information collected  
- **Research Purpose**: Advance keystroke dynamics research
- **Easy Export**: Use the "Export Data" button in the app

## 🛡️ Privacy & Security

- **100% Local Processing**: Your text never leaves your device
- **Timing Only**: Only keystroke timing patterns are analyzed
- **No Content Storage**: Your actual text is never stored or transmitted
- **Anonymous Data**: Optional research contributions are fully anonymous

## 📁 Project Structure

```
my_tiger/
├── app.py                 # Main Flask application
├── run.py                 # Startup script with checks
├── start.sh              # Bash startup script
├── requirements.txt       # Python dependencies
├── svr_model_hist.joblib # Your trained model (required)
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Application styling
│   └── js/
│       └── app.js        # Frontend functionality
└── README.md             # This file
```

## 🎯 Usage Tips

1. **Type naturally** - Don't change your typing style
2. **Minimum data** - Need at least 5 keystrokes for analysis
3. **Avoid deletions** - Backspace/Delete are filtered out automatically
4. **Real-time** - Emotions update as you continue typing
5. **Export data** - Use the export feature to save your session

## 🔧 Troubleshooting

### Common Issues

**Port 5000 in use (macOS)**:
- App uses port 8000 by default to avoid macOS Control Center conflicts

**Model not found**:
- Ensure `svr_model_hist.joblib` is in the root directory
- Check the model was trained with the same feature format

**Dependencies missing**:
- Run: `pip3 install -r requirements.txt`
- Consider using a virtual environment

**WebSocket connection issues**:
- Check firewall settings
- Ensure port 8000 is available
- Try refreshing the browser

## 🚀 Advanced Usage

### Custom Model Integration
Replace `svr_model_hist.joblib` with your own model:
- Must accept histogram features (10 bins)
- Must output 7-emotion probabilities
- Use scikit-learn compatible format

### Data Analysis
Export format includes:
```json
{
  "metadata": {
    "app": "My Tiger - Emotion Detection",
    "timestamp": "2024-01-01T00:00:00.000Z",
    "contribute_url": "https://bit.ly/keystroke-emotion"
  },
  "predictions": [...]
}
```

## 📄 License

This project is for research and educational purposes. Please respect privacy and use responsibly.

## 🤖 About

My Tiger demonstrates the power of keystroke dynamics for emotion detection. Built with Flask, Socket.IO, and modern web technologies for real-time analysis.

---

**Ready to explore your emotions through typing?** Start the app and begin typing naturally! 