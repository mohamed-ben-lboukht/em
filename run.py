#!/usr/bin/env python3
"""
My Tiger - Emotion Detection Startup Script
==========================================

This script starts the My Tiger emotion detection application.
Make sure you have installed the required dependencies first:

    pip3 install -r requirements.txt

Then run this script:

    python3 run.py

The application will be available at http://localhost:8000
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'flask_socketio', 
        'joblib',
        'numpy',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_model_file():
    """Check if the model file exists"""
    return os.path.exists('svr_model_hist.joblib')

def main():
    print("🐅 My Tiger - Emotion Detection Application")
    print("=" * 50)
    
    # Check if model file exists
    if not check_model_file():
        print("❌ Error: Model file 'svr_model_hist.joblib' not found!")
        print("   Please ensure your trained model is in the current directory.")
        sys.exit(1)
    else:
        print("✅ Model file found: svr_model_hist.joblib")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("   Please install them with: pip3 install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependencies are installed")
    
    print("\n🚀 Starting My Tiger application...")
    print("   Access the application at: http://localhost:8000")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the app
        from app import socketio, app
        socketio.run(app, debug=True, host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        print("\n👋 My Tiger application stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("   Check the error details above and ensure all requirements are met.")
        sys.exit(1)

if __name__ == "__main__":
    main() 