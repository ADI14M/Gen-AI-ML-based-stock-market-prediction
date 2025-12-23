import os
import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name} imported successfully.")
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        return False

def verify_model_loading():
    print("Verifying model loading...")
    try:
        import tensorflow as tf
        model_path = 'keras_model_3.h5'
        if not os.path.exists(model_path):
            print(f"[FAIL] Model file {model_path} not found.")
            return False
        
        # Suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        try:
            model = tf.keras.models.load_model(model_path)
            print("[OK] Model loaded successfully.")
            return True
        except Exception as e:
            # Sometimes models from older keras versions need special handling or custom objects
            print(f"[FAIL] Error loading model: {e}")
            return False
    except ImportError:
        print("[FAIL] TensorFlow not installed, cannot verify model.")
        return False

def verify_finnhub_api():
    print("Verifying Finnhub API...")
    try:
        import requests
        # Key from analyst_ratings.py
        api_key = "cutllthr01qv6ijj0i9gcutllthr01qv6ijj0ia0"
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol=AAPL&token={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                print("[OK] Finnhub API key is valid and returning data.")
                return True
            else:
                print("[WARN] Finnhub API returned 200 but no data (might be expected for some stocks, but AAPL should have data).")
                return True
        else:
            print(f"[FAIL] Finnhub API check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Error checking Finnhub API: {e}")
        return False

def main():
    print("Starting verification...")
    
    # Check critical dependencies
    dependencies = [
        "streamlit", "pandas", "numpy", "matplotlib", "yfinance",
        "tensorflow", "transformers", "torch", "requests", "feedparser", "plotly"
    ]
    
    all_imports_pass = True
    for dep in dependencies:
        if not check_import(dep):
            all_imports_pass = False
            
    if not all_imports_pass:
        print("\n[CRITICAL] Some dependencies are missing. Please run: pip install -r requirements.txt")
    
    # Check Model
    model_pass = verify_model_loading()
    
    # Check API
    api_pass = verify_finnhub_api()
    
    if all_imports_pass and model_pass and api_pass:
        print("\n[SUCCESS] Setup verification passed! You can try running the app.")
    else:
        print("\n[FAILURE] Verification failed. Please fix the issues above.")

if __name__ == "__main__":
    main()
