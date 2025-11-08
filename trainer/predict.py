import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
import sys
import os

# Add Backend to path
sys.path.append(os.path.abspath('../Backend'))
from preprocess_csv import load_and_preprocess_csv

# Configuration
CSV_FILE = '../backend_database/bmtc_cctv_counts.csv'
SEQUENCE_LENGTH = 10
CHECK_INTERVAL = 10

# Load models and scaler
print("Loading models and scaler...")
gru_model = load_model('gru_model.h5', compile=False)

with open('gbm_model.pkl', 'rb') as f:
    gbm_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Models and scaler loaded successfully!")
print(f"Scaler info - Features: {scaler.n_features_in_}")
print(f"Target (last feature) - Mean: {scaler.mean_[-1]:.2f}, Std: {scaler.scale_[-1]:.2f}")

def get_latest_sequence():
    """Get the latest sequence using saved scaler"""
    try:
        # Load sequences with the saved scaler
        sequences, _ = load_and_preprocess_csv(CSV_FILE, SEQUENCE_LENGTH, scaler=scaler)
        
        if len(sequences) == 0:
            print("Not enough data for a sequence")
            return None, None
        
        # Get the last sequence
        latest_seq = sequences[-1]
        
        # Prepare input (remove target column from all timesteps)
        X_input = latest_seq[:, :-1]
        X_input = X_input.reshape(1, SEQUENCE_LENGTH, -1)
        
        return X_input, latest_seq
    
    except Exception as e:
        print(f"Error getting sequence: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def make_prediction(X_input):
    """Run inference with GRU+GBM models"""
    gru_features = gru_model.predict(X_input, verbose=0)
    prediction = gbm_model.predict(gru_features)
    return prediction[0]

def inverse_scale_prediction(scaled_prediction):
    """Convert scaled prediction back to original scale using StandardScaler formula"""
    # StandardScaler: scaled = (original - mean) / std
    # So: original = (scaled * std) + mean
    
    target_mean = scaler.mean_[-1]
    target_std = scaler.scale_[-1]
    
    unscaled_prediction = (scaled_prediction * target_std) + target_mean
    
    return unscaled_prediction

def save_prediction(prediction, output_file='predictions.csv'):
    """Save prediction to CSV with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    pred_df = pd.DataFrame({
        'timestamp': [timestamp],
        'predicted_total_count': [prediction]
    })
    
    try:
        file_exists = os.path.isfile(output_file)
        pred_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"Error saving prediction: {e}")

def main():
    """Main loop to monitor CSV and make predictions"""
    print(f"\nStarting prediction service...")
    print(f"Monitoring: {CSV_FILE}")
    print(f"Checking every {CHECK_INTERVAL} seconds\n")
    
    last_row_count = 0
    
    while True:
        try:
            df = pd.read_csv(CSV_FILE, header=None)
            current_row_count = len(df)
            
            if current_row_count > last_row_count and current_row_count >= SEQUENCE_LENGTH:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New data detected!")
                
                X_input, latest_seq = get_latest_sequence()
                
                if X_input is not None:
                    # Make prediction (scaled)
                    scaled_prediction = make_prediction(X_input)
                    
                    # Inverse transform to actual count
                    prediction = inverse_scale_prediction(scaled_prediction)
                    
                    # Show recent totals
                    df_recent = pd.read_csv(CSV_FILE, header=None).tail(5)
                    recent_totals = df_recent.iloc[:, -1].values
                    
                    print(f"Recent counts: {recent_totals}")
                    print(f"Scaled prediction: {scaled_prediction:.2f}")
                    print(f"Unscaled prediction: {prediction:.2f}")
                    
                    save_prediction(prediction)
                    
                    last_row_count = current_row_count
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for new data...")
            
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nStopping prediction service...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
