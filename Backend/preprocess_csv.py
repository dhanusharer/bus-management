import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_and_preprocess_csv(csv_file, sequence_length, scaler=None, save_scaler=False):
    """
    Load and preprocess CSV with optional scaler saving/loading
    
    Args:
        csv_file: Path to CSV file
        sequence_length: Number of timesteps per sequence
        scaler: Pre-fitted scaler (for inference). If None, creates new one
        save_scaler: If True, saves the fitted scaler to file
    
    Returns:
        sequences: Numpy array of shape (n_sequences, sequence_length, n_features)
        scaler: The fitted StandardScaler (useful for saving)
    """
    # Read CSV with no header
    df = pd.read_csv(csv_file, header=None)
    
    # Get feature columns: every second index after first column, plus total
    feature_indices = list(range(2, df.shape[1], 2))
    feature_indices.append(df.shape[1] - 1)  # Add total count
    
    # Extract count columns
    data = df.iloc[:, feature_indices].values
    
    # Scale features
    if scaler is None:
        # Training mode: fit new scaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        if save_scaler:
            # Save scaler for later use
            scaler_path = os.path.join(os.path.dirname(csv_file), '../trainer/scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to {scaler_path}")
    else:
        # Inference mode: use existing scaler
        scaled_data = scaler.transform(data)
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - sequence_length + 1):
        seq = scaled_data[i:i + sequence_length]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    return sequences, scaler


# Example usage
if __name__ == "__main__":
    csv_file = "bmtc_cctv_counts.csv"
    sequence_length = 10
    
    seq_data, scaler = load_and_preprocess_csv(csv_file, sequence_length, save_scaler=True)
    print("Shape of sequential data:", seq_data.shape)
