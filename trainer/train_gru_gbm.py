import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# ---- Import preprocessing function ----
import sys
import os
sys.path.append(os.path.abspath('../Backend'))
from preprocess_csv import load_and_preprocess_csv

print("Training script started")

# ---- Parameters ----
CSV_FILE = "../backend_database/bmtc_cctv_counts.csv"
SEQUENCE_LENGTH = 10
TARGET_COL = -1  # Last column is the target

# ---- Data Preparation ----
print("Loading and preprocessing CSV data...")
sequences, scaler = load_and_preprocess_csv(CSV_FILE, SEQUENCE_LENGTH, save_scaler=True)
print("Loaded sequence data with shape:", sequences.shape)

# ---- Target/Inputs ----
y = sequences[:, -1, TARGET_COL]  # Last time-step, last feature
X = sequences[:, :, :-1]          # Remove target column from all time steps
print("Prepared X shape:", X.shape)
print("Prepared y shape:", y.shape)

# ---- Split Data ----
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# ---- Build & Train GRU ----
print("Building GRU model...")
gru_model = Sequential([
    GRU(64, input_shape=(SEQUENCE_LENGTH, X.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training GRU model...")
gru_model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)
print("GRU training complete")

# ---- Feature Extraction ----
print("Extracting GRU features...")
gru_train_feats = gru_model.predict(X_train)
gru_test_feats = gru_model.predict(X_test)
print("GRU features extracted")

# ---- Train GBM on GRU outputs ----
print("Training GBM regressor on GRU features...")
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gbm.fit(gru_train_feats, y_train)
print("GBM training complete")

# ---- Evaluate hybrid model ----
print("Predicting with GBM and evaluating hybrid model...")
gbm_preds = gbm.predict(gru_test_feats)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n=== Model Evaluation ===")
print("GRU+GBM MAE:", mean_absolute_error(y_test, gbm_preds))
print("GRU+GBM RMSE:", np.sqrt(mean_squared_error(y_test, gbm_preds)))
print("GRU+GBM R2:", r2_score(y_test, gbm_preds))

# ---- Save Models ----
print("\n=== Saving Models ===")
gru_model.save('gru_model.h5')
print("GRU model saved to gru_model.h5")

with open('gbm_model.pkl', 'wb') as f:
    pickle.dump(gbm, f)
print("GBM model saved to gbm_model.pkl")

print("Scaler was already saved during preprocessing")
print("\nTraining complete! All models and scaler saved successfully.")
