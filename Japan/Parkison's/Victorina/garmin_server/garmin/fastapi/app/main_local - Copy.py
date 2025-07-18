import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
MODEL_TYPE = "lstm"  # or "gb"
USER_ID = 14
TIMESTAMP = "2021-03-21 04:45:00"
N_MINUTES_AHEAD = 30
SEQUENCE_LENGTH = 8
INTERVAL_MINUTES = 15

DATA_PATH = r"c:\Users\Shared-PC\Desktop\For Submission\For Submission\data\combined_data\combined_data_participant1_15min.xlsx"
MODEL_DIR = "./models"

# === Load Combined Data ===
columns = ['timestamp', 'heart_rate', 'steps', 'stress_score',
           'awake', 'deep', 'light', 'rem', 'nonrem_total', 'total',
           'nonrem_percentage', 'sleep_efficiency',
           'time_from_last_drug_taken', 'wearing_off']
data = pd.read_excel(DATA_PATH, usecols=columns, engine='openpyxl')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp').sort_index().fillna(0)


# === Prepare Input ===
def prepare_input(df, target_time, history_minutes):
    start_time = target_time - timedelta(minutes=history_minutes)
    window = df[(df.index >= start_time) & (df.index <= target_time)].copy()
    if window.empty:
        raise ValueError("❌ No data available for the selected window.")
    return window


# === Load Model + Scaler ===
def load_latest_model(model_type):
    files = sorted(os.listdir(MODEL_DIR), reverse=True)
    if model_type == "lstm":
        model_file = next(f for f in files if f.endswith(".keras"))
        scaler_file = model_file.replace(".keras", "_scaler.pkl")
        model = load_model(os.path.join(MODEL_DIR, model_file))
        scaler = joblib.load(os.path.join(MODEL_DIR, scaler_file))
        return model, scaler
    else:
        model_file = next(f for f in files if f.endswith(".pkl"))
        model = joblib.load(os.path.join(MODEL_DIR, model_file))
        return model, None


# === Main Prediction ===
try:
    adjusted_time = pd.to_datetime(TIMESTAMP) - pd.Timedelta(minutes=N_MINUTES_AHEAD)
    input_data = prepare_input(data, adjusted_time, SEQUENCE_LENGTH * INTERVAL_MINUTES)

    model, scaler = load_latest_model(MODEL_TYPE)

    if MODEL_TYPE == "lstm":
        feature_cols = [col for col in input_data.columns if col != "wearing_off"]
        X_raw = input_data[feature_cols].values
        X_scaled = scaler.transform(X_raw)

        if len(X_scaled) < SEQUENCE_LENGTH:
            raise ValueError("❌ Not enough data for LSTM sequence.")

        X_seq = X_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)
        prob = model.predict(X_seq)[0][0]
        pred = int(prob > 0.5)

        print(f"\n🔮 LSTM Prediction (in {N_MINUTES_AHEAD} min): {pred}")
        print(f"📊 Probability of wearing_off = {prob:.4f}")

    else:  # GB model
        expected_features = model.feature_names_in_
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0  # fallback if feature missing

        X_input = input_data[expected_features]
        pred = model.predict(X_input)[-1]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_input)[-1][1]
        else:
            prob = None

        print(f"\n🔮 GB Prediction (in {N_MINUTES_AHEAD} min): {pred}")
        if prob is not None:
            print(f"📊 Probability of wearing_off = {prob:.4f}")

except Exception as e:
    print("❌ Error during prediction:", str(e))
