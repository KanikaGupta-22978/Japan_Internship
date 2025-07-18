# --- Modified part_predict.py ---
# --- Only the forecasted part is extracted ---

import pandas as pd
import os
import joblib
from datetime import datetime
from train_local import train_model

user_id = 14
timestamp = "2021-03-21 04:45:00"
hours_ago = 10
n_minutes_ahead = 30

# Check if model exists
if not os.path.exists('./models'):
  os.makedirs('./models')
# Check if any file is in model directory
if len(os.listdir('./models')) == 0:
  # Train the model
  train_model()
# Load latest keras model from models directory
model_files = os.listdir('./models')
model_files.sort(reverse=True)
latest_model_file = model_files[0]
latest_model_file = os.path.join('./models', latest_model_file)
# Load the model
model = joblib.load(latest_model_file)

columns = ['timestamp', 'heart_rate', 'steps', 'stress_score',
           'awake', 'deep', 'light', 'rem', 'nonrem_total', 'total', 'nonrem_percentage', 'sleep_efficiency',
           'time_from_last_drug_taken', 'wearing_off']

combined_data = pd.read_excel("combined_data_participant1_15min.xlsx", index_col="timestamp", usecols=columns, engine='openpyxl')

def prepare_input_for_prediction(user_id: int, timestamp: str, history_minutes: int = 60):
    garmin_data = combined_data
    if garmin_data.empty:
        raise ValueError("No data available for user.")

    garmin_data.index = pd.to_datetime(garmin_data.index)
    garmin_data = garmin_data.sort_index().fillna(0)

    window_end = pd.to_datetime(timestamp)
    window_start = window_end - pd.Timedelta(minutes=history_minutes)
    data_window = garmin_data[(garmin_data.index >= window_start) & (garmin_data.index <= window_end)].copy()

    if data_window.empty:
        raise ValueError("No data in selected window.")

    if 'wearing_off_id' in data_window.columns:
        data_window.drop(columns=['wearing_off_id'], inplace=True)

    return data_window

try:
    # For predicting n minutes ahead, shift the input window n minutes back
    prediction_time = pd.to_datetime(timestamp)
    adjusted_timestamp = prediction_time - pd.Timedelta(minutes=n_minutes_ahead)

    input_data = prepare_input_for_prediction(user_id=user_id, timestamp=adjusted_timestamp, history_minutes=hours_ago * 60)

    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_features]

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    print("\n🔮 Predicted label (wearing_off in", n_minutes_ahead, "min):", prediction)
    print("📊 Prediction probabilities:", prediction_proba)

    class_labels = model.classes_
    for label, prob in zip(class_labels, prediction_proba):
        print(f"  Class {label}: {prob:.4f}")

except Exception as e:
    print("❌ Error during prediction:", str(e))
