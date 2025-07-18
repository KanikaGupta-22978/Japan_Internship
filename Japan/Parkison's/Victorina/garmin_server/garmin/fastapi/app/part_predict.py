# Core libraries
import pandas as pd
import numpy as np
import os
import csv
import random
import joblib
import os
from datetime import datetime

# FastAPI libraries
from fastapi import FastAPI, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Database libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from dotenv import load_dotenv

# Machine learning / Deep learning libraries
import tensorflow as tf
from keras import backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
from train_logistic_model import train_model

# @app.get("/users/{user_id}/timestamp/{timestamp}/{hours_ago}/predict") #, response_model=PredictResponse)
# def predict_user(
#     user_id: int = Path(..., title="Garmin Server ID", example=14),
#     timestamp: str = Path(..., title="End timestamp of data to forecast", example="2025-01-16 15:30:00"),
#     hours_ago: int = Path(..., title="9時間以上でないと動かない(9時間未満の場合、9時間に修正)", example=10)
#   ):
user_id=14
timestamp="2021-03-21 04:45:00"
hours_ago=10

user = 'participant1'
frequency = '15min' # 15min | 15s
dataset_type = '' # ''

if frequency == '15min':
    record_size_per_day = 96
elif frequency == '15s':
    record_size_per_day = 5760

# Columns to include    
if dataset_type == '':
    columns = [ 'timestamp', 'heart_rate', 'steps', 'stress_score',
            'awake', 'deep', 'light', 'rem', 'nonrem_total', 'total', 'nonrem_percentage', 'sleep_efficiency',
            'time_from_last_drug_taken', 'wearing_off' ]

metrics = {
    'balanced_accuracy': 'Bal. Acc.',
    'f1_score': 'F1 Score',
    'accuracy': 'Acc.',
    'precision': 'Precision',
    'sensitivity': 'Recall / Sn',
    'specificity': 'Sp',
    'auc': 'AUC'
}
  
# Load your trained model (adjust path as needed)
model_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Parkison's\Victorina\garmin_server\garmin\fastapi\app\logreg_pipeline.pkl"
model = joblib.load(model_path)

combined_data = pd.read_excel(r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Parkison's\Victorina\Old_dataset\Garmin Paper 2021 Supplementary Materials\Garmin論文2021 補足資料\For Submission\data\combined_data\combined_data_participant1_15min.xlsx",
                              index_col="timestamp",
                              usecols=columns,
                              engine='openpyxl')


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

    # Drop 'wearing_off_id' column if present (label column, not feature)
    if 'wearing_off_id' in data_window.columns:
        data_window.drop(columns=['wearing_off_id'], inplace=True)
        print("🚫 Removed 'wearing_off_id' column from input.")

    return data_window


# Prepare input data
input_data = prepare_input_for_prediction(user_id=user_id, timestamp=timestamp, history_minutes=hours_ago * 60)

# Get expected feature columns (features used in training)
expected_features = model.feature_names_in_

# Add missing expected columns with zeros (but DO NOT add 'wearing_off_id' if missing)
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Select and reorder columns exactly as expected by the model
input_data = input_data[expected_features]

print("🧾 Input data used for prediction:\n", input_data)

# Predict
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]



# Step 1: Prepare input
try:
    # Step 1: Prepare input data (dropping 'wearing_off_id' inside function if present)
    input_data = prepare_input_for_prediction(user_id=user_id, timestamp=timestamp, history_minutes=hours_ago * 60)

    # Step 2: Ensure all expected features are present (fill missing with 0)
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    # Step 3: Reorder columns exactly as model expects
    input_data = input_data[expected_features]

    print("🧾 Input data used for prediction:\n", input_data)

    # Step 4: Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Step 5: Print results
    print("\n🔮 Predicted label (wearing_off):", prediction)
    print("📊 Prediction probabilities:", prediction_proba)

    # Optional: Detailed class probability print
    class_labels = model.classes_
    print("\n🧠 Class probability breakdown:")
    for label, prob in zip(class_labels, prediction_proba):
        print(f"  Class {label}: {prob:.4f}")

except Exception as e:
    print("❌ Error during prediction:", str(e))
