# Core libraries
import pandas as pd
import numpy as np
import os
import csv
import random

from datetime import datetime

# FastAPI libraries
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Database libraries
from app.preprocess import get_and_merge_data

# Machine learning / Deep learning libraries
import joblib
from app.train import train_model

app = FastAPI()
origins = [
  "https://garmin-server.tomlab.jp",
  "http://localhost:3001"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

TARGET_FREQ = '15min'

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

# Check if model exists
if not os.path.exists('./app/models'):
  os.makedirs('./app/models')
# Check if any file is in model directory
if len(os.listdir('./app/models')) == 0:
  # Train the model
  train_model()
# Load latest keras model from models directory
model_files = os.listdir('./app/models')
model_files.sort(reverse=True)
latest_model_file = model_files[0]
latest_model_file = os.path.join('./app/models', latest_model_file)
# Load the model
model = joblib.load(latest_model_file)

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

@app.get("/users/{user_id}/{timestamp}/{hours_ago}/{n_minutes_ahead}/predict")
def predict_user(
    user_id: int = Path(..., title="Garmin Server's User ID", example=14),
    timestamp: str = Path(..., title="End timestamp of data to forecast", example="2025-01-16 15:30:00"),
    hours_ago: int = Path(..., title="Period used for forecasting", example=10),
    n_minutes_ahead: int = Path(..., title="Indicates how many minutes ahead to predict", example=60)
  ):

  # For predicting n minutes ahead, shift the input window n minutes back
  adjusted_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(minutes=n_minutes_ahead)
  target_timestamp = pd.to_datetime(timestamp) + pd.Timedelta(minutes=n_minutes_ahead)

  try:
    input_data = get_and_merge_data(user_id, adjusted_timestamp, hours_ago)
    if len(input_data) <= 0:
      return {
        "user_id": user_id,
        "timestamp": timestamp,
        "hours_ago": hours_ago,
        "n_minutes_ahead": n_minutes_ahead,
        "target_timestamp": target_timestamp,
        "forecasts": -1.0,  # -1.0 indicates no prediction
        "error": "404 No data found"
      }

  except Exception as e:
    return {
      "user_id": user_id,
      "timestamp": timestamp,
      "hours_ago": hours_ago,
      "n_minutes_ahead": n_minutes_ahead,
      "target_timestamp": target_timestamp,
      "forecasts": -1.0,  # -1.0 indicates no prediction
      "error": "500 Internal Server Error: " + str(e)
    }

  
  try:
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_features]

    prediction = model.predict(input_data)[0]
    # prediction_proba = model.predict_proba(input_data)[0]

    print("\n🔮 Predicted label (wearing_off in", n_minutes_ahead, "min):", prediction)
    # print("📊 Prediction probabilities:", prediction_proba)

    # class_labels = model.classes_
    # for label, prob in zip(class_labels, prediction_proba):
    #     print(f"  Class {label}: {prob:.4f}")

  except Exception as e:
    return {
      "user_id": user_id,
      "timestamp": timestamp,
      "hours_ago": hours_ago,
      "n_minutes_ahead": n_minutes_ahead,
      "target_timestamp": target_timestamp,
      "forecasts": -1.0,  # -1.0 indicates no prediction
      "error": "Error during prediction: " + str(e)
    }

  return_dict = {
    "user_id": user_id,
    "timestamp": timestamp,
    "hours_ago": hours_ago,
    "n_minutes_ahead": n_minutes_ahead,
    "target_timestamp": target_timestamp,
    "forecasts": prediction,
    "error": "0"  # No error
  }

  file_path = 'app/record.csv'
  if (os.path.isfile(file_path)):
    with open(file_path, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([user_id, timestamp, hours_ago, n_minutes_ahead, target_timestamp, prediction])
  else:
    with open(file_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow([user_id, timestamp, hours_ago, n_minutes_ahead, target_timestamp, prediction])

  # Return data
  return return_dict


@app.get("/forecasting-logs")
def get_forecasting_logs():
  file_path = 'app/record.csv'
  with open(file_path) as f:
    reader = csv.reader(f)
    result = []
    for row in reader:
        result.append(row)
  columns = ['user_id', 'timestamp', 'hours_ago', 'n_minutes_ahead', 'target_timestamp', 'prediction']
    
  return pd.DataFrame(data = result, columns = columns).to_dict(orient='records')


@app.get("/users/{user_id}/{timestamp}/{hours_ago}/data")
def get_user(
    user_id: int = Path(..., title="Garmin Server ID", example=14),
    timestamp: str = Path(..., title="End timestamp of data to forecast", example="2025-01-16 15:30:00"),
    hours_ago: int = Path(..., title="Garmin Server ID", example=10)
  ):

  garmin_data = get_and_merge_data(user_id, timestamp, hours_ago)
  file_path = f'./app/tmp/combined_{user_id}_{timestamp}_{hours_ago}.xlsx'
  garmin_data.to_excel(
    file_path, 
    sheet_name='garmin'
  )

  # Return the excel file in API
  return FileResponse(
    path=file_path, 
    media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    filename=os.path.basename(file_path)
  )


@app.get("/users/{user_id}/{timestamp}/{hours_ago}/{n_minutes_ahead}/example")
def example(
    user_id: int = Path(..., title="Garmin Server's User ID", example=14),
    timestamp: str = Path(..., title="End timestamp of data to forecast", example="2025-01-16 15:30:00"),
    hours_ago: int = Path(..., title="Period used for forecasting", example=10),
    n_minutes_ahead: int = Path(..., title="Indicates how many minutes ahead to predict", example=60)
  ):

  target_timestamp = pd.to_datetime(timestamp) + pd.Timedelta(minutes=n_minutes_ahead)

  return {
    "user_id": user_id,
    "timestamp": timestamp,
    "hours_ago": hours_ago,
    "n_minutes_ahead": n_minutes_ahead,
    "target_timestamp": target_timestamp,
    "forecasts": float(random.randint(0, 1)),
    "error": "0"  # No error
  }