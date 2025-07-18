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
from app.train import data_loader, train_model

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

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
TARGET_FREQ = '15min'

# Check if model exists
if not os.path.exists('./app/models'):
  os.makedirs('./app/models')
# Check if any file is in model directory
if len(os.listdir('./app/models')) == 0:
  # Train the model
  train_model()
else:
  # Load latest keras model from models directory
  model_files = os.listdir('./app/models')
  model_files.sort(reverse=True)
  latest_model_file = model_files[0]
  latest_model_file = os.path.join('./app/models', latest_model_file)

  # Load the model
  multi_conv_model = tf.keras.models.load_model(
    latest_model_file,
    safe_mode=False
  )

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

load_dotenv()
DB_NAME = f"{os.getenv('DATABASE_NAME')}_{os.getenv('RAILS_ENV')}"
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
  db = SessionLocal()
  try:
    return db
  finally:
    db.close()

def get_heart_rate_from_db(user_id, start_timestamp, end_timestamp):
  db = get_db()
  start_calendar_date = start_timestamp - pd.Timedelta(days=1)
  end_calendar_date = end_timestamp

  query = text("""
    SELECT 
      (data->>'calendarDate')::timestamp + (key::int * INTERVAL '1 second') AS timestamp,
      value::int AS heart_rate
    FROM dailies, 
        jsonb_each_text(data->'timeOffsetHeartRateSamples')
    WHERE user_id = :user_id AND
      (data->>'calendarDate')::timestamp >= :start_calendar_date AND
      (data->>'calendarDate')::timestamp <= :end_calendar_date;
  """)

  data = db.execute(query, {
    "user_id": user_id,
    "start_calendar_date": start_calendar_date,
    "end_calendar_date": end_calendar_date,
  }).fetchall()
  
  data = pd.DataFrame(
    data, 
    columns=["Timestamp", "heart_rate"]
  )
  data['Timestamp'] = pd.to_datetime(data['Timestamp'])
  data = data.query(
    f"`Timestamp` >= '{start_timestamp}' and `Timestamp` <= '{end_timestamp}'"
  ).copy()
  
  return data

def get_stress_from_db(user_id, start_timestamp, end_timestamp):
  db = get_db()
  start_calendar_date = start_timestamp - pd.Timedelta(days=1)
  end_calendar_date = end_timestamp

  query = text("""
    SELECT 
      (data->>'calendarDate')::timestamp + (key::int * INTERVAL '1 second') AS timestamp,
      value::int AS stress_score
    FROM stresses, 
        jsonb_each_text(data->'timeOffsetStressLevelValues')
    WHERE user_id = :user_id AND
      (data->>'calendarDate')::timestamp >= :start_calendar_date AND
      (data->>'calendarDate')::timestamp <= :end_calendar_date;
  """)
  
  data = db.execute(query, {
    "user_id": user_id,
    "start_calendar_date": start_calendar_date,
    "end_calendar_date": end_calendar_date,
  }).fetchall()
  
  data = pd.DataFrame(
    data, 
    columns=["Timestamp", "stress_score"]
  )
  data['Timestamp'] = pd.to_datetime(data['Timestamp'])
  data = data.query(
    f"`Timestamp` >= '{start_timestamp}' and `Timestamp` <= '{end_timestamp}'"
  ).copy()

  return data

def get_steps_from_db(user_id, start_timestamp, end_timestamp):
  db = get_db()
  start_calendar_date = start_timestamp - pd.Timedelta(days=1)
  end_calendar_date = end_timestamp

  query = text("""               
    SELECT 
      TO_CHAR(TO_TIMESTAMP((epoch_data->>'startTimeInSeconds')::bigint), 'YYYY-MM-DD"T"HH24:MI:SS') AS timestamp,
      epoch_data->>'steps' AS steps
    FROM epoches,
      jsonb_array_elements(data::jsonb) AS epoch_data
    WHERE user_id = :user_id AND
      calendar_date::timestamp >= :start_calendar_date AND
      calendar_date::timestamp <= :end_calendar_date;
  """)
  
  data = db.execute(query, {
    "user_id": user_id,
    "start_calendar_date": start_calendar_date,
    "end_calendar_date": end_calendar_date,
  }).fetchall()
  
  data = pd.DataFrame(
    data, 
    columns=["Timestamp", "steps"]
  )
  data['Timestamp'] = pd.to_datetime(data['Timestamp'])
  data = data.query(
    f"`Timestamp` >= '{start_timestamp}' and `Timestamp` <= '{end_timestamp}'"
  ).copy()

  return data

def get_sleeps_from_db(user_id, start_timestamp, end_timestamp):
  db = get_db()
  start_calendar_date = start_timestamp - pd.Timedelta(days=1)
  end_calendar_date = end_timestamp

  query = text("""               
    SELECT 
      data->>'calendarDate' AS calendar_date,
      TO_CHAR(TO_TIMESTAMP((duration->>'startTimeInSeconds')::bigint) AT TIME ZONE 'Asia/Tokyo', 'YYYY-MM-DD"T"HH24:MI:SS') AS start_time,
      TO_CHAR(TO_TIMESTAMP((duration->>'endTimeInSeconds')::bigint) AT TIME ZONE 'Asia/Tokyo', 'YYYY-MM-DD"T"HH24:MI:SS') AS end_time,
      key AS sleep_type
    FROM sleeps, 
      jsonb_each(data->'sleepLevelsMap'),
      jsonb_array_elements(value) AS duration
    WHERE user_id = :user_id AND
      (data->>'calendarDate')::timestamp >= :start_calendar_date AND
      (data->>'calendarDate')::timestamp <= :end_calendar_date;
  """)
  
  data = db.execute(query, {
    "user_id": user_id,
    "start_calendar_date": start_calendar_date,
    "end_calendar_date": end_calendar_date,
  }).fetchall()
  
  data = pd.DataFrame(
    data, 
    columns=["Calendar Date", "Start Time", "End Time", "Sleep Type"]
  )
  data['Calendar Date'] = pd.to_datetime(data['Calendar Date'])
  data['Start Time'] = pd.to_datetime(data['Start Time'])
  data['End Time'] = pd.to_datetime(data['End Time'])

  return data

def get_and_merge_data(user_id, timestamp, hours_ago):
  # Get data from database
  end_timestamp = pd.to_datetime(timestamp)
  start_timestamp = end_timestamp - pd.Timedelta(hours = hours_ago)

  heart_rate = get_heart_rate_from_db(user_id, start_timestamp, end_timestamp)
  heart_rate.sort_values('Timestamp', inplace=True)
  heart_rate.set_index('Timestamp', inplace=True)

  stress = get_stress_from_db(user_id, start_timestamp, end_timestamp)
  stress.sort_values('Timestamp', inplace=True)
  stress.set_index('Timestamp', inplace=True)

  steps = get_steps_from_db(user_id, start_timestamp, end_timestamp)
  steps = steps.groupby('Timestamp').sum().reset_index()
  steps.sort_values('Timestamp', inplace=True)
  steps.set_index('Timestamp', inplace=True)
  steps = steps.astype({'steps': int})
  
  sleep = get_sleeps_from_db(user_id, start_timestamp, end_timestamp)
  sleep.sort_values('Start Time', inplace=True)
  # Compute duration in minutes
  sleep['End Time'] = pd.to_datetime(sleep['End Time'])
  sleep['Start Time'] = pd.to_datetime(sleep['Start Time'])
  sleep['Duration'] = (sleep['End Time'] - sleep['Start Time']) / np.timedelta64(1, "m")

  # Transform sleep data by sleep classification type
  sleep = sleep.pivot_table(
      index='Calendar Date',
      columns='Sleep Type',
      values='Duration',
      aggfunc='sum'
  )
  sleep = pd.DataFrame(sleep.to_records()).set_index('Calendar Date').fillna(0)
  # Make sure that sleep index is a DateTimeIndex type
  sleep.index = pd.to_datetime(sleep.index)
  sleep.index.name = 'Timestamp'

  if len(sleep) == 0:
    sleep_reference = pd.DataFrame([[0, 0, 0, 0]])
  else:
    sleep_reference = pd.DataFrame([[0, 0, 0, 0]] * len(sleep))
  sleep_reference.columns = ['awake', 'deep', 'light', 'rem']

  sleep_reference.set_index(sleep.index, inplace=True)
  sleep = sleep_reference.T.add(sleep.T, fill_value=0).T

  # Compute total non-rem sleep
  sleep['nonrem_total'] = (sleep['deep'] + sleep['light'])
  sleep['total'] = (sleep['nonrem_total'] + sleep['rem'])
  sleep['nonrem_percentage'] = sleep['nonrem_total'] / sleep['total']
  sleep['sleep_efficiency'] = sleep['total'] / (sleep['total'] + sleep['awake'])

  # Ignore unmeasurable column from sleep dataset
  if 'unmeasurable' in sleep.columns:
    sleep.drop(columns=['unmeasurable'], inplace=True)

  # Create reference
  heart_rate_freq = '15s'
  reference = pd.DataFrame(
      index=pd.date_range(
          start_timestamp, end_timestamp,
          freq=heart_rate_freq, name='Timestamp'
      )
  ).resample(heart_rate_freq).mean()
  heart_rate = reference.merge(
      heart_rate.resample(heart_rate_freq).mean(), on='Timestamp', how='left'
  ).fillna(-1)

  stress_freq = '3min'
  reference = pd.DataFrame(
    index=pd.date_range(
      start_timestamp, end_timestamp,
      freq=stress_freq, name='Timestamp'
    )
  ).resample(stress_freq).mean()
  stress = reference.merge(
    stress.resample(stress_freq).mean(), on='Timestamp', how='left'
  ).fillna(-1)

  steps_freq = '15min'
  reference = pd.DataFrame(
    index=pd.date_range(
      start_timestamp, end_timestamp,
      freq=steps_freq, name='Timestamp'
    )
  ).resample(steps_freq).mean()
  steps = reference.merge(
    steps.resample(steps_freq).mean(numeric_only=True), on='Timestamp', how='left'
  ).fillna(-1)

  sleep_freq = 'D'
  reference = pd.DataFrame(
    index=pd.date_range(
      start_timestamp, end_timestamp,
      freq=sleep_freq, name='Timestamp'
    )
  ).resample(sleep_freq).mean()
  sleep = reference.merge(
    sleep.resample(sleep_freq).mean(), on="Timestamp", how='left'
  ).fillna(-1)

  # Combine Garmin dataset
  # Create reference timestamp dataframe for the collection period
  reference = pd.DataFrame(
    index=pd.date_range(
      start_timestamp, end_timestamp,
      freq=TARGET_FREQ, name='Timestamp'
    )
  ).resample(TARGET_FREQ).mean()

  # Combine each Garmin dataset to reference timestamp dataframe
  garmin_data = reference.merge(
    # downsample heart rate from 15sec to 1min
    #   missing values = -1 same treatment with Garmin
    # with regards to missing value, fitness tracker not worn
    heart_rate.resample(TARGET_FREQ).mean(), on='Timestamp', how='left'
  ).ffill()
  garmin_data = garmin_data.merge(
    stress.resample(TARGET_FREQ).mean(), on='Timestamp', how='left'
  ).ffill()
  garmin_data = garmin_data.merge(
    steps.resample(TARGET_FREQ).mean(), on='Timestamp', how='left'
  ).ffill()
  garmin_data = garmin_data.merge(
    sleep.resample(TARGET_FREQ).mean(), on='Timestamp', how='left'
  ).ffill()

  garmin_data['timestamp_dayofweek'] = garmin_data.index.dayofweek
  # Fix timestamp format
  date_time = pd.to_datetime(garmin_data.index, format='%d.%m.%Y %H:%M:%S')

  # Convert to timestamp
  timestamp_s = date_time.map(pd.Timestamp.timestamp)

  # Get seconds per day
  day = 24 * 60 * 60 
  # Get seconds per year
  year = 365.2425 * day

  # Get sine(), cosine() for hour-feature
  garmin_data['timestamp_hour_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
  garmin_data['timestamp_hour_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
  garmin_data.fillna(0, inplace=True)

  return garmin_data

#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
class PredictResponse(BaseModel):
  user_id: int = Field(..., title="Garmin Server ID", example=1)
  timestamp: str = Field(..., 
    title="End timestamp of data to forecast", example="2025-01-12 12:00:00")
  forecasts: dict = Field(..., 
    title="1-hour forecast from the timestamp", example={
      "2025-01-12 13:00:00": 0.5, 
      "2025-01-12 14:00:00": 0.6,
      "2025-01-12 15:00:00": 0.6,
      "2025-01-12 16:00:00": 0.7,
    })


@app.get("/users/{user_id}/timestamp/{timestamp}/{hours_ago}/predict") #, response_model=PredictResponse)
def predict_user(
    user_id: int = Path(..., title="Garmin Server ID", example=14),
    timestamp: str = Path(..., title="End timestamp of data to forecast", example="2025-01-16 15:30:00"),
    hours_ago: int = Path(..., title="9時間以上でないと動かない(9時間未満の場合、9時間に修正)", example=10)
  ):

  if (hours_ago < 9): hours_ago = 9

  # while True:
  #   try:
  #     garmin_data = get_and_merge_data(user_id, timestamp, hours_ago)
  #     if len(garmin_data) > 0:
  #       break
  #     timestamp = (pd.to_datetime(timestamp) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
  #   except Exception as e:
  #     print(f"Error: {e}")
  #     timestamp = (pd.to_datetime(timestamp) - pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
  #     continue
  try:
    garmin_data = get_and_merge_data(user_id, timestamp, hours_ago)
    if len(garmin_data) > 0:
      return {"error": "No data found"}
  except Exception as e:
    return {"error": f"No data found: {e}"}
  
  

  garmin_data['wearing_off'] = None
  # Fill missing data with 0
  garmin_data.fillna(0, inplace=True)

  result = np.round(multi_conv_model.predict(data_loader(garmin_data))[0,:4]).astype(int)

  timelist = []
  for i in range(1, 5):
    timelist += [str(pd.to_datetime(timestamp) + pd.Timedelta(minutes=i*15))]

  predictions = pd.Series(data=result.flatten(), index=timelist).to_dict()

  return_dict = {
    "user_id": user_id,
    "timestamp": timestamp,
    "forecasts": result.flatten().tolist(),
    "timestamps": timelist
  }

  file_path = 'app/record.csv'
  if (os.path.isfile(file_path)):
    with open(file_path, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([user_id, timestamp, hours_ago, (result[0])[0], (result[1])[0], (result[2])[0], (result[3])[0]])
  else:
    with open(file_path, 'w') as f:
      writer = csv.writer(f)
      writer.writerow([user_id, timestamp, hours_ago, (result[0])[0], (result[1])[0], (result[2])[0], (result[3])[0]])

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
  columns = ["user_id", "timestamp", "hours_ago", "after1h", "after2h", "after3h", "after4h"]
    
  return pd.DataFrame(data = result, columns = columns).to_dict(orient='records')


@app.get("/users/{user_id}/timestamp/{timestamp}/{hours_ago}")
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


@app.get("/example/{user_id}/timestamp/{timestamp}/{hours_ago}") #, response_model=PredictResponse)
def example(
    user_id: int = Path(..., title="Garmin Server ID", example=14),
    timestamp: str = Path(..., title="End timestamp of data to forecast", example="2025-01-16 15:30:00"),
    hours_ago: int = Path(..., title="Garmin Server ID", example=10)
  ):

  timelist = [
    "2025-03-26 00:15:00",
    "2025-03-26 00:30:00",
    "2025-03-26 00:45:00",
    "2025-03-26 01:00:00"
  ]
  
  data = [random.randint(0, 1) for i in range(4)]

  predictions = pd.Series(data=data, index=timelist).to_dict()

  dict = {
    "user_id": user_id,
    "timestamp": timestamp,
    "forecasts": data,
    "timestamps": timelist
  }

  return dict