import pandas as pd
import numpy as np

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())


TARGET_FREQ = '15min'

start_date = pd.to_datetime("2025-05-19 00:00:15")
end_date = pd.to_datetime("2025-05-19 12:44:00")

file_name = f'Takasu01_garmin'
data_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\raw\Takasu01_garmin.xlsx"
# Heart rate
heart_rate = pd.read_excel(data_path,
                           sheet_name='Heart Rate', index_col='Timestamp',
                           engine='openpyxl')
heart_rate.sort_values('Timestamp', inplace=True)
heart_rate = heart_rate.rename(
    columns={'Heart Rate (in Beats per minute)': 'heart_rate'}
)

# Steps
steps = pd.read_excel(data_path,
                      sheet_name='Steps', index_col='Timestamp',
                      engine='openpyxl')
steps.sort_values('Timestamp', inplace=True)
steps = steps.rename(
    columns={'Number of Steps': 'steps'}
)

# Stress
stress = pd.read_excel(data_path,
                       sheet_name='Stress', index_col='Timestamp',
                       engine='openpyxl')
stress.sort_values('Timestamp', inplace=True)
stress = stress.rename(
    columns={'Stress Score': 'stress_score', 'Stress Interpretation': 'stress_interpretation'}
)

# Sleep
sleep = pd.read_excel(data_path,
                      sheet_name='Sleep', index_col='Calendar Date',
                      engine='openpyxl')
sleep.sort_values('Start Time', inplace=True)
# Compute duration in minutes
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
        start_date, end_date,
        freq=heart_rate_freq, name='Timestamp'
    )
).resample(heart_rate_freq).mean()
heart_rate = reference.merge(
    heart_rate.resample(heart_rate_freq).mean(), on='Timestamp', how='left'
).fillna(-1)

steps_freq = '15min'
reference = pd.DataFrame(
    index=pd.date_range(
        start_date, end_date,
        freq=steps_freq, name='Timestamp'
    )
).resample(steps_freq).mean()
steps = reference.merge(
    steps.resample(steps_freq).mean(), on='Timestamp', how='left'
).fillna(-1)

stress_freq = '3min'
reference = pd.DataFrame(
    index=pd.date_range(
        start_date, end_date,
        freq=stress_freq, name='Timestamp'
    )
).resample(stress_freq).mean()
stress = reference.merge(
    stress.loc[:, ['stress_score']].resample(stress_freq).mean(), on='Timestamp', how='left'
).fillna(-1)

sleep_freq = 'D'
reference = pd.DataFrame(
    index=pd.date_range(
        start_date, end_date,
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
        start_date, end_date,
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
    steps.resample(TARGET_FREQ).mean(), on='Timestamp', how='left'
).ffill()
garmin_data = garmin_data.merge(
    stress.resample(TARGET_FREQ).mean(), on='Timestamp', how='left'
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

garmin_data.to_excel(f'./data/preprocessed/{file_name}_preprocessed.xlsx', sheet_name='garmin')
print(heart_rate.head(5))