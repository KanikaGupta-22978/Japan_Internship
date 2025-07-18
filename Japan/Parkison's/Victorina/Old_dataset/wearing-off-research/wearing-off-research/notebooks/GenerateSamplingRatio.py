# %%
import xgboost as xgb
from scipy.special import expit as sigmoid
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import Normalizer

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# For visualization
from sklearn.decomposition import PCA
from openTSNE import TSNE
import umap.umap_ as umap

# For evaluation
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# For resampling
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# %% [markdown]
# # Load configuration

# %%
# Participant to process
USER = 'participant1'
# USER = f'participant{sys.argv[1]}'

# Collection dataset
# COLLECTION = '2-person'
COLLECTION = '10-person'
# COLLECTION = '3-person'

# Define base path
BASE_DATA_PATH = '/workspaces/data'

# Define results path
RESULTS_PATH = '/workspaces/results'

FIGSIZE = (20, 7)
FIGSIZE_CM = (13, 7)

# %%
# Choose features
# Garmin features
features = ['heart_rate', 'steps', 'stress_score',
            'awake', 'deep', 'light', 'rem',
            'nonrem_total', 'total', 'nonrem_percentage', 'sleep_efficiency']

# FonLog features
# features += ['time_from_last_drug_taken', 'wo_duration']

# Additional features
# features += ['timestamp_hour', 'timestamp_dayofweek',
features += ['timestamp_dayofweek',
             'timestamp_hour_sin', 'timestamp_hour_cos']

TARGET_COLUMN = 'wearing_off'
features.append(TARGET_COLUMN)

columns = ['timestamp'] + features + ['participant']

# Normalize features
normalize_features = features

# %%
# Metrics & Other Hyperparameters
SHIFT = 4
RECORD_SIZE_PER_DAY = 96  # 60 minutes / 15 minutes * 24 hours

# %%
# Test set periods
test_set_horizons = {
  "participant1": ["2021-12-02 0:00", "2021-12-03 23:45"],
  "participant2": ["2021-11-28 0:00", "2021-11-29 23:45"],
  "participant3": ["2021-11-25 0:00", "2021-11-26 23:45"],
  "participant4": ["2021-12-06 0:00", "2021-12-07 7:15"],
  "participant5": ["2021-11-28 0:00", "2021-11-29 23:45"],
  "participant6": ["2021-12-06 0:00", "2021-12-07 23:45"],
  "participant7": ["2021-12-12 0:00", "2021-12-13 9:45"],
  "participant8": ["2021-12-23 0:00", "2021-12-24 23:45"],
  "participant9": ["2021-12-23 0:00", "2021-12-24 23:45"],
  "participant10": ["2021-12-23 0:00", "2021-12-24 23:45"],
  "participant11": ["2023-01-30 0:00", "2023-01-31 23:45"],
  "participant12": ["2023-01-10 0:00", "2023-01-11 23:45"],
  "participant13": ["2023-01-29 0:00", "2023-01-30 23:45"],
}

# %% [markdown]
# # Load dataset

# %%
# Load participant's Excel file
dataset = pd.read_excel(f'{BASE_DATA_PATH}/{COLLECTION}/combined_data.xlsx',
                        index_col="timestamp",
                        usecols=columns,
                        engine='openpyxl')

# Fill missing data with 0
dataset.fillna(0, inplace=True)

# %%
# Load participant's Excel file
dataset2 = pd.read_excel(f'{BASE_DATA_PATH}/3-person/combined_data.xlsx',
                         index_col="timestamp",
                         usecols=columns,
                         engine='openpyxl')

# Fill missing data with 0
dataset2.fillna(0, inplace=True)

# Combine datasets
dataset = pd.concat([dataset, dataset2])

# %% [markdown]
# # Prepare dataset

# %% [markdown]
# ## Split dataset into train and test sets

# %%
# Keep a copy of the original dataset
original_dataset = dataset.copy()

# %%
# USER = 'participant13'
for USER in ['participant8', 'participant9', 'participant10', 'participant11', 'participant12', 'participant13']:
  # Filter by participant
  dataset = original_dataset.query(f'participant == "{USER}"').drop(
      columns=['participant']).copy()

  # %%
  train_df = dataset.loc[
    dataset.index < test_set_horizons[USER][0]
  ].copy()
  test_df = dataset.loc[test_set_horizons[USER][0]:].copy()

  # # Divide train_df to train_df and validation_df where validation_df is the last 20% of train_df
  # validation_df = train_df.iloc[int(len(train_df) * 0.8):].copy()
  # train_df = train_df.iloc[:int(len(train_df) * 0.8)].copy()

  # %% [markdown]
  # ## Transform series to supervised learning

  # %%
  # Convert series to supervised learning

  def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    var_names = data.columns
    n_vars = len(var_names)
    df = pd.DataFrame(data)
    cols, names = list(), list()  # new column values, new columne names

    # input sequence (t-i, ... t-1)
    # timesteps before (e.g., n_in = 3, t-3, t-2, t-1)
    for i in range(n_in, 0, -1):
      cols.append(df.shift(i))
      names += list(
          map(lambda var_name: f'{var_name}(t-{i})', var_names)
      )

    # forecast sequence (t, t+1, ... t+n)
    # timesteps after (e.g., n_out = 3, t, t+1, t+2)
    for i in range(0, n_out):
      cols.append(df.shift(-i))
      if i == 0:
        names += list(map(lambda var_name: f'{var_name}(t)', var_names))
      else:
        names += list(map(lambda var_name: f'{var_name}(t+{i})', var_names))

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
      agg.dropna(inplace=True)

    return agg

  # %%
  # Split into X and y

  def split_x_y(df, target_columns, SHIFT=SHIFT):
    # Drop extra columns i.e., (t+1), (t+2), (t+3), (t+4)
    regex = r".*\(t\+[1-{SHIFT}]\)$"  # includes data(t)
    # regex = r"\(t(\+([1-{SHIFT}]))?\)$" # removes data(t)

    # Drop extra columns except target_columns
    df.drop(
      [x for x in df.columns if re.search(
        regex, x) and x not in target_columns],
      axis=1, inplace=True
    )

    # Split into X and y
    y = df[target_columns].copy()
    X = df.drop(target_columns, axis=1)

    return (X, y)

  # %%
  N_IN = SHIFT  # Last hour
  N_OUT = SHIFT + 1  # Next hour

  # For a similar sliding window with TF's WindowGenerator,
  #   n_in = last day, t - 47
  #   n_out = next 1 hour

  # %%
  # Convert train set to supervised learning
  reframed_train_df = series_to_supervised(train_df,
                                           n_in=N_IN,
                                           n_out=N_OUT,
                                           dropnan=True)

  train_X, train_y = split_x_y(
      reframed_train_df, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])

  # display(train_y.head(5))
  # display(train_X.head(5))

  # %%
  # Convert test set to supervised learning
  # with train data's N_IN tail
  reframed_test_df = series_to_supervised(pd.concat([train_df.tail(N_IN),
                                                    test_df,
                                                     ]),
                                          n_in=N_IN,
                                          n_out=N_OUT,
                                          dropnan=True)
  test_X, test_y = split_x_y(
    reframed_test_df, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])

  '''test data only
  reframed_test_df = series_to_supervised(test_df,
                                          n_in=N_IN,
                                          n_out=N_OUT,
                                          dropnan=True)
  test_X, test_y = split_x_y(reframed_test_df, [f'{TARGET_COLUMN}(t+{SHIFT})'])
  '''

  '''with validation data's N_IN tail
  reframed_test_df = series_to_supervised(pd.concat([validation_df.tail(N_IN),
                                                    test_df,
                                                    ]),
                                          n_in=N_IN,
                                          n_out=N_OUT,
                                          dropnan=True)
  test_X, test_y = split_x_y(reframed_test_df, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])
  '''

  # display(test_y.head(5))
  # display(test_X.head(5))

  # %% [markdown]
  # ## Scale features

  # %%
  # Scale data
  scaler = MinMaxScaler(feature_range=(0, 1))
  # scaler = StandardScaler()
  train_X_scaled = scaler.fit_transform(train_X)
  train_X_scaled = pd.DataFrame(train_X_scaled,
                                columns=train_X.columns,
                                index=train_X.index)
  test_X_scaled = scaler.fit_transform(test_X)
  test_X_scaled = pd.DataFrame(test_X_scaled,
                               columns=test_X.columns,
                               index=test_X.index)

  # display(test_X_scaled.head(5))
  # display(train_X_scaled.head(5))

  # %% [markdown]
  # ## Normalize features

  # %%
  # Normalize data
  normalizer = Normalizer()
  train_X_scaled_normalized = normalizer.fit_transform(train_X_scaled)

  train_X_scaled_normalized = pd.DataFrame(train_X_scaled_normalized,
                                           columns=train_X.columns,
                                           index=train_X.index)

  test_X_scaled_normalized = normalizer.fit_transform(test_X_scaled)

  test_X_scaled_normalized = pd.DataFrame(test_X_scaled_normalized,
                                          columns=test_X.columns,
                                          index=test_X.index)

  # display(train_X_scaled_normalized.head(5))
  # display(test_X_scaled_normalized.head(5))

  # %%
  # Keep original data
  original_train_X_scaled_normalized = train_X_scaled_normalized.copy()
  original_train_y = train_y.copy()

  # %% [markdown]
  # ## Shuffle train set

  # %%
  # Shuffle train dataset
  # Combine train_y to train_X_scaled_normalized
  train = pd.concat(
      [original_train_X_scaled_normalized, original_train_y], axis=1)

  # Shuffle train
  train = train.sample(frac=1, random_state=4)

  # Split train into X and y
  train_y = train[f'{TARGET_COLUMN}(t+{SHIFT})']
  train_X_scaled_normalized = train.drop(f'{TARGET_COLUMN}(t+{SHIFT})', axis=1)

  # Original train_X, train_y
  original_train_X = train_X.copy()
  original_test_X = test_X.copy()

  # Renamed to train_X, train_y for easier reference
  train_X = train_X_scaled_normalized.copy()
  test_X = test_X_scaled_normalized.copy()

  train_X.head(5)

  # %% [markdown]
  # # Model Development

  # %%

  def logistic_obj(labels: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Logistic loss objective function for binary-class classification
    '''
    # grad = grad.flatten()
    # hess = hess.flatten()
    # return grad, hess
    y = labels
    p = sigmoid(predt)
    grad = p - y
    hess = p * (1.0 - p)

    return grad, hess

  # %%

  def adjust_class(conditional_probability, wrongprob, trueprob):
    a = conditional_probability / (wrongprob / trueprob)
    comp_cond = 1 - conditional_probability
    comp_wrong = 1 - wrongprob
    comp_true = 1 - trueprob
    b = comp_cond / (comp_wrong / comp_true)
    return a / (a + b)

  # %%
  # Check if file exists
  if not os.path.isfile(f'{RESULTS_PATH}/metric scores.xlsx'):
    # Create ExcelWriter
    writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                            engine='openpyxl', mode='w')

    # Create an empty DataFrame to write to Excel for Metric Scores
    pd.DataFrame(
      columns=['participant', 'model',
               'f1 score', 'recall', 'precision', 'accuracy',
               'auc-roc', 'auc-prc']
    ).to_excel(excel_writer=writer, sheet_name='Metric Scores', index=False)

    # Create an empty DataFrame to write to Excel for Classification Report
    pd.DataFrame(
      columns=['participant', 'model', 'classification report',
               'precision', 'recall', 'f1-score', 'support']
    ).to_excel(excel_writer=writer, sheet_name='Classification Report', index=False)

    # Create an empty DataFrame to write to Excel for Confusion Matrix
    pd.DataFrame(
      columns=['participant', 'model', 'TN', 'FP', 'FN', 'TP']
    ).to_excel(excel_writer=writer, sheet_name='Confusion Matrix', index=False)

    # Create an empty DataFrame to write to Excel for Sampling Ratio
    pd.DataFrame(
      columns=['participant', 'model', 'original_N0', 'original_N1', 'resampled_N0',
               'resampled_N1', 'sampling_rate']
    ).to_excel(excel_writer=writer, sheet_name='Sampling Ratio', index=False)

    writer.close()

  # %% [markdown]
  # # Oversampled Model

  # %%
  model_name = 'Oversampled SMOTE Model'

  # Oversample subsampled_X, subsampled_y using SMOTE
  sm = SMOTE(random_state=4, k_neighbors=5)
  oversampled_X, oversampled_y = sm.fit_resample(train_X, train_y)

  # Create XGBClassifier with custom objective function model instance
  oversampled_model = XGBClassifier(objective=logistic_obj,
                                    random_state=4, n_estimators=1000)
  # fit model using oversampled train data
  oversampled_model.fit(oversampled_X, oversampled_y)

  # save XGBClassifier model
  oversampled_model.save_model(f'{RESULTS_PATH}/{USER}, {model_name}.json')

  # %%
  # Combine these two into one dataframe
  ratios = pd.DataFrame({
    'original_N0': train_y.value_counts()[0],
    'original_N1': train_y.value_counts()[1],
    'resampled_N0': oversampled_y.value_counts()[0],
    'resampled_N1': oversampled_y.value_counts()[1]
  }, index=[0]).assign(participant=USER, model=model_name)

  ratios.set_index(['participant', 'model'], inplace=True)
  ratios.reset_index(inplace=True)

  # Recreate writer to open existing file
  writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                          engine='openpyxl', mode='a', if_sheet_exists='overlay')

  # Append data frame to Metric Scores sheet
  ratios.to_excel(excel_writer=writer, sheet_name='Sampling Ratio',
                  startrow=writer.sheets['Sampling Ratio'].max_row,
                  header=False, index=False)
  writer.close()

  # %% [markdown]
  # # Adj. Oversampled Model

  # %%
  model_name = 'Adj. Oversampled SMOTE Model'

  # Oversample subsampled_X, subsampled_y using SMOTE
  sm = SMOTE(random_state=4, k_neighbors=5)
  oversampled_X, oversampled_y = sm.fit_resample(train_X, train_y)

  # Create XGBClassifier with custom objective function model instance
  oversampled_model = XGBClassifier(objective=logistic_obj,
                                    random_state=4, n_estimators=1000)
  # fit model using oversampled train data
  oversampled_model.fit(oversampled_X, oversampled_y)

  # save XGBClassifier model
  oversampled_model.save_model(f'{RESULTS_PATH}/{USER}, {model_name}.json')

  # %%
  # Combine these two into one dataframe
  ratios = pd.DataFrame({
    'original_N0': train_y.value_counts()[0],
    'original_N1': train_y.value_counts()[1],
    'resampled_N0': oversampled_y.value_counts()[0],
    'resampled_N1': oversampled_y.value_counts()[1]
  }, index=[0]).assign(participant=USER, model=model_name)

  ratios.set_index(['participant', 'model'], inplace=True)
  ratios.reset_index(inplace=True)

  # Recreate writer to open existing file
  writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                          engine='openpyxl', mode='a', if_sheet_exists='overlay')

  # Append data frame to Metric Scores sheet
  ratios.to_excel(excel_writer=writer, sheet_name='Sampling Ratio',
                  startrow=writer.sheets['Sampling Ratio'].max_row,
                  header=False, index=False)
  writer.close()

  # %% [markdown]
  # # Undersampled+Oversampled Model

  # %%
  model_name = 'Undersampled+Oversampled Model'

  nm = NearMiss()

  undersampled_X, undersampled_y = nm.fit_resample(train_X, train_y)
  # Combine so that we can filter together
  undersampled_train = pd.concat([undersampled_X, undersampled_y], axis=1)
  # Get rows with wearing_off(t+4) == 0
  undersampled_train = undersampled_train[undersampled_train.iloc[:, -1] == 0].copy()

  sm = SMOTE(random_state=4, k_neighbors=5)

  oversampled_X, oversampled_y = sm.fit_resample(train_X, train_y)
  # Combine so that we can filter together
  oversampled_train = pd.concat([oversampled_X, oversampled_y], axis=1)
  # Get rows with wearing_off(t+4) == 1
  oversampled_train = oversampled_train[oversampled_train.iloc[:, -1] == 1].copy()

  resampled_train = pd.concat([undersampled_train, oversampled_train], axis=0)
  resampled_X, resampled_y = split_x_y(
      resampled_train, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])

  # %%
  # Combine these two into one dataframe
  ratios = pd.DataFrame({
    'original_N0': train_y.value_counts()[0],
    'original_N1': train_y.value_counts()[1],
    'resampled_N0': resampled_y.value_counts()[0],
    'resampled_N1': resampled_y.value_counts()[1]
  }, index=[0]).assign(participant=USER, model=model_name)

  ratios.set_index(['participant', 'model'], inplace=True)
  ratios.reset_index(inplace=True)

  # Recreate writer to open existing file
  writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                          engine='openpyxl', mode='a', if_sheet_exists='overlay')

  # Append data frame to Metric Scores sheet
  ratios.to_excel(excel_writer=writer, sheet_name='Sampling Ratio',
                  startrow=writer.sheets['Sampling Ratio'].max_row,
                  header=False, index=False)
  writer.close()

  # %% [markdown]
  # # Adj. Undersampled+Oversampled Model

  # %%
  model_name = 'Adj. Undersampled+Oversampled Model'

  nm = NearMiss()

  undersampled_X, undersampled_y = nm.fit_resample(train_X, train_y)
  # Combine so that we can filter together
  undersampled_train = pd.concat([undersampled_X, undersampled_y], axis=1)
  # Get rows with wearing_off(t+4) == 0
  undersampled_train = undersampled_train[undersampled_train.iloc[:, -1] == 0].copy()

  sm = SMOTE(random_state=4, k_neighbors=5)

  oversampled_X, oversampled_y = sm.fit_resample(train_X, train_y)
  # Combine so that we can filter together
  oversampled_train = pd.concat([oversampled_X, oversampled_y], axis=1)
  # Get rows with wearing_off(t+4) == 1
  oversampled_train = oversampled_train[oversampled_train.iloc[:, -1] == 1].copy()

  resampled_train = pd.concat([undersampled_train, oversampled_train], axis=0)
  resampled_X, resampled_y = split_x_y(
      resampled_train, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])

  # %%
  # Combine these two into one dataframe
  ratios = pd.DataFrame({
    'original_N0': train_y.value_counts()[0],
    'original_N1': train_y.value_counts()[1],
    'resampled_N0': resampled_y.value_counts()[0],
    'resampled_N1': resampled_y.value_counts()[1]
  }, index=[0]).assign(participant=USER, model=model_name)

  ratios.set_index(['participant', 'model'], inplace=True)
  ratios.reset_index(inplace=True)

  # Recreate writer to open existing file
  writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                          engine='openpyxl', mode='a', if_sheet_exists='overlay')

  # Append data frame to Metric Scores sheet
  ratios.to_excel(excel_writer=writer, sheet_name='Sampling Ratio',
                  startrow=writer.sheets['Sampling Ratio'].max_row,
                  header=False, index=False)
  writer.close()

  # %% [markdown]
  # # SMOTETomek Model

  # %%
  model_name = 'SMOTETomek Resampled Model'

  smote_tomek = SMOTETomek(random_state=4)

  resampled_X, resampled_y = sm.fit_resample(train_X, train_y)

  # %%
  # Combine these two into one dataframe
  ratios = pd.DataFrame({
    'original_N0': train_y.value_counts()[0],
    'original_N1': train_y.value_counts()[1],
    'resampled_N0': resampled_y.value_counts()[0],
    'resampled_N1': resampled_y.value_counts()[1]
  }, index=[0]).assign(participant=USER, model=model_name)

  ratios.set_index(['participant', 'model'], inplace=True)
  ratios.reset_index(inplace=True)

  # Recreate writer to open existing file
  writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                          engine='openpyxl', mode='a', if_sheet_exists='overlay')

  # Append data frame to Metric Scores sheet
  ratios.to_excel(excel_writer=writer, sheet_name='Sampling Ratio',
                  startrow=writer.sheets['Sampling Ratio'].max_row,
                  header=False, index=False)
  writer.close()

  # %% [markdown]
  # # Adj. SMOTETomek Model

  # %%
  model_name = 'Adj. SMOTETomek Resampled Model'

  smote_tomek = SMOTETomek(random_state=4)

  resampled_X, resampled_y = sm.fit_resample(train_X, train_y)

  # %%
  # Combine these two into one dataframe
  ratios = pd.DataFrame({
    'original_N0': train_y.value_counts()[0],
    'original_N1': train_y.value_counts()[1],
    'resampled_N0': resampled_y.value_counts()[0],
    'resampled_N1': resampled_y.value_counts()[1]
  }, index=[0]).assign(participant=USER, model=model_name)

  ratios.set_index(['participant', 'model'], inplace=True)
  ratios.reset_index(inplace=True)

  # Recreate writer to open existing file
  writer = pd.ExcelWriter(f'{RESULTS_PATH}/metric scores.xlsx',
                          engine='openpyxl', mode='a', if_sheet_exists='overlay')

  # Append data frame to Metric Scores sheet
  ratios.to_excel(excel_writer=writer, sheet_name='Sampling Ratio',
                  startrow=writer.sheets['Sampling Ratio'].max_row,
                  header=False, index=False)
  writer.close()
