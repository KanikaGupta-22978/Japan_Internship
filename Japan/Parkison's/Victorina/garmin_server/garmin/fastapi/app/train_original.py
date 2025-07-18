import json
import pandas as pd
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras import backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

TARGET_FREQ = '15min'

columns = [ 'timestamp', 'heart_rate', 'steps', 'stress_score',
            'awake', 'deep', 'light', 'rem', 
            'nonrem_total', 'total', 'nonrem_percentage', 'sleep_efficiency']

# Include FonLog data
# columns += ['time_from_last_drug_taken'] #, 'wo_duration']

# Additional data
columns += ['timestamp_dayofweek', 'timestamp_hour_sin', 'timestamp_hour_cos']

# 'wearing_off' | 'wearing_off_post_meds' | 'wearing_off_lead60'
target_column = 'wearing_off' 
columns.append(target_column)

participant_dictionary = json.load(open(f'./app/data/participant_dictionary.json'))

# CV splits
if TARGET_FREQ == '15min':
  record_size_per_day = 96
elif TARGET_FREQ == '15s':
  record_size_per_day = 5760
elif TARGET_FREQ == '1min':
  record_size_per_day = 1440

METRICS = [
  tf.keras.metrics.TruePositives(name='tp'),
  tf.keras.metrics.FalsePositives(name='fp'),
  tf.keras.metrics.TrueNegatives(name='tn'),
  tf.keras.metrics.FalseNegatives(name='fn'), 
  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),
  tf.keras.metrics.AUC(name='auc'),
  tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
#       BalancedSparseCategoricalAccuracy(),
#       BalancedAccuracy()
# ]

BATCH_SIZE = 1
MAX_EPOCHS = 10
LEARNING_RATE = 1e-3
SHIFT = 4 # 1 = 15 min, 2 = 30 min, 4 = 1 hour
MULTI_STEP_WIDTH = 36 # input 36 = 9 hours, input 96 = 24 hours
USE_HOURLY = False
SAVEFIG = False
EXPERIMENT_NAME = 'with wearing-off'
REMOVE_WEARING_OFF_IN_PREVIOUS_STEP = False

# features to normalize
# timestamp_dayofweek, wearing_off were not normalized
normalize_features = ['heart_rate', 'steps', 'stress_score', 'awake', 'deep', 
                      'light', 'rem', 'nonrem_total', 'total', 'nonrem_percentage',
                      'sleep_efficiency', 'timestamp_hour_sin', 'timestamp_hour_cos']
def normalize_data(df, mean, std, normalize_features=normalize_features):
    df_to_normalize = df.copy()
    df_to_normalize.loc[:, normalize_features] = ((
        df_to_normalize.loc[:, normalize_features] - mean
    ) / std)
    
    return df_to_normalize


def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min',
                                                    restore_best_weights=True)

  # model.compile(loss=tf.keras.losses.MeanSquaredError(),
  #               optimizer=tf.keras.optimizers.Adam(),
  #               metrics=[tf.keras.metrics.MeanAbsoluteError()])
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                metrics=METRICS)

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


def data_loader(new_df):
    return np.array(new_df, dtype=np.float32)[np.newaxis, ...]


combined_dataset = None
for participant_number in range(1,11):
    user = f'participant{participant_number}'
    dataset = pd.read_excel(f'./app/data/4-combined_data_{user}_{TARGET_FREQ}.xlsx',
                                  index_col="timestamp",
                                  usecols=columns,
                                  engine='openpyxl')
        
    # Fill missing data with 0
    dataset.fillna(0, inplace=True)

    # Filter data based on participants' dictionary
    dataset = dataset.loc[
        (dataset.index >= participant_dictionary[user]['start_date']) &
        (dataset.index < participant_dictionary[user]['end_date_plus_two'])
    ]
    combined_dataset = pd.concat([combined_dataset, dataset])

dataset = combined_dataset
column_indices = { name: i for i, name in enumerate(dataset.columns) }
df = dataset


# training data 60% 
TRAINING_PERCENTAGE = 0.6
# validation data 20%
VALIDATION_PERCENTAGE = 0.2

column_indices = { name: i for i, name in enumerate(df.columns) }
total_rows = len(df)
num_features = len(df.columns)

training_end_index = int(total_rows * TRAINING_PERCENTAGE)
validation_end_index = int(total_rows * (TRAINING_PERCENTAGE + VALIDATION_PERCENTAGE))

train_df = df[0:training_end_index].copy()
val_df = df[training_end_index:validation_end_index].copy()
test_df = df[validation_end_index:].copy()

train_mean = train_df.loc[:, normalize_features].mean()
train_std = train_df.loc[:, normalize_features].std()

train_df = normalize_data(train_df, train_mean, train_std)
val_df = normalize_data(val_df, train_mean, train_std)
test_df = normalize_data(test_df, train_mean, train_std)

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
               label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [h]')


  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        seed=4,
        batch_size=BATCH_SIZE,)

    ds = ds.map(self.split_window)

    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result



def train_model():
  CONV_WIDTH = MULTI_STEP_WIDTH
  OUT_STEPS = SHIFT

  multi_window = WindowGenerator(input_width=MULTI_STEP_WIDTH,
                                label_width=OUT_STEPS,
                                shift=OUT_STEPS,
                                label_columns=['wearing_off']
                                )

  K.clear_session()
  multi_conv_model = tf.keras.Sequential([
      tf.keras.layers.BatchNormalization(),
      # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
      tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
      # Shape => [batch, 1, conv_units]
      tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
      # Shape => [batch, 1,  out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS,
                            activation='sigmoid',
                            kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, -1])


      # tf.keras.layers.Conv1D(filters=64,
      #                        kernel_size=(MULTI_STEP_WIDTH,),
      #                        activation='relu'),
      # tf.keras.layers.Dense(units=64, activation='relu'),
      # tf.keras.layers.Dense(units=4, activation='sigmoid', name="output"),
  ])

  history = compile_and_fit(multi_conv_model, multi_window)

  # Save the model with timestamp using `model.save('my_model.keras')
  try:
    multi_conv_model.save(f'./app/models/{datetime.now().strftime("%Y%m%d-%H%M%S")}.keras')
  except Exception as e:
    print(f"Error saving model: {e}")