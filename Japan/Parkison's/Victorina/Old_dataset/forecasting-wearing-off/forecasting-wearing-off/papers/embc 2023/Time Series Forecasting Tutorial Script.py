# %% [markdown]
# # Load libraries

# %%
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import backend as K

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from sklearn.model_selection import LeaveOneGroupOut

# %% [markdown]
# # Load configuration

# %%
# Garmin features
features = ['heart_rate', 'steps', 'stress_score',
           'awake', 'deep', 'light', 'rem', 
           'nonrem_total', 'total', 'nonrem_percentage', 'sleep_efficiency']

# Additional features
#   'timestamp_hour'
features += ['timestamp_dayofweek', 'timestamp_hour_sin', 'timestamp_hour_cos']

TARGET_COLUMN = 'wearing_off'
features.append(TARGET_COLUMN)

columns = ['timestamp'] + features + ['participant']

# %% [markdown]
# # Load dataset

# %%
# Combine participant 11-13
# df = pd.read_excel('./data/4-combined_data.xlsx',
#                   index_col="timestamp",
#                   engine='openpyxl')

# df13 = pd.read_excel('./data/4-combined_data_participant13_15min.xlsx',
#                   index_col="timestamp",
#                   engine='openpyxl')
# df13 = df13.assign(participant=13)

# pd.concat([df, df13], axis=0).drop(columns=['activity_type_id']).to_excel('./data/4-combined_data.xlsx')

# %%
df = pd.read_excel('./data/4-combined_data.xlsx',
                  index_col="timestamp",
                  usecols=columns,
                  engine='openpyxl')

# Fill missing data with 0
df.fillna(0, inplace=True)

# %%
df.head()

# %%
# Remove days without wearing-off record
#   * not sure whether there is no actual wearing-off periods that day
#   * or if there were actual wearing-off periods but not recorded
df.pivot_table(values='wearing_off', index=["participant"], aggfunc='count')

df_day = df.resample('D').sum()
days_without_wearing_off = list(df_day.query('wearing_off != 0').index)
days_without_wearing_off = [day.date().strftime('%Y-%m-%d') for day in days_without_wearing_off]

df['date'] = pd.to_datetime(df.index.date)
df[df['date'].dt.date.astype(str).isin(days_without_wearing_off)].pivot_table(values='wearing_off', index=["participant"], aggfunc='count')

# %% [markdown]
# # Split dataset

# %%
for PARTICIPANT_AS_TEST in [1,2,3,4,5,6,7,8,9,10,12,13]:
  # PARTICIPANT_AS_TEST = 13

  column_indices = {name: i for i, name in enumerate(df.columns)}
  # Select all except specified participant (used for training)
  general_df = df.query(f'participant != {PARTICIPANT_AS_TEST}')[features].copy()
  train_df = general_df[0:int( len(general_df) * 0.6 )]
  val_df = general_df[int( len(general_df) * 0.6 ):]

  test_participant_df = df.query(f'participant == {PARTICIPANT_AS_TEST}')[features].copy()
  fine_tuning_df = test_participant_df[0:int( len(test_participant_df) * 0.5 )]
  test_df = test_participant_df[int( len(test_participant_df) * 0.5 ):]

  # %% [markdown]
  # # Normalize dataset

  # %%
  normalize_features = ['heart_rate', 'steps', 'stress_score', 'awake', 'deep', 
                        'light', 'rem', 'nonrem_total', 'total', 'nonrem_percentage',
                        'sleep_efficiency', 'timestamp_hour_sin', 'timestamp_hour_cos']
  def normalize_data(df, mean, std, normalize_features=normalize_features):
      df_to_normalize = df.copy()
      df_to_normalize.loc[:, normalize_features] = ((
          df_to_normalize.loc[:, normalize_features] - mean
      ) / std)
      
      return df_to_normalize

  # %%
  general_mean = general_df.loc[:, normalize_features].mean()
  general_std = general_df.loc[:, normalize_features].std()

  train_df = normalize_data(train_df, general_mean, general_std)
  val_df = normalize_data(val_df, general_mean, general_std)
  fine_tuning_df = normalize_data(fine_tuning_df, general_mean, general_std)
  test_df = normalize_data(test_df, general_mean, general_std)

  # %% [markdown]
  # # WindowGenerator

  # %%
  BATCH_SIZE = 32
  SHIFT = 4
  RECORD_SIZE_PER_DAY = 96


  class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df, val_df, fine_tuning_df, test_df, 
                label_columns=None, batch_size=1):
      # Store the raw data.
      self.train_df = train_df
      self.val_df = val_df
      self.fine_tuning_df = fine_tuning_df
      self.test_df = test_df
      self.batch_size = batch_size

      # Work out the label column indices.
      self.label_columns = label_columns
      if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in
                                      enumerate(label_columns)}
      self.column_indices = {name: i for i, name in
                            enumerate(self.train_df.columns)}
      
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
    
    def plot(self, model=None, plot_col='wearing_off', max_subplots=3):
      inputs, labels = self.example
      
      fig = plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        ax = plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        
        # NEW
        plt.ylim(-0.1,1.1) 
        ax.set_yticks(
            [0.0, 0.5, 1.0]
        )
        ax.set_xticks([])
        if n == 2:
            ax.set_xticks(
                np.append(self.input_indices[::SHIFT], self.input_indices[-1] + 1),
                list(range(0, len( self.input_indices[::SHIFT] ) + 1 )),
                minor=True
            )
        # NEW

        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)
        if self.label_columns:
          label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
          label_col_index = plot_col_index
        if label_col_index is None:
          continue

        # plt.scatter(self.label_indices, labels[n, :, label_col_index],
        #             edgecolors='k', label='Labels', c='#2ca02c', s=64)
        plt.scatter(self.label_indices[::SHIFT], labels[n, :, label_col_index][::SHIFT],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
          predictions = model(inputs)
          plt.scatter(self.label_indices[::SHIFT], predictions[n, :, label_col_index][::SHIFT],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)
          # plt.scatter(self.label_indices, predictions[n, :, label_col_index],
          #             marker='X', edgecolors='k', label='Predictions',
          #             c='#ff7f0e', s=64)

        if n == 2:
            # Put a legend below current axis
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.025),
                        bbox_transform=fig.transFigure,
                      fancybox=True, shadow=True, ncol=3)
            # plt.legend()
        # if n == 0:
        #   plt.legend()


      plt.xlabel('Time [h]')

    def make_dataset(self, data):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=False,
        seed=4,
        batch_size=BATCH_SIZE,).shuffle(buffer_size=10000)

      ds = ds.map(self.split_window)

      return ds
    
    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def fine_tuning(self):
      return self.make_dataset(self.fine_tuning_df)

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

  # %% [markdown]
  # ## Single Step Window

  # %%
  single_step_window = WindowGenerator(
      train_df=train_df, val_df=val_df,
      fine_tuning_df=fine_tuning_df, test_df=test_df,
      input_width=1, label_width=1, shift=SHIFT,
      label_columns=[TARGET_COLUMN], batch_size=BATCH_SIZE)
  # print(single_step_window)
  # for example_inputs, example_labels in single_step_window.train.take(1):
  #   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  #   print(f'Labels shape (batch, time, features): {example_labels.shape}')
  # single_step_window.plot(plot_col=TARGET_COLUMN)

  # %% [markdown]
  # ## Wide Window

  # %%
  wide_window = WindowGenerator(
      train_df=train_df, val_df=val_df,
      fine_tuning_df=fine_tuning_df, test_df=test_df,
      input_width=24, label_width=24, shift=SHIFT,
      label_columns=[TARGET_COLUMN], batch_size=BATCH_SIZE)
  # print(wide_window)
  # for example_inputs, example_labels in wide_window.train.take(1):
  #   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  #   print(f'Labels shape (batch, time, features): {example_labels.shape}')
  # wide_window.plot(plot_col=TARGET_COLUMN)

  # %% [markdown]
  # ## Convolutional Window

  # %%
  CONV_WIDTH = RECORD_SIZE_PER_DAY # 24 hours or 96 records
  conv_window = WindowGenerator(
      train_df=train_df, val_df=val_df,
      fine_tuning_df=fine_tuning_df, test_df=test_df,
      input_width=CONV_WIDTH, label_width=1, shift=SHIFT,
      label_columns=[TARGET_COLUMN], batch_size=BATCH_SIZE)
  # print(conv_window)
  # for example_inputs, example_labels in conv_window.train.take(1):
  #   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  #   print(f'Labels shape (batch, time, features): {example_labels.shape}')
  # conv_window.plot(plot_col=TARGET_COLUMN)

  # %% [markdown]
  # ## Wide Convolutional Window

  # %%
  LABEL_WIDTH = CONV_WIDTH # 4 * 6 # 4 records * 6 hours
  INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
  wide_conv_window = WindowGenerator(
      train_df=train_df, val_df=val_df,
      fine_tuning_df=fine_tuning_df, test_df=test_df,
      input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
      label_columns=[TARGET_COLUMN], batch_size=BATCH_SIZE)
  print(wide_conv_window)
  for example_inputs, example_labels in wide_conv_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
  wide_conv_window.plot(plot_col=TARGET_COLUMN)

  # %% [markdown]
  # # Model Training

  # %%
  MAX_EPOCHS = 20
  LEARNING_RATE = 1e-3
  METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR')] # precision-recall curve]


  def brier_score_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

  def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    # model.compile(loss=tf.keras.losses.MeanSquaredError(),
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #               metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=METRICS)
    # model.compile(loss=brier_score_loss,
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #               metrics=METRICS)

    # history = model.fit(window.train, epochs=MAX_EPOCHS,
    #                     validation_data=window.val,
    #                     callbacks=[early_stopping])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    
    return history

  def fine_tune(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=METRICS)
    
    # model.compile(loss=brier_score_loss,
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #               metrics=METRICS)

    history = model.fit(window.fine_tuning, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    
    return history

  # %%
  val_performance = {}
  before_fine_tuning_performance = {}
  performance = {}

  # %% [markdown]
  # ## Baseline

  # %%
  class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
      super().__init__()
      self.label_index = label_index

    def call(self, inputs):
      if self.label_index is None:
        return inputs
      result = inputs[:, :, self.label_index]
      return result[:, :, tf.newaxis]

  # %%
  K.clear_session()
  baseline = Baseline(label_index=column_indices[TARGET_COLUMN])
  history = compile_and_fit(baseline, single_step_window)

  # %%
  val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
  before_fine_tuning_performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
  for name, value in zip(baseline.metrics_names, before_fine_tuning_performance['Baseline']):
    print(name, ': ', value)

  # %%
  history = fine_tune(baseline, single_step_window)

  # %%
  performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
  for name, value in zip(baseline.metrics_names, performance['Baseline']):
    print(name, ': ', value)

  # %%
  wide_window.plot(model=baseline, plot_col="wearing_off")

  # %% [markdown]
  # ## Linear

  # %%
  K.clear_session()
  linear = tf.keras.Sequential([
      tf.keras.layers.Dense(units=1, activation="sigmoid")
  ])
  history = compile_and_fit(linear, single_step_window)

  # %%
  val_performance['Linear'] = linear.evaluate(single_step_window.val)
  before_fine_tuning_performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
  for name, value in zip(linear.metrics_names, before_fine_tuning_performance['Linear']):
    print(name, ': ', value)

  # %%
  history = fine_tune(linear, single_step_window)

  # %%
  performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
  for name, value in zip(linear.metrics_names, performance['Linear']):
    print(name, ': ', value)

  # %%
  wide_window.plot(model=linear, plot_col="wearing_off")

  # %% [markdown]
  # ## Dense

  # %%
  K.clear_session()
  dense = tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=1, activation="sigmoid")

  ])
  history = compile_and_fit(dense, single_step_window)

  # %%
  val_performance['Dense'] = dense.evaluate(single_step_window.val)
  before_fine_tuning_performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
  for name, value in zip(dense.metrics_names, before_fine_tuning_performance['Dense']):
    print(name, ': ', value)

  # %%
  history = fine_tune(dense, single_step_window)

  # %%
  performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
  for name, value in zip(dense.metrics_names, performance['Dense']):
    print(name, ': ', value)

  # %%
  wide_window.plot(model=dense, plot_col="wearing_off")

  # %% [markdown]
  # ## Multi-step Dense

  # %%
  K.clear_session()
  multi_step_dense = tf.keras.Sequential([
      # Shape: (time, features) => (time*features)
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='sigmoid'),
      # Add back the time dimension.
      # Shape: (outputs) => (1, outputs)
      tf.keras.layers.Reshape([1, -1]),
  ])
  history = compile_and_fit(multi_step_dense, conv_window)

  # %%
  val_performance['Multi step Dense'] = multi_step_dense.evaluate(conv_window.val)
  before_fine_tuning_performance['Multi step Dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
  for name, value in zip(multi_step_dense.metrics_names, before_fine_tuning_performance['Multi step Dense']):
    print(name, ': ', value)

  # %%
  history = fine_tune(multi_step_dense, conv_window)

  # %%
  performance['Multi step Dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
  for name, value in zip(multi_step_dense.metrics_names, performance['Multi step Dense']):
    print(name, ': ', value)

  # %%
  conv_window.plot(model=multi_step_dense, plot_col="wearing_off")

  # %% [markdown]
  # ## CNN

  # %%
  K.clear_session()
  conv_model = tf.keras.Sequential([
      tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(CONV_WIDTH,),
                            activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='sigmoid'),
  ])
  history = compile_and_fit(conv_model, conv_window)

  # %%
  val_performance['CNN'] = conv_model.evaluate(conv_window.val)
  before_fine_tuning_performance['CNN'] = conv_model.evaluate(conv_window.test, verbose=0)
  for name, value in zip(conv_model.metrics_names, before_fine_tuning_performance['CNN']):
    print(name, ': ', value)

  # %%
  history = fine_tune(conv_model, conv_window)

  # %%
  performance['CNN'] = conv_model.evaluate(conv_window.test, verbose=0)
  for name, value in zip(conv_model.metrics_names, performance['CNN']):
    print(name, ': ', value)

  # %%
  wide_conv_window.plot(model=conv_model, plot_col='wearing_off')

  # %% [markdown]
  # ## LSTM

  # %%
  K.clear_session()
  lstm_model = tf.keras.models.Sequential([
      # Shape [batch, time, features] => [batch, time, lstm_units]
      tf.keras.layers.LSTM(64, return_sequences=True),
      # Shape => [batch, time, features]
      tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])

  history = compile_and_fit(lstm_model, wide_window)


  # %%
  val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
  before_fine_tuning_performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
  for name, value in zip(lstm_model.metrics_names, before_fine_tuning_performance['LSTM']):
    print(name, ': ', value)

  # %%
  history = fine_tune(lstm_model, wide_window)

  # %%
  performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
  for name, value in zip(lstm_model.metrics_names, performance['LSTM']):
    print(name, ': ', value)

  # %%
  wide_window.plot(model=lstm_model, plot_col='wearing_off')

  # %% [markdown]
  # ## Performance

  # %%
  x = np.arange(len(performance))
  width = 0.3
  metric_name = 'prc'
  metric_index = lstm_model.metrics_names.index(metric_name)
  # val_mae = [v[metric_index] for v in val_performance.values()]
  before_fine_tuning_mae = [v[metric_index] for v in before_fine_tuning_performance.values()]
  test_mae = [v[metric_index] for v in performance.values()]

  plt.ylabel(f'{metric_name} [wearing_off, normalized]')
  # plt.bar(x - 0.17, before_fine_tuning_mae, width, label='Validation')
  plt.bar(x - 0.17, before_fine_tuning_mae, width, label='Before Fine-Tuning Test')
  plt.bar(x + 0.17, test_mae, width, label='After Fine-Tuning Test')
  plt.xticks(ticks=x, labels=performance.keys(),
            rotation=45)
  _ = plt.legend()

  # %%
  if os.path.exists(f'./results/final_performance.xlsx'):
      with pd.ExcelWriter(f'./results/final_performance.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
          pd.DataFrame(
              val_performance, index=lstm_model.metrics_names
          ).assign(
              participant=PARTICIPANT_AS_TEST
          ).to_excel(writer, sheet_name="validation", startrow=writer.sheets['validation'].max_row, header=None)

          pd.DataFrame(
              before_fine_tuning_performance, index=lstm_model.metrics_names
          ).assign(
              participant=PARTICIPANT_AS_TEST
          ).to_excel(writer, sheet_name="before fine-tuning", startrow=writer.sheets['before fine-tuning'].max_row, header=None)

          pd.DataFrame(
              performance, index=lstm_model.metrics_names
          ).assign(
              participant=PARTICIPANT_AS_TEST
          ).to_excel(writer, sheet_name="after fine-tuning", startrow=writer.sheets['after fine-tuning'].max_row, header=None)
  else:
      with pd.ExcelWriter(f'./results/final_performance.xlsx', engine='openpyxl', mode='w') as writer:
          pd.DataFrame(
              val_performance, index=lstm_model.metrics_names
          ).assign(
              participant=PARTICIPANT_AS_TEST
          ).to_excel(writer, sheet_name="validation")

          pd.DataFrame(
              before_fine_tuning_performance, index=lstm_model.metrics_names
          ).assign(
              participant=PARTICIPANT_AS_TEST
          ).to_excel(writer, sheet_name="before fine-tuning")

          pd.DataFrame(
              performance, index=lstm_model.metrics_names
          ).assign(
              participant=PARTICIPANT_AS_TEST
          ).to_excel(writer, sheet_name="after fine-tuning")

  # %%
  # display(pd.DataFrame(val_performance, index=lstm_model.metrics_names))
  # display(pd.DataFrame(before_fine_tuning_performance, index=lstm_model.metrics_names))
  # display(pd.DataFrame(performance, index=lstm_model.metrics_names))

  # %% [markdown]
  # ## Residual Connections

  # %%
  # wide_window = WindowGenerator(
  #     train_df=train_df, val_df=val_df,
  #     fine_tuning_df=fine_tuning_df, test_df=test_df,
  #     input_width=24, label_width=24, shift=1,
  #     batch_size=BATCH_SIZE)
  # print(wide_window)
  # for example_inputs, example_labels in wide_window.train.take(1):
  #   print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  #   print(f'Labels shape (batch, time, features): {example_labels.shape}')
  # wide_window.plot(plot_col=TARGET_COLUMN)

  # %%
  # def compile_and_fit(model, window, patience=2):
  #   early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
  #                                                     patience=patience,
  #                                                     mode='min')

  #   model.compile(loss=tf.keras.losses.MeanSquaredError(),
  #                 optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
  #                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
  #   # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
  #   #               optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
  #   #               metrics=METRICS)
  #   # model.compile(loss=brier_score_loss,
  #   #               optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
  #   #               metrics=METRICS)

  #   # history = model.fit(window.train, epochs=MAX_EPOCHS,
  #   #                     validation_data=window.val,
  #   #                     callbacks=[early_stopping])
  #   history = model.fit(window.train, epochs=MAX_EPOCHS,
  #                       validation_data=window.val,
  #                       callbacks=[early_stopping])

  # %%
  # class ResidualWrapper(tf.keras.Model):
  #   def __init__(self, model):
  #     super().__init__()
  #     self.model = model

  #   def call(self, inputs, *args, **kwargs):
  #     delta = self.model(inputs, *args, **kwargs)

  #     # The prediction for each time step is the input
  #     # from the previous time step plus the delta
  #     # calculated by the model.
  #     return inputs + delta

  # %%
  # K.clear_session()
  # residual_lstm = ResidualWrapper(
  #     tf.keras.Sequential([
  #         tf.keras.layers.LSTM(32, return_sequences=True),
  #         tf.keras.layers.Dense(
  #             units=len(features),
  #             # # The predicted deltas should start small.
  #             # # Therefore, initialize the output layer with zeros.
  #             kernel_initializer=tf.initializers.zeros())
  #     ])
  # )

  # history = compile_and_fit(residual_lstm, wide_window)

  # %%
  # val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
  # performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
  # for name, value in zip(residual_lstm.metrics_names, performance['Residual LSTM']):
  #   print(name, ': ', value)

  # %%
  # wide_window.plot(model=residual_lstm)

  # %%
  # x = np.arange(len(performance))
  # width = 0.3
  # metric_name = 'mean_absolute_error'
  # metric_index = lstm_model.metrics_names.index('mean_absolute_error')
  # val_mae = [v[metric_index] for v in val_performance.values()]
  # test_mae = [v[metric_index] for v in performance.values()]

  # plt.ylabel('mean_absolute_error [T (degC), normalized]')
  # plt.bar(x - 0.17, val_mae, width, label='Validation')
  # plt.bar(x + 0.17, test_mae, width, label='Test')
  # plt.xticks(ticks=x, labels=performance.keys(),
  #            rotation=45)
  # _ = plt.legend()

  # %% [markdown]
  # # Plot loss

  # %%
  # metrics = ['loss'] #, 'balanced_accuracy', 'auc', 'prc', 'precision', 'recall']
  # plt.figure(figsize=(25, 10))
  # for n, metric in enumerate(metrics):
  #     name = metric.replace("_"," ").capitalize()
  #     plt.subplot(3,3,n+1)
  #     plt.plot(history.epoch, history.history[metric], label='Train')
  #     plt.plot(history.epoch, history.history['val_'+metric], label='Validation')
  #     plt.xlabel('Epoch')
  #     plt.ylabel(name)
  #     if metric == 'loss':
  #       plt.ylim([0, plt.ylim()[1]])
  #     # # elif metric == 'auc':
  #     # #   plt.ylim([0.8,1])
  #     # else:
  #     #   plt.ylim([0,1])
  #     plt.legend()
  # plt.show()

  # %% [markdown]
  # # Predict using Model

  # %%
  # def data_loader(new_df):
  #     return np.array(new_df, dtype=np.float32)[np.newaxis, ...]
  # linear.predict(data_loader(train_df.loc[:, train_df.columns != 'T (degC) classification'].head(4))) # [0,:4]

  # %% [markdown]
  # # Export Model

  # %%
  # base_path = "models"
  # model_version = "1"
  # model_name = "multi_conv_model"
  # model_path = os.path.join(base_path, model_name, model_version)
  # tf.saved_model.save(multi_conv_model, model_path)

  # %%
  # saved_model = tf.saved_model.load(model_path)

  # %%
  # saved_model(data_loader(test_df.iloc[0:36, :]), training=False)[:,:,14]

  # %%
  # saved_model(data_loader(test_df.iloc[0:24, :]).tolist(), training=False)[:,:,14]

  # %%
  # print(data_loader(test_df.iloc[0:36, :]).tolist())

  # %%
  # data_loader(test_df.iloc[0:36, :]).shape


