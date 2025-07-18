# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# %%
# ✅ Configuration
custom_minutes = 60  # <- prediction horizon
sequence_length = 8  # <- input window size (e.g., 2 hours if 15min freq)
data_path = r"c:\Users\Shared-PC\Desktop\For Submission\For Submission\data\combined_data\combined_data_participant2_15s.xlsx"

columns = ['timestamp', 'heart_rate', 'steps', 'stress_score',
           'awake', 'deep', 'light', 'rem', 'nonrem_total', 'total',
           'nonrem_percentage', 'sleep_efficiency',
           'time_from_last_drug_taken', 'wearing_off']

# %%
# ✅ Load and prepare data
data = pd.read_excel(data_path, usecols=columns, engine='openpyxl')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values("timestamp").set_index("timestamp")

# Drop missing target rows
data = data.dropna(subset=["wearing_off"])
data = data.fillna(0)

# Estimate step size and compute steps ahead
median_step = data.index.to_series().diff().median()
predict_ahead = pd.Timedelta(minutes=custom_minutes)
steps_ahead = int(predict_ahead / median_step)

# Shift label to the future
data['wearing_off_future'] = data['wearing_off'].shift(-steps_ahead)
data = data.dropna(subset=['wearing_off_future'])

# Print label distribution
print(f"Target counts ({custom_minutes}-minute ahead prediction):")
print((data['wearing_off_future'] == 0).sum())
print((data['wearing_off_future'] == 1).sum())

# %%
# ✅ Features and labels
feature_cols = ['heart_rate', 'steps', 'stress_score',
                'awake', 'deep', 'light', 'rem', 'nonrem_total', 'total',
                'nonrem_percentage', 'sleep_efficiency', 'time_from_last_drug_taken']
X = data[feature_cols].values
y = data['wearing_off_future'].astype(int).values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# ✅ Create 3D sequences for CNN input
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])  # Label at the end of window
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
print("X_seq shape:", X_seq.shape)  # (samples, timesteps, features)

# %%
# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
)

# %%
# ✅ Build 1D CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %%
# ✅ Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)

# %%
# ✅ Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"CNN1D Training vs Validation Loss ({custom_minutes}-minute ahead)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ✅ Evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nClassification Report (CNN1D, {custom_minutes}-minute ahead):")
print(pd.DataFrame(report).transpose())

# %%
# ✅ Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Wearing-Off (0)", "Wearing-Off (1)"],
            yticklabels=["No Wearing-Off (0)", "Wearing-Off (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (CNN1D, {custom_minutes}-minute ahead)")
plt.tight_layout()
plt.show()
