# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# %%
# ✅ Configuration
custom_minutes = 60  # <-- Customize prediction horizon (in minutes)
data_path = r"c:\Users\Shared-PC\Desktop\For Submission\For Submission\data\combined_data\combined_data_participant2_15s.xlsx"

columns = ['timestamp', 'heart_rate', 'steps', 'stress_score',
           'awake', 'deep', 'light', 'rem', 'nonrem_total', 'total',
           'nonrem_percentage', 'sleep_efficiency',
           'time_from_last_drug_taken', 'wearing_off']

# %%
# ✅ Load and process data
data = pd.read_excel(data_path, usecols=columns, index_col="timestamp", engine='openpyxl')
data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Drop rows where target is missing
data = data.dropna(subset=["wearing_off"])
data = data.fillna(0)

# Estimate step duration (assumes regular interval)
median_step = data.index.to_series().diff().median()

# Calculate steps ahead for label shift
predict_ahead = pd.Timedelta(minutes=custom_minutes)
steps_ahead = int(predict_ahead / median_step)

# Shift target to future
data['wearing_off_future'] = data['wearing_off'].shift(-steps_ahead)

# Drop rows where shifted label is NaN
data = data.dropna(subset=['wearing_off_future'])

# Print target label distribution
print(f"Target counts ({custom_minutes}-minute ahead prediction):")
print((data['wearing_off_future'] == 0).sum())
print((data['wearing_off_future'] == 1).sum())

# %%
# ✅ Feature and label split
X = data.drop(columns=['wearing_off', 'wearing_off_future'])  # All features except target
y = data['wearing_off_future'].astype(int)

# %%
# ✅ Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# %%
# ✅ Build MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %%
# ✅ Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)

# %%
# ✅ Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training vs Validation Loss (MLP, {custom_minutes}min ahead)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ✅ Evaluate on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nClassification Report (MLP, {custom_minutes}-minute ahead):")
print(pd.DataFrame(report).transpose())

# %%
# ✅ Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Wearing-Off (0)", "Wearing-Off (1)"],
            yticklabels=["No Wearing-Off (0)", "Wearing-Off (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (MLP, {custom_minutes}-minute ahead)")
plt.tight_layout()
plt.show()
