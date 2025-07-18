# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# %%
# Load and clean data
data_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined\final_merged_combined.csv"
all_data = pd.read_csv(data_path)
all_data = all_data.dropna(subset=["bad_qol"])
all_data = all_data.fillna(0)

# %%
# Feature columns
feature_cols = ["heart_rate", "steps", "stress_score", "awake", "deep", "light", "rem",
                "nonrem_total", "total", "nonrem_percentage", "sleep_efficiency"]
X = all_data[feature_cols].values
y = all_data["bad_qol"].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Create sequences
def create_sequences(X, y, seq_length=8):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y, seq_length=8)
print("CNN input shape:", X_seq.shape)

# %%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq)

# %%
# 1D CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# %%
# Train
history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                    validation_split=0.2, verbose=1)

# %%
# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('CNN Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Evaluation
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# Classification report and confusion matrix
print("Classification Report (1D CNN):")
print(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["on(0)", "off(1)"],
            yticklabels=["on(0)", "off(1)"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (1D CNN)")
plt.tight_layout()
plt.show()
