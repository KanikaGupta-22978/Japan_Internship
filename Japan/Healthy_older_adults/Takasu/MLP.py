# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# %%
# ✅ Load and clean the data
data_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined\final_merged_combined.csv"
all_data = pd.read_csv(data_path)

all_data = all_data.dropna(subset=["bad_qol"])
all_data = all_data.fillna(0)

print("All")
print((all_data['bad_qol'] == 0).sum())
print((all_data['bad_qol'] == 1).sum())

# %%
# Feature selection
X = all_data[["heart_rate", "steps", "stress_score", "awake", "deep", "light", "rem",
              "nonrem_total", "total", "nonrem_percentage", "sleep_efficiency"]]
y = all_data["bad_qol"]

# %%
# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.title('2D plot with PCA')
plt.colorbar(scatter, label='bad_qol')
plt.show()

# %%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# %%
# Build MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # sigmoid for binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# %%
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)

# %%
# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (MLP)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Evaluate on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# Classification report and confusion matrix
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report (MLP):\n", pd.DataFrame(report).transpose())

# %%
# Confusion Matrix Visualization
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["on(0)", "off(1)"],
            yticklabels=["on(0)", "off(1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (MLP)")
plt.tight_layout()
plt.show()
