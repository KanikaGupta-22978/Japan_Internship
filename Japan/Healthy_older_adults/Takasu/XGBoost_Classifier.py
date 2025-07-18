# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# %%
# ✅ Load the final merged data
data_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined\final_merged_combined.csv"
all_data = pd.read_csv(data_path)

# Clean missing
all_data = all_data.dropna(subset=["bad_qol"])
all_data = all_data.fillna(0)

print(all_data.head())
print("All")
print((all_data['bad_qol'] == 0).sum())
print((all_data['bad_qol'] == 1).sum())

# %%
# Features and target
X = all_data[["heart_rate", "steps", "stress_score", "awake", "deep", "light", "rem",
              "nonrem_total", "total", "nonrem_percentage", "sleep_efficiency"]]
y = all_data["bad_qol"]
print(X.shape)
print(y.shape)

# %%
# Standardize
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# %%
# PCA for visualization
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0, stratify=y)

# %%
# Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                              random_state=42)

xgb_model.fit(X_train, y_train)

# %%
# Predict and evaluate
y_pred = xgb_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report (XGBoost):\n", pd.DataFrame(report).transpose())

# %%
# Feature importance from XGBoost
importance = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="rocket")
plt.xlabel("Feature Importance (Gain)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature Importance in XGBoost", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["on(0)", "off(1)"],
            yticklabels=["on(0)", "off(1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (XGBoost)")
plt.tight_layout()
plt.show()
