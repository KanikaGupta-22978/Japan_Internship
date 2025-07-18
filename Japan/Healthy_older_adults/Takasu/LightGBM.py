# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb

# %%
# ✅ Load and clean the dataset
data_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined\final_merged_combined.csv"
all_data = pd.read_csv(data_path)

all_data = all_data.dropna(subset=["bad_qol"])
all_data = all_data.fillna(0)

print("All")
print((all_data['bad_qol'] == 0).sum())
print((all_data['bad_qol'] == 1).sum())

# %%
# Select features and target
X = all_data[["heart_rate", "steps", "stress_score", "awake", "deep", "light", "rem",
              "nonrem_total", "total", "nonrem_percentage", "sleep_efficiency"]]
y = all_data["bad_qol"]

# %%
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
plt.tight_layout()
plt.show()

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.3, random_state=0)

# %%
# Train LightGBM classifier
lgb_model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
lgb_model.fit(X_train, y_train)

# %%
# Predict and evaluate
y_pred_lgb = lgb_model.predict(X_test)
report = classification_report(y_test, y_pred_lgb, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred_lgb)

print("Classification Report (LightGBM):\n", pd.DataFrame(report).transpose())

# %%
# Feature Importance
importance = lgb_model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_df, palette="flare")
plt.xlabel("Feature Importance (Split Count)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature Importance in LightGBM", fontsize=14)
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
plt.title("Confusion Matrix (LightGBM)")
plt.tight_layout()
plt.show()
