# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import seaborn as sns

# %%
# ✅ Load the final merged data
data_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined\final_merged_combined.csv"
all_data = pd.read_csv(data_path)

# Optional cleanup
all_data = all_data.dropna(subset=["bad_qol"])  # drop rows with no label
all_data = all_data.fillna(0)                   # fill rest with 0

print(all_data.head())
print("All")
print((all_data['bad_qol'] == 0).sum())
print((all_data['bad_qol'] == 1).sum())


# %%
# 必要な特徴量の選択
X = all_data[["heart_rate", "steps", "stress_score", "awake", "deep", "light", "rem",
              "nonrem_total", "total", "nonrem_percentage", "sleep_efficiency"]]
y = all_data["bad_qol"]
print(X.shape)
print(y.shape)

# %%
# データの標準化
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# %%
# PCAの実行
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 結果のプロット
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.title('2D plot with PCA')
plt.colorbar(scatter, label='bad_qol')
plt.show()

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# %%
# GridSearchCVで最適なハイパーパラメータを探索
def param():
    return {
        'C': [1, 10, 100],
        'kernel': ['rbf'],
        'gamma': [1, 0.1, 0.01]
    }

# %%
# モデルの学習
model = GridSearchCV(SVC(class_weight='balanced'), param(), cv=5, verbose=1)
model.fit(X_train, y_train)

# %%
# 最高性能のモデルを取得
best = model.best_estimator_
print(f"Best Parameters: {best}")

# %%
# テストデータでの評価
y_pred = best.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:\n", pd.DataFrame(report).transpose())

# %%
# Permutation Importance
perm_importance = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=0)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.xlabel("Permutation Importance (Decrease in Accuracy)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature Importance in SVM (Permutation Importance)", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# %%
# 混同行列の可視化
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["on(0)", "off(1)"],
            yticklabels=["on(0)", "off(1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (SVM)")
plt.show()
