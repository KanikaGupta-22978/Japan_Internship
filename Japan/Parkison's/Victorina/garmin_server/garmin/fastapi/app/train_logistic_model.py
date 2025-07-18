# train_logistic_model.py

# ----------------------------- #
#       IMPORT LIBRARIES       #
# ----------------------------- #
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# ----------------------------- #
#        TRAINING LOGIC         #
# ----------------------------- #

def train_model(
    file_path=None,
    model_output_path=r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Parkison's\Victorina\garmin_server\garmin\fastapi\app\logreg_pipeline.pkl"
):
    if file_path is None:
        file_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Parkison's\Victorina\Old_dataset\Garmin Paper 2021 Supplementary Materials\Garmin論文2021 補足資料\For Submission\data\combined_data\combined_data_participant1_15min.xlsx"

    # Load Excel file
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Raw 'wearing_off' distribution:")
    print(df['wearing_off'].value_counts(dropna=False))

    # Drop rows where target is missing
    df = df.dropna(subset=['wearing_off'])

    # Drop object/datetime columns
    cols_to_exclude = ['timestamp', 'wearing_off_start', 'wearing_off_end']
    df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])

    # Fill numeric NaNs
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print("\nFinal class distribution:")
    print(df['wearing_off'].value_counts())

    # Split features/target
    X = df.drop(columns=['wearing_off'])
    y = df['wearing_off']

    if y.nunique() < 2:
        raise ValueError("❌ Need at least 2 classes in target variable 'y'. Found only one class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTrain class distribution:")
    print(y_train.value_counts())

    scaler = StandardScaler()
    sampler = RandomUnderSampler(random_state=42)
    model = LogisticRegression(C=1.3369, random_state=42, max_iter=1000)

    pipeline = ImbPipeline(steps=[
        ('scaler', scaler),
        ('undersampler', sampler),
        ('classifier', model)
    ])

    try:
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        print(f"\n⚠️ Undersampling failed: {e}")
        print("Retrying without undersampling...")
        pipeline = ImbPipeline([
            ('scaler', scaler),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"\n✅ Model saved to {model_output_path}")

    return pipeline  # Optional: useful for testing

# ----------------------------- #
#   Run directly for testing    #
# ----------------------------- #

if __name__ == "__main__":
    train_model()
