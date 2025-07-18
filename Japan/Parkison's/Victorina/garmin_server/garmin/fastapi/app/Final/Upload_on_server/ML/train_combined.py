# --- Modified train_logistic_model.py ---

from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import os
import joblib


def train_model(
    excel_data_path=None,
    model_output_path=f"./app/models/gb_pipeline_future_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl",
    minutes_ahead=30,
    interval_minutes=15
):
    if excel_data_path is None:
        excel_data_path = "./app/combined_data_participant1_15min.xlsx"

    df = pd.read_excel(excel_data_path, engine='openpyxl')
    print("Raw 'wearing_off' distribution:")
    print(df['wearing_off'].value_counts(dropna=False))

    # Shift wearing_off to predict the future state
    shift_steps = minutes_ahead // interval_minutes
    df['wearing_off'] = df['wearing_off'].shift(-shift_steps)
    df = df.dropna(subset=['wearing_off'])

    df = df.drop(columns=[col for col in ['timestamp', 'wearing_off_start', 'wearing_off_end'] if col in df.columns])

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print("\nFinal class distribution:")
    print(df['wearing_off'].value_counts())

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
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('undersampler', sampler),
        ('classifier', model)
    ])

    try:
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        print(f"\n⚠️ Undersampling failed: {e}")
        print("Retrying without undersampling...")
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"\n✅ Future prediction model saved to {model_output_path}")

    return pipeline


if __name__ == "__main__":
    train_model()
