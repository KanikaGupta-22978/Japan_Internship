from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(
    excel_data_path=None,
    model_output_path = r"C:\Users\Shared-PC\Desktop\garmin\garmin\fastapi\app\models\lstm_model.keras",
    minutes_ahead=60,
    interval_minutes=15,
    sequence_length=8
):
    if excel_data_path is None:
        excel_data_path = r'c:\Users\Shared-PC\Desktop\For Submission\For Submission\data\combined_data\combined_data_participant1_15min.xlsx'

    df = pd.read_excel(excel_data_path, engine='openpyxl')

    print("Raw 'wearing_off' distribution:")
    print(df['wearing_off'].value_counts(dropna=False))

    shift_steps = minutes_ahead // interval_minutes
    df['wearing_off'] = df['wearing_off'].shift(-shift_steps)
    df = df.dropna(subset=['wearing_off'])

    df = df.fillna(0)
    df = df.drop(columns=[col for col in ['timestamp', 'wearing_off_start', 'wearing_off_end'] if col in df.columns])

    feature_cols = [col for col in df.columns if col != 'wearing_off']
    X = df[feature_cols].values
    y = df['wearing_off'].astype(int).values

    if len(np.unique(y)) < 2:
        raise ValueError("❌ Need at least 2 classes in target variable 'y'. Found only one class.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
    print(f"Sequence shape: {X_seq.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
    )

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and scaler
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    joblib.dump(scaler, model_output_path.replace('.keras', '_scaler.pkl'))

    print(f"\n✅ LSTM model saved to {model_output_path}")
    return model, scaler


if __name__ == "__main__":
    train_lstm_model()
