# src/pipeline/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import (
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE, SCALER_SAVE_PATH, PROCESSED_DATA_FILE
)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values — like filling blank boxes in a form!"""
    print("🔧 Handling missing values...")

    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"   ✅ Missing values after fix: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows — no double counting customers!"""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"🔧 Removed {before - after} duplicate rows.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text columns to numbers.
    AI only understands numbers — like converting 'Male/Female' to '0/1'!
    """
    print("🔧 Encoding categorical columns...")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"   ✅ Encoded: {col}")

    return df


def scale_numerical(df: pd.DataFrame, fit: bool = True) -> tuple:
    """
    Scale numbers to same range.
    Like converting marks out of 100, 50, 1000 — all to same scale!
    """
    print("🔧 Scaling numerical columns...")

    scaler = StandardScaler()
    available_cols = [col for col in NUMERICAL_COLS if col in df.columns]

    if fit:
        df[available_cols] = scaler.fit_transform(df[available_cols])
        os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"   ✅ Scaler saved to: {SCALER_SAVE_PATH}")
    else:
        scaler = joblib.load(SCALER_SAVE_PATH)
        df[available_cols] = scaler.transform(df[available_cols])

    return df, scaler


def split_data(df: pd.DataFrame) -> tuple:
    """
    Split data into training and testing sets.
    Like keeping 80% data to STUDY and 20% for the EXAM!
    """
    print("🔧 Splitting data into train/test sets...")

    drop_cols = [TARGET_COLUMN, "customer_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"   ✅ Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df: pd.DataFrame) -> tuple:
    """
    Runs ALL cleaning steps one by one — the full car wash! 🚗
    """
    print("\n" + "="*50)
    print("🧹 STARTING PREPROCESSING PIPELINE")
    print("="*50)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categoricals(df)
    df, scaler = scale_numerical(df, fit=True)

    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"\n✅ Processed data saved to: {PROCESSED_DATA_FILE}")

    X_train, X_test, y_train, y_test = split_data(df)

    print("\n✅ PREPROCESSING COMPLETE!")
    print("="*50 + "\n")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.pipeline.data_loader import load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_pipeline(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")