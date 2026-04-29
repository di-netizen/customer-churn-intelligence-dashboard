# src/models/train_model.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import (
    RANDOM_STATE, MODELS_DIR, PROCESSED_DATA_FILE, TARGET_COLUMN
)


def apply_smote(X_train, y_train):
    """
    SMOTE = Smart data balancer!
    If we have 800 loyal customers but only 200 churners,
    AI will be lazy and just guess 'loyal' every time.
    SMOTE creates fake churner examples to balance it out!
    """
    print("⚖️  Applying SMOTE to balance classes...")

    # Fill any empty boxes with 0 before SMOTE
    X_train = X_train.fillna(0)

    sm = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    print(f"   Before SMOTE: {dict(y_train.value_counts())}")
    print(f"   After SMOTE : {dict(pd.Series(y_resampled).value_counts())}")
    return X_resampled, y_resampled


def get_models() -> dict:
    """
    Our 4 contestants for the model competition! 🏆
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=-1
        )
    }


def train_all_models(X_train, y_train) -> dict:
    """
    Train ALL 4 models one by one!
    Like sending 4 students to study the same textbook.
    """
    print("\n" + "="*50)
    print("🤖 TRAINING ALL MODELS")
    print("="*50)

    # Balance the data first
    X_balanced, y_balanced = apply_smote(X_train, y_train)

    models = get_models()
    trained_models = {}

    for name, model in models.items():
        print(f"\n🔄 Training: {name}...")
        model.fit(X_balanced, y_balanced)
        trained_models[name] = model
        print(f"   ✅ {name} trained!")

    return trained_models


def save_models(trained_models: dict) -> None:
    """
    Save all trained models to disk.
    Like saving your game progress! 💾
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, path)
        print(f"   💾 Saved: {filename}")


def load_best_model():
    """Load the best model (XGBoost by default)"""
    path = os.path.join(MODELS_DIR, "xgboost.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print("❌ No saved model found. Please train first!")
        return None


if __name__ == "__main__":
    from src.pipeline.data_loader import load_data
    from src.pipeline.preprocessor import preprocess_pipeline
    from src.features.feature_engineering import run_feature_engineering
    from sklearn.model_selection import train_test_split
    from src.config import TEST_SIZE

    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test = preprocess_pipeline(df)

    # Step 3: Feature engineering on processed data
    processed_df = pd.read_csv(PROCESSED_DATA_FILE)
    enriched_df  = run_feature_engineering(processed_df)

    # Step 4: Re-split enriched data
    drop_cols    = [TARGET_COLUMN, "customer_id"]
    feature_cols = [c for c in enriched_df.columns if c not in drop_cols]

    X = enriched_df[feature_cols].fillna(0)
    y = enriched_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Step 5: Train all models
    trained_models = train_all_models(X_train, y_train)

    # Step 6: Save all models
    print("\n💾 Saving all models...")
    save_models(trained_models)

    print("\n🎉 ALL MODELS TRAINED AND SAVED!")
    print(f"   Check your /models folder!")