# src/models/evaluate_model.py

import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, TARGET_COLUMN


def evaluate_single_model(name, model, X_test, y_test) -> dict:
    """
    Test ONE model — like grading one student's exam!
    """
    X_test = X_test.fillna(0)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall"   : round(recall_score(y_test, y_pred), 4),
        "F1 Score" : round(f1_score(y_test, y_pred), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_pred_prob), 4),
    }

    return results, y_pred, y_pred_prob


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Test ALL models and make a report card!
    """
    print("\n" + "="*50)
    print("📊 EVALUATING ALL MODELS")
    print("="*50)

    all_results = []

    for name, model in trained_models.items():
        results, y_pred, y_prob = evaluate_single_model(
            name, model, X_test, y_test
        )
        all_results.append(results)
        print(f"\n✅ {name}")
        print(f"   Accuracy : {results['Accuracy']}")
        print(f"   F1 Score : {results['F1 Score']}")
        print(f"   ROC-AUC  : {results['ROC-AUC']}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("ROC-AUC", ascending=False)

    print("\n" + "="*50)
    print("🏆 FINAL LEADERBOARD")
    print("="*50)
    print(results_df.to_string(index=False))

    return results_df


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    """
    Confusion Matrix — shows where the model got confused!
    Like showing which questions a student got wrong.
    """
    X_test  = X_test.fillna(0)
    y_pred  = model.predict(X_test)
    cm      = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    print(f"   📊 Saved confusion matrix: {path}")


def plot_roc_curves(trained_models: dict, X_test, y_test) -> None:
    """
    ROC Curve — shows how good each model is at separating churners!
    Higher curve = smarter model!
    """
    X_test = X_test.fillna(0)

    plt.figure(figsize=(8, 6))

    for name, model in trained_models.items():
        y_prob     = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc         = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend()
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "roc_curves_all_models.png")
    plt.savefig(path)
    plt.close()
    print(f"   📊 Saved ROC curves: {path}")


def plot_feature_importance(model, feature_names, model_name: str) -> None:
    """
    Which features does the model think are most important?
    Like asking — what's the biggest reason customers leave?
    """
    if not hasattr(model, "feature_importances_"):
        print(f"   ⚠️  {model_name} doesn't support feature importance.")
        return

    importance_df = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top 15 Feature Importances — {model_name}")
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    print(f"   📊 Saved feature importance: {path}")


def save_results(results_df: pd.DataFrame) -> None:
    """Save the leaderboard to a CSV file"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(path, index=False)
    print(f"\n   💾 Results saved to: {path}")


if __name__ == "__main__":
    from src.pipeline.data_loader import load_data
    from src.pipeline.preprocessor import preprocess_pipeline
    from src.features.feature_engineering import run_feature_engineering
    from src.models.train_model import train_all_models, save_models
    from sklearn.model_selection import train_test_split
    from src.config import TEST_SIZE, PROCESSED_DATA_FILE

    # Step 1-4: Full pipeline
    df           = load_data()
    preprocess_pipeline(df)
    processed_df = pd.read_csv(PROCESSED_DATA_FILE)
    enriched_df  = run_feature_engineering(processed_df)

    drop_cols    = [TARGET_COLUMN, "customer_id"]
    feature_cols = [c for c in enriched_df.columns if c not in drop_cols]

    X = enriched_df[feature_cols].fillna(0)
    y = enriched_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = TEST_SIZE,
        random_state= RANDOM_STATE,
        stratify    = y
    )

    # Step 5: Train
    trained_models = train_all_models(X_train, y_train)
    save_models(trained_models)

    # Step 6: Evaluate
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    save_results(results_df)

    # Step 7: Generate all charts
    print("\n📊 Generating charts...")
    plot_roc_curves(trained_models, X_test, y_test)

    for name, model in trained_models.items():
        plot_confusion_matrix(model, X_test, y_test, name)
        plot_feature_importance(model, feature_cols, name)

    print("\n🎉 EVALUATION COMPLETE!")
    print(f"   Check your /reports folder for all charts!")