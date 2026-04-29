# src/models/explainability.py

import pandas as pd
import numpy as np
import shap
import joblib
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_FILE, TARGET_COLUMN


def load_model(model_name: str = "xgboost"):
    """Load a saved model from disk"""
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"✅ Loaded model: {model_name}")
        return model
    else:
        print(f"❌ Model not found: {path}")
        return None


def compute_shap_values(model, X_data: pd.DataFrame):
    """
    Compute SHAP values — asking AI to explain its decisions!
    Like asking a doctor WHY they gave a diagnosis.
    """
    print("\n🔍 Computing SHAP values (this takes a moment)...")

    X_clean = X_data.fillna(0)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)

    print("   ✅ SHAP values computed!")
    return explainer, shap_values, X_clean


def plot_shap_summary(shap_values, X_data: pd.DataFrame, model_name: str) -> None:
    """
    Summary plot — shows which features matter MOST overall!
    Like a top 10 list of reasons customers leave.
    """
    print("   📊 Creating SHAP summary plot...")

    plt.figure()
    shap.summary_plot(
        shap_values, X_data,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title(f"Top Churn Drivers — {model_name}")
    plt.tight_layout()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"shap_summary_{model_name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_shap_beeswarm(shap_values, X_data: pd.DataFrame, model_name: str) -> None:
    """
    Beeswarm plot — shows HOW each feature pushes customers toward churn!
    Red = pushes toward churn, Blue = pushes away from churn.
    """
    print("   📊 Creating SHAP beeswarm plot...")

    plt.figure()
    shap.summary_plot(
        shap_values, X_data,
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Beeswarm — {model_name}")
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, f"shap_beeswarm_{model_name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"   ✅ Saved: {path}")


def plot_shap_single_customer(
    explainer, shap_values,
    X_data: pd.DataFrame,
    customer_index: int,
    model_name: str
) -> None:
    """
    Waterfall plot — explains ONE specific customer's churn prediction!
    Like a personalised report card for that customer.
    """
    print(f"   📊 Creating waterfall plot for customer #{customer_index}...")

    shap_explanation = shap.Explanation(
        values    = shap_values[customer_index],
        base_values = explainer.expected_value,
        data      = X_data.iloc[customer_index].values,
        feature_names = X_data.columns.tolist()
    )

    plt.figure()
    shap.waterfall_plot(shap_explanation, show=False)
    plt.title(f"Customer #{customer_index} — Why Churn?")
    plt.tight_layout()

    path = os.path.join(
        REPORTS_DIR,
        f"shap_customer_{customer_index}_{model_name}.png"
    )
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    print(f"   ✅ Saved: {path}")


def get_top_churn_drivers(shap_values, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """
    Get a simple table of top reasons customers churn.
    Easy to read for business teams!
    """
    mean_shap = np.abs(shap_values).mean(axis=0)

    drivers_df = pd.DataFrame({
        "Feature"         : feature_names,
        "SHAP Importance" : mean_shap
    }).sort_values("SHAP Importance", ascending=False).head(top_n)

    drivers_df["Rank"] = range(1, len(drivers_df) + 1)
    drivers_df = drivers_df[["Rank", "Feature", "SHAP Importance"]]

    return drivers_df


def run_explainability(X_test: pd.DataFrame, model_name: str = "xgboost") -> None:
    """
    Run the full explainability pipeline!
    """
    print("\n" + "="*50)
    print("🔍 STARTING SHAP EXPLAINABILITY")
    print("="*50)

    # Load model
    model = load_model(model_name)
    if model is None:
        return

    # Compute SHAP
    explainer, shap_values, X_clean = compute_shap_values(model, X_test)

    # Generate all plots
    print("\n📊 Generating SHAP visualizations...")
    plot_shap_summary(shap_values, X_clean, model_name)
    plot_shap_beeswarm(shap_values, X_clean, model_name)
    plot_shap_single_customer(explainer, shap_values, X_clean, 0, model_name)
    plot_shap_single_customer(explainer, shap_values, X_clean, 1, model_name)

    # Print top drivers
    print("\n" + "="*50)
    print("🎯 TOP 10 CHURN DRIVERS")
    print("="*50)
    drivers = get_top_churn_drivers(shap_values, X_clean.columns.tolist())
    print(drivers.to_string(index=False))

    # Save drivers table
    path = os.path.join(REPORTS_DIR, "top_churn_drivers.csv")
    drivers.to_csv(path, index=False)
    print(f"\n   💾 Saved churn drivers to: {path}")

    print("\n✅ EXPLAINABILITY COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    from src.pipeline.data_loader import load_data
    from src.pipeline.preprocessor import preprocess_pipeline
    from src.features.feature_engineering import run_feature_engineering
    from src.models.train_model import train_all_models, save_models
    from sklearn.model_selection import train_test_split
    from src.config import TEST_SIZE, RANDOM_STATE

    # Full pipeline
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
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y
    )

    # Train and save models
    trained_models = train_all_models(X_train, y_train)
    save_models(trained_models)

    # Run explainability on XGBoost
    run_explainability(X_test, model_name="xgboost")