# src/models/retention_strategy.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_FILE, TARGET_COLUMN


def load_model(model_name: str = "xgboost"):
    """Load saved model"""
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    print(f"❌ Model not found: {path}")
    return None


def predict_churn_probability(model, X_data: pd.DataFrame) -> np.ndarray:
    """
    Get churn probability for every customer.
    Like getting a percentage chance they will leave — 0% to 100%!
    """
    X_clean = X_data.fillna(0)
    probs   = model.predict_proba(X_clean)[:, 1]
    return probs


def assign_risk_segment(probability: float) -> str:
    """
    Put each customer in a risk bucket!
    Like traffic lights — Green, Yellow, Orange, Red 🚦
    """
    if probability >= 0.75:
        return "🔴 Critical Risk"
    elif probability >= 0.50:
        return "🟠 High Risk"
    elif probability >= 0.25:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


def get_retention_action(row: pd.Series) -> dict:
    """
    Recommend the RIGHT action for each customer based on their profile.
    Like a personalised prescription for each patient!
    """
    actions     = []
    offer       = ""
    priority    = ""
    estimated_save = 0

    prob     = row["churn_probability"]
    segment  = row["risk_segment"]

    # ── Critical Risk Customers ─────────────────────────
    if "Critical" in segment:
        priority = "🚨 URGENT — Act within 24 hours"
        actions  = [
            "📞 Personal call from account manager",
            "💰 Offer 30% discount for 3 months",
            "🎁 Free upgrade to premium plan",
            "🔧 Dedicated support agent assigned"
        ]
        offer           = "30% discount + Free upgrade"
        estimated_save  = round(row.get("monthly_charges", 50) * 12 * 0.7, 2)

    # ── High Risk Customers ──────────────────────────────
    elif "High" in segment:
        priority = "⚠️  HIGH — Act within 3 days"
        actions  = [
            "📧 Personalised retention email",
            "💰 Offer 15% loyalty discount",
            "🎯 Targeted re-engagement campaign",
            "📊 Share personalised usage report"
        ]
        offer           = "15% loyalty discount"
        estimated_save  = round(row.get("monthly_charges", 50) * 12 * 0.85, 2)

    # ── Medium Risk Customers ────────────────────────────
    elif "Medium" in segment:
        priority = "📋 MEDIUM — Act within 1 week"
        actions  = [
            "📧 Send feature highlight newsletter",
            "🎮 Invite to product webinar",
            "⭐ Ask for feedback / NPS survey",
            "🎁 Offer loyalty rewards points"
        ]
        offer           = "Loyalty rewards + Webinar invite"
        estimated_save  = round(row.get("monthly_charges", 50) * 12 * 0.95, 2)

    # ── Low Risk Customers ───────────────────────────────
    else:
        priority = "✅ LOW — Routine engagement"
        actions  = [
            "📧 Monthly newsletter",
            "🎉 Celebrate customer anniversary",
            "⭐ Request product review",
            "🔼 Upsell to higher tier plan"
        ]
        offer           = "Upsell opportunity"
        estimated_save  = round(row.get("monthly_charges", 50) * 12, 2)

    return {
        "priority"      : priority,
        "recommended_actions": " | ".join(actions),
        "best_offer"    : offer,
        "estimated_annual_value": estimated_save
    }


def build_retention_plan(df: pd.DataFrame, model_name: str = "xgboost") -> pd.DataFrame:
    """
    Build a full retention plan for ALL customers!
    The final output business teams actually use. 📋
    """
    print("\n" + "="*50)
    print("🎯 BUILDING RETENTION STRATEGY")
    print("="*50)

    # Load model
    model = load_model(model_name)
    if model is None:
        return None

    # Get feature columns
    drop_cols    = [TARGET_COLUMN, "customer_id"] if TARGET_COLUMN in df.columns else ["customer_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Predict churn probability
    X = df[feature_cols].fillna(0)
    df["churn_probability"] = predict_churn_probability(model, X)
    df["churn_probability"] = df["churn_probability"].round(4)

    # Assign risk segments
    df["risk_segment"] = df["churn_probability"].apply(assign_risk_segment)

    # Get retention actions
    print("🔄 Generating personalised retention actions...")
    actions = df.apply(get_retention_action, axis=1, result_type="expand")
    df      = pd.concat([df, actions], axis=1)

    print("   ✅ Retention plan generated!")
    return df


def print_retention_summary(retention_df: pd.DataFrame) -> None:
    """
    Print a business-friendly summary!
    Like an executive report for the CEO.
    """
    print("\n" + "="*60)
    print("📊 RETENTION STRATEGY SUMMARY")
    print("="*60)

    total     = len(retention_df)
    segments  = retention_df["risk_segment"].value_counts()

    print(f"\n👥 Total Customers Analysed : {total}")
    print(f"\n🚦 Risk Breakdown:")
    for seg, count in segments.items():
        pct = round(count / total * 100, 1)
        print(f"   {seg:<25} : {count:>4} customers ({pct}%)")

    critical  = retention_df[retention_df["risk_segment"].str.contains("Critical")]
    high      = retention_df[retention_df["risk_segment"].str.contains("High")]
    at_risk   = pd.concat([critical, high])

    if "monthly_charges" in retention_df.columns:
        revenue_at_risk = at_risk["monthly_charges"].sum() * 12
        print(f"\n💸 Annual Revenue at Risk  : ${revenue_at_risk:,.2f}")
        print(f"🚨 Urgent Action Needed    : {len(critical)} customers")

    print("\n" + "="*60)


def save_retention_plan(retention_df: pd.DataFrame) -> None:
    """Save the full retention plan to CSV"""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Full plan
    full_path = os.path.join(REPORTS_DIR, "full_retention_plan.csv")
    retention_df.to_csv(full_path, index=False)
    print(f"\n💾 Full plan saved    : {full_path}")

    # Critical only
    critical_df   = retention_df[
        retention_df["risk_segment"].str.contains("Critical|High")
    ]
    critical_path = os.path.join(REPORTS_DIR, "urgent_action_list.csv")
    critical_df.to_csv(critical_path, index=False)
    print(f"💾 Urgent list saved  : {critical_path}")


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

    # Train models
    trained_models = train_all_models(X_train, y_train)
    save_models(trained_models)

    # Build retention plan on test data
    test_df              = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    retention_df = build_retention_plan(test_df, model_name="xgboost")

    # Print summary
    print_retention_summary(retention_df)

    # Save outputs
    save_retention_plan(retention_df)

    # Preview top 5 at-risk customers
    print("\n🔍 SAMPLE — Top 5 Critical Risk Customers:")
    critical = retention_df[
        retention_df["risk_segment"].str.contains("Critical")
    ][["churn_probability", "risk_segment", "best_offer", "priority"]].head(5)
    print(critical.to_string(index=False))