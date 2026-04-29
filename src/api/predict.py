# src/api/predict.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import MODELS_DIR, SCALER_SAVE_PATH, TARGET_COLUMN
from src.models.retention_strategy import assign_risk_segment, get_retention_action

# ── Exact column order model was trained on ─────────────
FEATURE_COLUMNS = [
    "age", "gender", "tenure_months", "contract_type",
    "payment_method", "internet_service", "monthly_charges",
    "num_products", "num_support_tickets", "login_frequency",
    "avg_session_duration", "days_since_last_login",
    "total_charges", "clv", "recency_score", "frequency_score",
    "monetary_score", "rfm_score", "engagement_rate",
    "is_inactive", "high_support_tickets", "session_quality",
    "charge_per_product", "is_new_customer",
    "churn_risk_score", "clv_segment"
]


def load_model(model_name: str = "xgboost"):
    """Load saved model from disk"""
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if os.path.exists(path):
        print(f"✅ Model loaded: {model_name}")
        return joblib.load(path)
    print(f"❌ Model not found: {path}")
    return None


def prepare_input(customer_data: dict) -> pd.DataFrame:
    """
    Convert raw customer data into numbers the AI understands.
    Like translating English into AI language!
    """
    df = pd.DataFrame([customer_data])

    # ── Drop customer_id ────────────────────────────────
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # ── Encode text columns to numbers ──────────────────
    gender_map   = {"Male": 1, "Female": 0}
    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    payment_map  = {
        "Credit Card": 0, "Bank Transfer": 1,
        "Electronic Check": 2, "Mailed Check": 3
    }
    internet_map = {"DSL": 0, "Fiber Optic": 1, "No": 2}

    df["gender"]           = df["gender"].map(gender_map).fillna(0)
    df["contract_type"]    = df["contract_type"].map(contract_map).fillna(0)
    df["payment_method"]   = df["payment_method"].map(payment_map).fillna(0)
    df["internet_service"] = df["internet_service"].map(internet_map).fillna(0)

    # ── RFM Features ────────────────────────────────────
    df["recency_score"] = pd.cut(
        df["days_since_last_login"],
        bins=[0, 7, 30, 60, 90, 999],
        labels=[5, 4, 3, 2, 1]
    ).astype(float)

    df["frequency_score"] = pd.cut(
        df["login_frequency"],
        bins=[-1, 5, 10, 15, 20, 999],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)

    df["monetary_score"] = pd.cut(
        df["monthly_charges"],
        bins=[0, 30, 50, 70, 90, 999],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)

    df["rfm_score"] = (
        df["recency_score"] +
        df["frequency_score"] +
        df["monetary_score"]
    )

    # ── Engagement Features ──────────────────────────────
    df["engagement_rate"]      = df["login_frequency"] / (df["tenure_months"] + 1)
    df["is_inactive"]          = (df["days_since_last_login"] > 30).astype(int)
    df["high_support_tickets"] = (df["num_support_tickets"] > 5).astype(int)
    df["session_quality"]      = df["avg_session_duration"] * df["login_frequency"]

    # ── Risk Features ────────────────────────────────────
    df["charge_per_product"] = df["monthly_charges"] / df["num_products"].replace(0, 1)
    df["is_new_customer"]    = (df["tenure_months"] < 12).astype(int)
    df["churn_risk_score"]   = (
        df["is_inactive"] * 0.3 +
        df["high_support_tickets"] * 0.25 +
        df["is_new_customer"] * 0.2 +
        (1 / (df["rfm_score"] + 1)) * 0.25
    ).round(4)

    df["clv_segment"] = 2.0

    # ── Reorder to EXACT training column order ───────────
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df = df.fillna(0)

    return df


def predict_single_customer(customer_data: dict, model_name: str = "xgboost") -> dict:
    """
    Predict churn for ONE customer and return full action plan!
    """
    model = load_model(model_name)
    if model is None:
        return {"error": "Model not found"}

    # Prepare input with correct column order
    input_df   = prepare_input(customer_data)

    # Get churn probability
    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = int(churn_prob >= 0.5)

    # Get risk segment
    risk_segment = assign_risk_segment(churn_prob)

    # Get retention actions
    action_row                      = input_df.copy()
    action_row["churn_probability"] = churn_prob
    action_row["risk_segment"]      = risk_segment
    actions = get_retention_action(action_row.iloc[0])

    # Build final result
    result = {
        "customer_id"           : customer_data.get("customer_id", "UNKNOWN"),
        "churn_prediction"      : "Will Churn ⚠️" if churn_pred else "Will Stay ✅",
        "churn_probability"     : round(float(churn_prob) * 100, 2),
        "risk_segment"          : risk_segment,
        "priority"              : actions["priority"],
        "recommended_actions"   : actions["recommended_actions"],
        "best_offer"            : actions["best_offer"],
        "estimated_annual_value": actions["estimated_annual_value"]
    }

    return result


def predict_batch(customers: list, model_name: str = "xgboost") -> pd.DataFrame:
    """
    Predict churn for many customers at once!
    """
    print(f"\n🔄 Processing {len(customers)} customers...")
    results    = [predict_single_customer(c, model_name) for c in customers]
    results_df = pd.DataFrame(results)
    print("✅ Batch prediction complete!")
    return results_df


def print_prediction_result(result: dict) -> None:
    """Print a clean, readable prediction result"""
    print("\n" + "="*55)
    print("🎯 CHURN PREDICTION RESULT")
    print("="*55)
    print(f"👤 Customer ID       : {result['customer_id']}")
    print(f"🔮 Prediction        : {result['churn_prediction']}")
    print(f"📊 Churn Probability : {result['churn_probability']}%")
    print(f"🚦 Risk Segment      : {result['risk_segment']}")
    print(f"⚡ Priority          : {result['priority']}")
    print(f"💰 Best Offer        : {result['best_offer']}")
    print(f"💵 Est. Annual Value : ${result['estimated_annual_value']:,.2f}")
    print("\n📋 Recommended Actions:")
    for action in result["recommended_actions"].split(" | "):
        print(f"   {action}")
    print("="*55)


if __name__ == "__main__":

    # ── Test Customer 1 — High Risk ─────────────────────
    customer_1 = {
        "customer_id"          : "CUST_TEST_001",
        "age"                  : 35,
        "gender"               : "Male",
        "tenure_months"        : 3,
        "contract_type"        : "Month-to-Month",
        "payment_method"       : "Electronic Check",
        "internet_service"     : "Fiber Optic",
        "monthly_charges"      : 95.0,
        "total_charges"        : 285.0,
        "num_products"         : 1,
        "num_support_tickets"  : 8,
        "login_frequency"      : 2,
        "avg_session_duration" : 5.0,
        "days_since_last_login": 75,
        "clv"                  : 1200.0
    }

    # ── Test Customer 2 — Low Risk ──────────────────────
    customer_2 = {
        "customer_id"          : "CUST_TEST_002",
        "age"                  : 45,
        "gender"               : "Female",
        "tenure_months"        : 48,
        "contract_type"        : "Two Year",
        "payment_method"       : "Credit Card",
        "internet_service"     : "DSL",
        "monthly_charges"      : 45.0,
        "total_charges"        : 2160.0,
        "num_products"         : 4,
        "num_support_tickets"  : 1,
        "login_frequency"      : 25,
        "avg_session_duration" : 45.0,
        "days_since_last_login": 2,
        "clv"                  : 5400.0
    }

    # ── Single Predictions ──────────────────────────────
    print("\n🧪 TESTING SINGLE CUSTOMER PREDICTIONS\n")

    result1 = predict_single_customer(customer_1)
    print_prediction_result(result1)

    result2 = predict_single_customer(customer_2)
    print_prediction_result(result2)

    # ── Batch Prediction ────────────────────────────────
    print("\n🧪 TESTING BATCH PREDICTION")
    batch_results = predict_batch([customer_1, customer_2])
    print(batch_results[[
        "customer_id", "churn_prediction",
        "churn_probability", "risk_segment"
    ]].to_string(index=False))