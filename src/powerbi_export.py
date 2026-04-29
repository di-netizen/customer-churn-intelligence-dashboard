# src/powerbi_export.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_FILE,
    TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
)


def load_enriched_data() -> pd.DataFrame:
    """Load the enriched customer data"""
    path = PROCESSED_DATA_FILE.replace(
        "customers_processed.csv",
        "customers_enriched.csv"
    )
    df = pd.read_csv(path)
    print(f"✅ Loaded enriched data: {df.shape}")
    return df


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add churn predictions to every customer.
    So Power BI can show predicted churn alongside actual!
    """
    print("🔄 Adding predictions...")

    model_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return df

    model      = joblib.load(model_path)
    drop_cols  = [TARGET_COLUMN, "customer_id"]
    feat_cols  = [c for c in df.columns if c not in drop_cols]

    X          = df[feat_cols].fillna(0)

    # Fix column order
    trained_features = model.get_booster().feature_names
    X = X.reindex(columns=trained_features, fill_value=0)

    df["churn_probability"]  = model.predict_proba(X)[:, 1].round(4)
    df["churn_predicted"]    = (df["churn_probability"] >= 0.5).astype(int)

    # Risk segment labels
    def risk_label(p):
        if p >= 0.75: return "Critical Risk"
        elif p >= 0.50: return "High Risk"
        elif p >= 0.25: return "Medium Risk"
        else: return "Low Risk"

    df["risk_segment"]       = df["churn_probability"].apply(risk_label)

    # Retention action
    def retention_action(row):
        seg = row["risk_segment"]
        if seg == "Critical Risk":
            return "30% Discount + Free Upgrade"
        elif seg == "High Risk":
            return "15% Loyalty Discount"
        elif seg == "Medium Risk":
            return "Loyalty Rewards + Webinar"
        else:
            return "Upsell Opportunity"

    df["recommended_action"] = df.apply(retention_action, axis=1)

    print("✅ Predictions added!")
    return df


def build_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main customer table for Power BI.
    One row per customer with all key info!
    """
    cols = [
        "customer_id", "age", "tenure_months",
        "monthly_charges", "total_charges", "clv",
        "num_products", "num_support_tickets",
        "login_frequency", "days_since_last_login",
        "churn", "churn_probability", "churn_predicted",
        "risk_segment", "recommended_action",
        "rfm_score", "churn_risk_score",
        "engagement_rate", "is_inactive"
    ]
    available = [c for c in cols if c in df.columns]
    return df[available]


def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table by risk segment.
    For the executive overview page in Power BI!
    """
    summary = df.groupby("risk_segment").agg(
        total_customers    = ("customer_id", "count"),
        avg_churn_prob     = ("churn_probability", "mean"),
        avg_monthly_charges= ("monthly_charges", "mean"),
        total_revenue      = ("monthly_charges", "sum"),
        avg_tenure         = ("tenure_months", "mean"),
        avg_clv            = ("clv", "mean")
    ).reset_index()

    summary["avg_churn_prob"]      = summary["avg_churn_prob"].round(3)
    summary["avg_monthly_charges"] = summary["avg_monthly_charges"].round(2)
    summary["total_revenue"]       = summary["total_revenue"].round(2)
    summary["avg_tenure"]          = summary["avg_tenure"].round(1)
    summary["avg_clv"]             = summary["avg_clv"].round(2)

    # Add risk order for sorting in Power BI
    risk_order = {
        "Critical Risk": 1,
        "High Risk"    : 2,
        "Medium Risk"  : 3,
        "Low Risk"     : 4
    }
    summary["risk_order"] = summary["risk_segment"].map(risk_order)
    summary = summary.sort_values("risk_order")

    return summary


def build_churn_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Top churn drivers table.
    For the feature importance page in Power BI!
    """
    path = os.path.join(REPORTS_DIR, "top_churn_drivers.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    # Fallback — correlation based
    num_df   = df.select_dtypes(include=[np.number])
    corr     = num_df.corr()["churn"].drop("churn").abs()
    drivers  = corr.sort_values(ascending=False).head(15).reset_index()
    drivers.columns = ["Feature", "SHAP Importance"]
    drivers["Rank"] = range(1, len(drivers) + 1)
    return drivers


def build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Churn trend by tenure month.
    For the trend chart in Power BI!
    """
    trend = df.groupby("tenure_months").agg(
        total_customers = ("customer_id", "count"),
        churned         = ("churn", "sum"),
        avg_churn_prob  = ("churn_probability", "mean")
    ).reset_index()

    trend["churn_rate"] = (
        trend["churned"] / trend["total_customers"]
    ).round(3)

    return trend


def build_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM segment breakdown.
    For the customer segmentation page in Power BI!
    """
    df["rfm_tier"] = pd.cut(
        df["rfm_score"],
        bins  = [0, 5, 8, 11, 15],
        labels= ["Low", "Medium", "High", "Champion"]
    )

    rfm_summary = df.groupby("rfm_tier", observed=True).agg(
        total_customers = ("customer_id", "count"),
        churn_rate      = ("churn", "mean"),
        avg_clv         = ("clv", "mean"),
        avg_revenue     = ("monthly_charges", "mean")
    ).reset_index()

    rfm_summary["churn_rate"]  = rfm_summary["churn_rate"].round(3)
    rfm_summary["avg_clv"]     = rfm_summary["avg_clv"].round(2)
    rfm_summary["avg_revenue"] = rfm_summary["avg_revenue"].round(2)

    return rfm_summary


def export_all(output_dir: str = None) -> None:
    """
    Export ALL tables to CSV for Power BI!
    Like packing all your reports into one folder.
    """
    if output_dir is None:
        output_dir = os.path.join(REPORTS_DIR, "powerbi")

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*55)
    print("📊 EXPORTING POWER BI DATA")
    print("="*55)

    # Load & enrich
    df = load_enriched_data()
    df = add_predictions(df)

    # Add customer_id if missing
    if "customer_id" not in df.columns:
        df["customer_id"] = [f"CUST_{i:04d}" for i in range(1, len(df) + 1)]

    # Build & save all tables
    tables = {
        "01_customer_table"  : build_customer_table(df),
        "02_segment_summary" : build_segment_summary(df),
        "03_churn_drivers"   : build_churn_drivers(df),
        "04_monthly_trend"   : build_monthly_trend(df),
        "05_rfm_segments"    : build_rfm_segments(df),
    }

    for name, table in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        table.to_csv(path, index=False)
        print(f"   💾 Saved: {name}.csv  ({len(table)} rows)")

    print("\n✅ ALL FILES EXPORTED!")
    print(f"   📁 Location: {output_dir}")
    print("\n📌 Import these CSV files into Power BI:")
    print("   1. Open Power BI Desktop")
    print("   2. Click 'Get Data' → Text/CSV")
    print("   3. Import each file one by one")
    print("="*55)


if __name__ == "__main__":
    from src.pipeline.data_loader import load_data
    from src.pipeline.preprocessor import preprocess_pipeline
    from src.features.feature_engineering import run_feature_engineering
    from src.models.train_model import train_all_models, save_models
    from sklearn.model_selection import train_test_split

    # Run full pipeline first
    df           = load_data()
    preprocess_pipeline(df)
    processed_df = pd.read_csv(PROCESSED_DATA_FILE)
    enriched_df  = run_feature_engineering(processed_df)

    drop_cols    = [TARGET_COLUMN, "customer_id"]
    feature_cols = [c for c in enriched_df.columns if c not in drop_cols]

    X = enriched_df[feature_cols].fillna(0)
    y = enriched_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    trained_models = train_all_models(X_train, y_train)
    save_models(trained_models)

    # Export Power BI files
    export_all()