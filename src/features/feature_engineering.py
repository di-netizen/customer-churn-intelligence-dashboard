# src/features/feature_engineering.py

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import PROCESSED_DATA_FILE


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM = Recency, Frequency, Monetary
    Like judging a customer by:
    - How RECENTLY they used the service
    - How FREQUENTLY they login
    - How much MONEY they spend
    """
    print("⚡ Adding RFM features...")

    # Recency Score (lower days = better = higher score)
    df["recency_score"] = pd.cut(
        df["days_since_last_login"],
        bins=[0, 7, 30, 60, 90, 999],
        labels=[5, 4, 3, 2, 1]
    ).astype(float)

    # Frequency Score
    df["frequency_score"] = pd.cut(
        df["login_frequency"],
        bins=[-1, 5, 10, 15, 20, 999],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)

    # Monetary Score
    df["monetary_score"] = pd.cut(
        df["monthly_charges"],
        bins=[0, 30, 50, 70, 90, 999],
        labels=[1, 2, 3, 4, 5]
    ).astype(float)

    # Combined RFM Score
    df["rfm_score"] = (
        df["recency_score"] +
        df["frequency_score"] +
        df["monetary_score"]
    )

    print("   ✅ RFM features added!")
    return df


def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How engaged is this customer?
    A customer who never logs in is probably leaving soon!
    """
    print("⚡ Adding engagement features...")

    # Engagement Rate = how often they login vs how long they've been customer
    df["engagement_rate"] = np.where(
        df["tenure_months"] > 0,
        df["login_frequency"] / (df["tenure_months"] + 1),
        0
    )

    # Is the customer inactive? (no login in 30+ days)
    df["is_inactive"] = (df["days_since_last_login"] > 30).astype(int)

    # High support ticket flag (complaining a lot = might leave)
    df["high_support_tickets"] = (df["num_support_tickets"] > 5).astype(int)

    # Session quality score
    df["session_quality"] = np.round(
        df["avg_session_duration"] * df["login_frequency"], 2
    )

    print("   ✅ Engagement features added!")
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine signals into a RISK SCORE.
    Higher score = more likely to leave!
    """
    print("⚡ Adding risk features...")

    # Charge per product (expensive per product = risky)
    df["charge_per_product"] = np.where(
        df["num_products"] > 0,
        df["monthly_charges"] / df["num_products"],
        df["monthly_charges"]
    )

    # Tenure risk (new customers leave more)
    df["is_new_customer"] = (df["tenure_months"] < 12).astype(int)

    # Combined churn risk score
    df["churn_risk_score"] = (
        df["is_inactive"] * 0.3 +
        df["high_support_tickets"] * 0.25 +
        df["is_new_customer"] * 0.2 +
        (1 / (df["rfm_score"] + 1)) * 0.25
    ).round(4)

    print("   ✅ Risk features added!")
    return df


def add_clv_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers by their value to the business.
    Like VIP, Regular, and Budget customers!
    """
    print("⚡ Adding CLV segments...")

    df["clv_segment"] = pd.cut(
        df["clv"],
        bins=3,
        labels=[1, 2, 3]   # 1=Low, 2=Medium, 3=High value
    ).astype(float)

    print("   ✅ CLV segments added!")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs ALL feature engineering steps — giving AI all superpowers! 🦸
    """
    print("\n" + "="*50)
    print("⚡ STARTING FEATURE ENGINEERING")
    print("="*50)

    df = add_rfm_features(df)
    df = add_engagement_features(df)
    df = add_risk_features(df)
    df = add_clv_segment(df)

    # Save enriched data
    enriched_path = PROCESSED_DATA_FILE.replace(
        "customers_processed.csv",
        "customers_enriched.csv"
    )
    df.to_csv(enriched_path, index=False)
    print(f"\n✅ Enriched data saved to: {enriched_path}")
    print(f"✅ Total features now: {len(df.columns)}")
    print("="*50 + "\n")

    return df


if __name__ == "__main__":
    from src.pipeline.data_loader import load_data
    from src.pipeline.preprocessor import preprocess_pipeline

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_pipeline(df)

    # Reload processed data to add features
    processed_df = pd.read_csv(PROCESSED_DATA_FILE)
    enriched_df  = run_feature_engineering(processed_df)

    print(enriched_df.head(3))
    print(f"\nNew columns added: {len(enriched_df.columns)} total")