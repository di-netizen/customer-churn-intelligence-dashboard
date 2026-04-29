# src/pipeline/data_loader.py

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import RAW_DATA_FILE, TARGET_COLUMN

def generate_sample_data(n_customers: int = 1000) -> pd.DataFrame:
    """
    Generates fake customer data for our project.
    Think of this as creating 1000 fake customers to practice on!
    """
    np.random.seed(42)

    data = {
        "customer_id": [f"CUST_{i:04d}" for i in range(1, n_customers + 1)],
        "age": np.random.randint(18, 70, n_customers),
        "gender": np.random.choice(["Male", "Female"], n_customers),
        "tenure_months": np.random.randint(1, 72, n_customers),
        "contract_type": np.random.choice(
            ["Month-to-Month", "One Year", "Two Year"],
            n_customers, p=[0.5, 0.3, 0.2]
        ),
        "payment_method": np.random.choice(
            ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"],
            n_customers
        ),
        "internet_service": np.random.choice(
            ["DSL", "Fiber Optic", "No"],
            n_customers, p=[0.4, 0.4, 0.2]
        ),
        "monthly_charges": np.round(np.random.uniform(20, 120, n_customers), 2),
        "num_products": np.random.randint(1, 6, n_customers),
        "num_support_tickets": np.random.randint(0, 10, n_customers),
        "login_frequency": np.random.randint(0, 30, n_customers),
        "avg_session_duration": np.round(np.random.uniform(1, 60, n_customers), 2),
        "days_since_last_login": np.random.randint(0, 180, n_customers),
    }

    df = pd.DataFrame(data)

    # Calculate total charges
    df["total_charges"] = np.round(
        df["monthly_charges"] * df["tenure_months"] * np.random.uniform(0.8, 1.2, n_customers), 2
    )

    # Calculate Customer Lifetime Value (CLV)
    df["clv"] = np.round(df["total_charges"] * np.random.uniform(1.0, 2.5, n_customers), 2)

    # Create churn label (realistic logic)
    churn_score = (
        (df["contract_type"] == "Month-to-Month").astype(int) * 0.3 +
        (df["days_since_last_login"] > 60).astype(int) * 0.25 +
        (df["num_support_tickets"] > 5).astype(int) * 0.2 +
        (df["tenure_months"] < 12).astype(int) * 0.15 +
        np.random.uniform(0, 0.1, n_customers)
    )
    df["churn"] = (churn_score > 0.4).astype(int)

    return df


def load_data() -> pd.DataFrame:
    """
    Loads customer data. If no file exists, generates sample data.
    """
    if os.path.exists(RAW_DATA_FILE):
        print(f"✅ Loading data from: {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)
    else:
        print("⚠️  No data file found. Generating sample data...")
        df = generate_sample_data(1000)
        os.makedirs(os.path.dirname(RAW_DATA_FILE), exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)
        print(f"✅ Sample data saved to: {RAW_DATA_FILE}")

    return df


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Prints a simple summary of our data — like a report card!
    """
    print("\n" + "="*50)
    print("📊 DATA SUMMARY")
    print("="*50)
    print(f"Total Customers   : {len(df)}")
    print(f"Total Features    : {len(df.columns)}")
    print(f"Churned Customers : {df[TARGET_COLUMN].sum()} ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"Missing Values    : {df.isnull().sum().sum()}")
    print(f"Duplicate Rows    : {df.duplicated().sum()}")
    print("="*50 + "\n")


if __name__ == "__main__":
    df = load_data()
    get_data_summary(df)
    print(df.head())