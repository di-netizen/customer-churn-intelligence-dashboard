# src/config.py
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ── Data File ──────────────────────────────────────────
RAW_DATA_FILE = os.path.join(RAW_DIR, "customers.csv")
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DIR, "customers_processed.csv")

# ── Model Settings ─────────────────────────────────────
TARGET_COLUMN   = "churn"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ── Features ───────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "contract_type",
    "payment_method", "internet_service"
]

NUMERICAL_COLS = [
    "age", "tenure_months", "monthly_charges",
    "total_charges", "num_products", "num_support_tickets",
    "login_frequency", "avg_session_duration",
    "days_since_last_login", "clv"
]

# ── Model Save Paths ───────────────────────────────────
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, "scaler.pkl")