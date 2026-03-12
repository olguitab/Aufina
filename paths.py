import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

HISTORICAL_FEATURES_FILE = os.path.join(DATA_DIR, "historical_features.csv")
PROCESSED_FEATURES_FILE = os.path.join(DATA_DIR, "processed_features.csv")
BACKTEST_TRADES_FILE = os.path.join(DATA_DIR, "backtest_trades.csv")
BACKTEST_RESULTS_FILE = os.path.join(DATA_DIR, "backtest_results.csv")

MODEL_FILE = os.path.join(ARTIFACTS_DIR, "predictive_model.pkl")
MODEL_V2_FILE = os.path.join(ARTIFACTS_DIR, "predictive_model_v2.pkl")
TRAINING_REPORT_V2_FILE = os.path.join(DATA_DIR, "training_report_v2.json")

TRADING_DB_FILE = os.path.join(STORAGE_DIR, "trading_vault.db")
PAPER_TRADING_DB_FILE = os.path.join(STORAGE_DIR, "paper_trading_vault.db")

BOT_LOG_FILE = os.path.join(LOGS_DIR, "bot.log")
TRAINING_LOG_FILE = os.path.join(LOGS_DIR, "training.log")


def ensure_project_dirs():
    for path in (DATA_DIR, ARTIFACTS_DIR, STORAGE_DIR, LOGS_DIR):
        os.makedirs(path, exist_ok=True)
