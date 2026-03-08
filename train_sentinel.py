import os
import sys
import logging
import pandas as pd
from features import download_historical_data, generate_features
from universe import get_training_watchlist
from models import train_model
from paths import TRAINING_LOG_FILE, ensure_project_dirs

# --- Setup Logging ---
ensure_project_dirs()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(TRAINING_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_institutional_training():
    """
    Professional training pipeline for Sentinel AI.
    1. Downloads 10 years of market data.
    2. Generates technical + macro features.
    3. Trains institutional ensemble (Stacking: XGB, LGBM, RF).
    """
    logger.info("=== 🚀 STARTING INSTITUTIONAL MODEL TRAINING (Sentinel AI) ===")
    
    # 1. Data Acquisition
    try:
        training_watchlist = get_training_watchlist(include_global=True)
        raw_df, macros = download_historical_data(training_watchlist, years=10)
        if raw_df is None or raw_df.empty:
            logger.error("Failed to download historical data.")
            return
    except Exception as e:
        logger.error(f"Error in data acquisition: {e}")
        return

    # 2. Feature Engineering
    try:
        features_df = generate_features(raw_df, macros)
        if features_df is None or features_df.empty:
            logger.error("Failed to generate features.")
            return
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return

    # 3. Model Training (Exhaustive Search)
    try:
        logger.info("Starting exhaustive Stacking Ensemble training...")
        result = train_model() or {}
        if result.get("status") == "ok":
            logger.info(
                "Training artifacts ready | model=%s | walkforward=%s | report=%s",
                result.get("model_file"),
                result.get("walkforward_file"),
                result.get("training_report_file"),
            )
            wf = result.get("walk_forward") or {}
            if wf:
                logger.info(
                    "Walk-forward summary | F1=%.3f | Precision=%.3f | Recall=%.3f | Accuracy=%.3f",
                    wf.get("f1_mean", 0.0),
                    wf.get("precision_mean", 0.0),
                    wf.get("recall_mean", 0.0),
                    wf.get("accuracy_mean", 0.0),
                )
        logger.info("=== ✅ TRAINING COMPLETE. Sentinel AI is now robust. ===")
    except Exception as e:
        logger.error(f"Error during model training: {e}")

if __name__ == "__main__":
    run_institutional_training()
