import os
import sys
import logging
import pandas as pd
from features import download_historical_data, generate_features, FULL_WATCHLIST
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
        raw_df, macros = download_historical_data(FULL_WATCHLIST, years=10)
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
        train_model()
        logger.info("=== ✅ TRAINING COMPLETE. Sentinel AI is now robust. ===")
    except Exception as e:
        logger.error(f"Error during model training: {e}")

if __name__ == "__main__":
    run_institutional_training()
