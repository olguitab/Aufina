import os
import sys
import logging
import pandas as pd
from features import download_historical_data, generate_features
from universe import get_training_watchlist
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
    Professional training pipeline for Sentinel AI V2.
    1. Downloads 10 years of market data.
    2. Generates V2 technical + macro + context features (60+ columns).
    3. Trains V2 multi-horizon ensemble (5d, 10d, 20d).
    4. Falls back to V1 training if V2 fails.
    """
    logger.info("=== 🚀 STARTING V2 MULTI-HORIZON MODEL TRAINING (Sentinel AI) ===")
    
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

    # 2. Feature Engineering (V2 expanded)
    try:
        features_df = generate_features(raw_df, macros)
        if features_df is None or features_df.empty:
            logger.error("Failed to generate features.")
            return
        logger.info(f"Features generated: {len(features_df)} samples, {len(features_df.columns)} columns")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return

    # 3. V2 Multi-Horizon Training (primary)
    v2_success = False
    try:
        from models_v2 import train_v2_model
        logger.info("Starting V2 Multi-Horizon Ensemble Training...")
        result = train_v2_model() or {}
        if result.get("status") == "ok":
            v2_success = True
            logger.info(
                "V2 Training complete | model=%s | report=%s",
                result.get("model_file"),
                result.get("training_report_file"),
            )
            wf = result.get("walk_forward") or {}
            for horizon_key, metrics in wf.items():
                logger.info(
                    "Walk-forward %s | F1=%.3f | Precision=%.3f | Recall=%.3f | Accuracy=%.3f",
                    horizon_key,
                    metrics.get("f1_mean", 0.0),
                    metrics.get("precision_mean", 0.0),
                    metrics.get("recall_mean", 0.0),
                    metrics.get("accuracy_mean", 0.0),
                )
        else:
            logger.warning("V2 training returned non-ok status. Will try V1 fallback.")
    except Exception as e:
        logger.error(f"V2 training failed: {e}. Falling back to V1.")

    # 4. V1 Fallback Training (legacy compatibility)
    if not v2_success:
        try:
            from models import train_model
            logger.info("Starting V1 Stacking Ensemble training (fallback)...")
            result = train_model() or {}
            if result.get("status") == "ok":
                logger.info(
                    "V1 Training artifacts ready | model=%s | walkforward=%s | report=%s",
                    result.get("model_file"),
                    result.get("walkforward_file"),
                    result.get("training_report_file"),
                )
        except Exception as e:
            logger.error(f"V1 fallback training also failed: {e}")
            return
    
    logger.info("=== ✅ TRAINING COMPLETE. Sentinel AI is ready. ===")

if __name__ == "__main__":
    run_institutional_training()
