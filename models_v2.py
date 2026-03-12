"""
Aureus V2 — Multi-Horizon Robust Trading Model
================================================
Professional ML pipeline for medium-term Chilean market investment.

Architecture:
- Multi-horizon direction classifiers (5d, 10d, 20d)
- Multi-horizon magnitude regressors (expected return per horizon)
- Exit timing model (optimal days to hold)
- Purged walk-forward validation (no data leakage)
- Optuna Bayesian hyperparameter optimization
- Stacking ensemble: XGBoost + LightGBM + RandomForest + ExtraTrees

Usage:
    from models_v2 import train_v2_model, MultiHorizonPredictor
    train_v2_model()  # Trains and saves model
    pred = MultiHorizonPredictor()
    result = pred.predict(tech_data, context_score=0.2)
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from datetime import datetime, timezone
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
except Exception:
    shap = None

from paths import MODEL_V2_FILE, PROCESSED_FEATURES_FILE, TRAINING_REPORT_V2_FILE, DATA_DIR, ensure_project_dirs

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Feature Definitions
# =============================================================================

# V2 expanded feature set (60+ features)
FEATURE_COLS_V2 = [
    # Momentum multi-timeframe
    'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d', 'Return_40d', 'Return_60d',
    # Trend / MA distances
    'Dist_MA20', 'Dist_MA50', 'Dist_MA200', 'MA20_MA50_Cross',
    'Dist_52w_High', 'Dist_52w_Low',
    # Volatility
    'Volatility_20d', 'Volatility_5d', 'Vol_Compression', 'ATR_Ratio',
    # Oscillators
    'RSI_14', 'StochRSI', 'WilliamsR', 'ADX_14',
    # MACD
    'MACD', 'MACD_Signal', 'MACD_Hist',
    # Bollinger
    'BB_Width', 'BB_Position',
    # Volume
    'OBV', 'OBV_MA20', 'OBV_ROC', 'Volume_Ratio',
    # Macro V1
    'Macro_Copper_Ret', 'Macro_SP500_Ret', 'Macro_USDCLP_Ret',
    'Macro_Lithium_Ret', 'Macro_MSCI_EM_Ret', 'Macro_VIX_Ret',
    # Macro V2
    'Macro_WTI_Oil_Ret', 'Macro_DXY_Ret', 'Macro_IronOre_Ret',
    'Macro_IPSA_Ret', 'Macro_Gold_Ret',
    # Inter-market correlations
    'Copper_Rolling_Corr_30d',
    'WTI_Oil_Corr_20d', 'Gold_Corr_20d', 'DXY_Corr_20d',
    'IronOre_Corr_20d', 'IPSA_Corr_20d',
    # Seasonality
    'DayOfWeek', 'MonthOfYear', 'IsMonday', 'IsFriday',
    # Context
    'Context_Score',
]

# V1 backward-compatible feature set
FEATURE_COLS_V1 = [
    'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
    'Dist_MA20', 'Dist_MA50', 'Dist_MA200', 'MA20_MA50_Cross',
    'Volatility_20d', 'Volatility_5d', 'ATR_Ratio',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'BB_Position', 'OBV', 'OBV_MA20', 'Volume_Ratio',
    'Macro_Copper_Ret', 'Macro_SP500_Ret', 'Macro_USDCLP_Ret',
    'Macro_Lithium_Ret', 'Macro_MSCI_EM_Ret', 'Macro_VIX_Ret',
    'Copper_Rolling_Corr_30d', 'Context_Score',
]

HORIZONS = [5, 10, 20]
MODEL_SCHEMA_VERSION = 3
PURGE_DAYS = 5    # Gap between train/test to prevent leakage
EMBARGO_DAYS = 3  # Extra buffer after test split

WALKFORWARD_V2_FILE = os.path.join(DATA_DIR, "walkforward_v2_results.csv")


# =============================================================================
# Purged Time Series Split (no data leakage)
# =============================================================================

class PurgedTimeSeriesSplit:
    """Walk-forward validation with purging and embargo to prevent temporal leakage."""

    def __init__(self, n_splits=6, purge_days=5, embargo_days=3):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, dates=None):
        n = len(X)
        min_train = max(500, n // (self.n_splits + 2))
        test_size = max(100, (n - min_train) // self.n_splits)
        gap = self.purge_days + self.embargo_days

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - gap

            if train_end < min_train // 2:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_n_splits(self):
        return self.n_splits


# =============================================================================
# Optuna Objective for Bayesian Optimization
# =============================================================================

def _optuna_objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    """Bayesian optimization objective for a single horizon model."""
    model_type = trial.suggest_categorical("model_type", ["xgb", "lgbm"])

    if model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = xgb.XGBClassifier(**params)
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)[:, 1]
    # Optimize for F1 with reasonable precision
    best_f1 = 0.0
    for thr in np.arange(0.25, 0.65, 0.05):
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


# =============================================================================
# Core Training Function
# =============================================================================

def _resolve_feature_cols(df: pd.DataFrame) -> list:
    """Returns the best possible feature set given the available columns."""
    available_v2 = [c for c in FEATURE_COLS_V2 if c in df.columns]
    if len(available_v2) >= len(FEATURE_COLS_V2) * 0.8:
        return available_v2
    available_v1 = [c for c in FEATURE_COLS_V1 if c in df.columns]
    return available_v1 if available_v1 else available_v2


def _build_stacking_model(scale_pos_weight: float, best_params: dict = None):
    """Build a professional stacking ensemble."""
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced',
            min_samples_leaf=5, n_jobs=-1,
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=best_params.get("xgb_n_estimators", 200) if best_params else 200,
            max_depth=best_params.get("xgb_max_depth", 5) if best_params else 5,
            learning_rate=best_params.get("xgb_lr", 0.05) if best_params else 0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
        )),
        ('lgbm', lgb.LGBMClassifier(
            n_estimators=best_params.get("lgbm_n_estimators", 200) if best_params else 200,
            max_depth=best_params.get("lgbm_max_depth", 5) if best_params else 5,
            learning_rate=best_params.get("lgbm_lr", 0.05) if best_params else 0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbosity=-1,
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=200, random_state=42, class_weight='balanced',
            min_samples_leaf=5, n_jobs=-1,
        )),
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight='balanced', max_iter=1000, C=0.5,
        ),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return stack


def _calibrate_threshold(y_true, y_prob, min_precision=0.25):
    """Calibrate decision threshold using OOS probabilities."""
    candidates = np.round(np.arange(0.20, 0.70, 0.03), 2)
    best_constrained = None
    best_any = None

    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        row = {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1}

        if best_any is None or f1 > best_any["f1"]:
            best_any = row
        if prec >= min_precision:
            if best_constrained is None or f1 > best_constrained["f1"]:
                best_constrained = row

    selected = best_constrained if best_constrained else best_any
    return float(selected["threshold"]) if selected else 0.40, selected or {}


def _run_optuna_for_horizon(X, y, dates, horizon_name, n_trials=40, scale_pos_weight=1.0):
    """Run Bayesian optimization for a specific horizon using purged splits."""
    if not HAS_OPTUNA:
        print(f"  ⚠️ Optuna not installed. Using default parameters for {horizon_name}.")
        return {}

    splitter = PurgedTimeSeriesSplit(n_splits=3, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS)
    splits = list(splitter.split(X, dates))
    if not splits:
        return {}

    # Use the last split for Optuna optimization
    train_idx, val_idx = splits[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    study = optuna.create_study(direction="maximize", study_name=f"horizon_{horizon_name}")
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    print(f"  ✅ Optuna best for {horizon_name}: F1={study.best_value:.3f} | params={best}")
    return best


def train_v2_model():
    """
    Professional V2 Training Pipeline.
    Trains multi-horizon direction classifiers, magnitude regressors, and exit timing model.
    """
    ensure_project_dirs()
    if not os.path.exists(PROCESSED_FEATURES_FILE):
        print("❌ No features found. Run features.py first.")
        return {"status": "error", "reason": "missing_features_file"}

    df = pd.read_csv(PROCESSED_FEATURES_FILE)
    feature_cols = _resolve_feature_cols(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️ Missing columns (filling with 0): {missing}")
        for c in missing:
            df[c] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e6, 1e6)
    dates = df['Date']

    print(f"\n{'='*60}")
    print(f"🚀 AUREUS V2 MULTI-HORIZON TRAINING")
    print(f"{'='*60}")
    print(f"📊 Samples: {len(df)}")
    print(f"📐 Features: {len(feature_cols)}")
    print(f"🎯 Horizons: {HORIZONS}")
    print(f"{'='*60}\n")

    model_payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "horizons": HORIZONS,
        "direction_models": {},
        "magnitude_models": {},
        "thresholds": {},
        "exit_model": None,
        "optuna_results": {},
        "trained_at": None,
    }

    training_report = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "trained_at": None,
        "samples": len(df),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "horizons": {},
        "exit_model": {},
        "walk_forward": {},
    }

    # =========================================================================
    # PHASE 1: Train Direction Classifiers per Horizon
    # =========================================================================
    for horizon in HORIZONS:
        direction_col = f"Target_Direction_{horizon}d"
        magnitude_col = f"Target_Magnitude_{horizon}d"

        if direction_col not in df.columns:
            print(f"⚠️ Column {direction_col} not found. Skipping horizon {horizon}d.")
            continue

        y_dir = df[direction_col].fillna(0).astype(int)
        valid_mask = y_dir.notna()
        X_h = X[valid_mask].copy()
        y_h = y_dir[valid_mask].copy()
        dates_h = dates[valid_mask]

        positives = max(int((y_h == 1).sum()), 1)
        negatives = max(int((y_h == 0).sum()), 1)
        spw = negatives / positives

        print(f"\n{'─'*50}")
        print(f"📈 HORIZON {horizon}d | samples={len(y_h)} | pos={positives} | neg={negatives}")
        print(f"{'─'*50}")

        # Step 1: Optuna hyperparameter optimization
        n_trials = int(os.environ.get("OPTUNA_TRIALS", 30))
        print(f"  🔍 Running Optuna ({n_trials} trials)...")
        optuna_params = _run_optuna_for_horizon(X_h, y_h, dates_h, f"{horizon}d", n_trials=n_trials, scale_pos_weight=spw)
        model_payload["optuna_results"][f"{horizon}d"] = optuna_params

        # Step 2: Build stacking ensemble with optimized params
        best_params = {}
        if optuna_params.get("model_type") == "xgb":
            best_params["xgb_n_estimators"] = optuna_params.get("n_estimators", 200)
            best_params["xgb_max_depth"] = optuna_params.get("max_depth", 5)
            best_params["xgb_lr"] = optuna_params.get("learning_rate", 0.05)
        elif optuna_params.get("model_type") == "lgbm":
            best_params["lgbm_n_estimators"] = optuna_params.get("n_estimators", 200)
            best_params["lgbm_max_depth"] = optuna_params.get("max_depth", 5)
            best_params["lgbm_lr"] = optuna_params.get("learning_rate", 0.05)

        print(f"  🏗️ Building Stacking Ensemble...")
        stack = _build_stacking_model(spw, best_params)
        stack.fit(X_h, y_h)
        model_payload["direction_models"][f"{horizon}d"] = stack

        # Step 3: Calibrate threshold via purged walk-forward OOS
        print(f"  📏 Calibrating threshold...")
        oos_true, oos_prob = [], []
        splitter = PurgedTimeSeriesSplit(n_splits=6, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS)
        wf_rows = []

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X_h, dates_h), 1):
            fold_model = clone(stack)
            fold_model.fit(X_h.iloc[train_idx], y_h.iloc[train_idx])
            probs = fold_model.predict_proba(X_h.iloc[test_idx])[:, 1]
            oos_true.extend(y_h.iloc[test_idx].tolist())
            oos_prob.extend(probs.tolist())

            # Walk-forward metrics per fold
            for thr_cand in [0.35, 0.40, 0.45]:
                y_pred = (probs >= thr_cand).astype(int)
                wf_rows.append({
                    'horizon': f'{horizon}d',
                    'fold': fold_id,
                    'threshold': thr_cand,
                    'precision': precision_score(y_h.iloc[test_idx], y_pred, zero_division=0),
                    'recall': recall_score(y_h.iloc[test_idx], y_pred, zero_division=0),
                    'f1': f1_score(y_h.iloc[test_idx], y_pred, zero_division=0),
                    'accuracy': accuracy_score(y_h.iloc[test_idx], y_pred),
                    'train_end': str(dates_h.iloc[train_idx[-1]]),
                    'test_start': str(dates_h.iloc[test_idx[0]]),
                    'test_end': str(dates_h.iloc[test_idx[-1]]),
                    'support': len(test_idx),
                })

        threshold, cal_summary = _calibrate_threshold(
            np.array(oos_true), np.array(oos_prob), min_precision=0.25,
        )
        model_payload["thresholds"][f"{horizon}d"] = threshold
        print(f"  ✅ Threshold {horizon}d = {threshold:.2f} | OOS F1={cal_summary.get('f1', 0):.3f}")

        # Step 4: Magnitude regressor
        if magnitude_col in df.columns:
            y_mag = df[magnitude_col].fillna(0.0).astype(float)
            valid_mag = y_mag.notna()
            mag_model = RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5,
            )
            mag_model.fit(X[valid_mag], y_mag[valid_mag])
            model_payload["magnitude_models"][f"{horizon}d"] = mag_model
            print(f"  ✅ Magnitude regressor trained for {horizon}d")

        # Final report for this horizon
        y_all_proba = stack.predict_proba(X_h)[:, 1]
        y_all_pred = (y_all_proba >= threshold).astype(int)
        report_text = classification_report(y_h, y_all_pred)
        print(f"\n  Classification Report ({horizon}d):")
        print(f"  {report_text}")

        training_report["horizons"][f"{horizon}d"] = {
            "samples": len(y_h),
            "positives": positives,
            "negatives": negatives,
            "threshold": threshold,
            "calibration": cal_summary,
            "optuna_best": optuna_params,
            "classification_report": report_text,
        }

    # =========================================================================
    # PHASE 2: Exit Timing Model (when to sell)
    # =========================================================================
    exit_col = "Target_OptimalExit"
    if exit_col in df.columns:
        print(f"\n{'─'*50}")
        print(f"🚪 Training EXIT TIMING model...")
        print(f"{'─'*50}")

        y_exit = df[exit_col].fillna(10).astype(int)
        valid_exit = y_exit.notna()
        X_exit = X[valid_exit]
        y_exit_valid = y_exit[valid_exit]

        # Bin exit days into classes: short (1-5), medium (6-12), long (13-20)
        def _bin_exit(day):
            if day <= 5:
                return 0  # short hold
            elif day <= 12:
                return 1  # medium hold
            else:
                return 2  # long hold

        y_exit_binned = y_exit_valid.apply(_bin_exit)

        exit_model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight='balanced', min_samples_leaf=5,
        )
        exit_model.fit(X_exit, y_exit_binned)
        model_payload["exit_model"] = exit_model

        # Also store the raw exit regressor for precise exit day estimate
        exit_regressor = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5,
        )
        exit_regressor.fit(X_exit, y_exit_valid)
        model_payload["exit_regressor"] = exit_regressor

        exit_classes = sorted(y_exit_binned.unique().tolist())
        training_report["exit_model"] = {
            "samples": len(y_exit_valid),
            "classes": exit_classes,
            "class_names": {0: "short (1-5d)", 1: "medium (6-12d)", 2: "long (13-20d)"},
        }
        print(f"  ✅ Exit timing model trained on {len(y_exit_valid)} samples")
    else:
        print(f"  ⚠️ No {exit_col} column found. Exit timing model skipped.")

    # =========================================================================
    # PHASE 3: Save Walk-Forward Results
    # =========================================================================
    if wf_rows:
        wf_df = pd.DataFrame(wf_rows)
        wf_df.to_csv(WALKFORWARD_V2_FILE, index=False)
        print(f"\n📈 Walk-forward results saved to {WALKFORWARD_V2_FILE}")

        # Aggregate per horizon
        for h in HORIZONS:
            h_key = f"{h}d"
            h_df = wf_df[wf_df['horizon'] == h_key]
            if h_df.empty:
                continue
            # Use threshold closest to calibrated
            cal_thr = model_payload["thresholds"].get(h_key, 0.40)
            closest_rows = h_df.iloc[(h_df['threshold'] - cal_thr).abs().argsort()[:len(h_df)//3 + 1]]
            training_report["walk_forward"][h_key] = {
                "f1_mean": float(closest_rows['f1'].mean()),
                "precision_mean": float(closest_rows['precision'].mean()),
                "recall_mean": float(closest_rows['recall'].mean()),
                "accuracy_mean": float(closest_rows['accuracy'].mean()),
                "folds": int(len(closest_rows)),
            }
            print(
                f"  Walk-forward {h}d | "
                f"F1={closest_rows['f1'].mean():.3f} | "
                f"Precision={closest_rows['precision'].mean():.3f} | "
                f"Recall={closest_rows['recall'].mean():.3f}"
            )

    # =========================================================================
    # PHASE 4: Save Model
    # =========================================================================
    model_payload["trained_at"] = datetime.now(timezone.utc).isoformat()
    training_report["trained_at"] = model_payload["trained_at"]

    joblib.dump(model_payload, MODEL_V2_FILE)
    print(f"\n💾 V2 Model saved to {MODEL_V2_FILE}")

    with open(TRAINING_REPORT_V2_FILE, "w", encoding="utf-8") as f:
        # JSON-serializable copy (strip sklearn objects)
        json.dump(training_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"🧾 Training report saved to {TRAINING_REPORT_V2_FILE}")

    print(f"\n{'='*60}")
    print(f"✅ AUREUS V2 TRAINING COMPLETE")
    print(f"{'='*60}\n")

    return {
        "status": "ok",
        "model_file": MODEL_V2_FILE,
        "training_report_file": TRAINING_REPORT_V2_FILE,
        "walkforward_file": WALKFORWARD_V2_FILE,
        "walk_forward": training_report.get("walk_forward", {}),
    }


# =============================================================================
# Prediction Interface (Production)
# =============================================================================

class MultiHorizonPredictor:
    """Production-grade predictor for multi-horizon signals."""
    _model = None

    @classmethod
    def load_model(cls):
        if cls._model is None and os.path.exists(MODEL_V2_FILE):
            loaded = joblib.load(MODEL_V2_FILE)
            if isinstance(loaded, dict) and loaded.get("schema_version", 0) >= MODEL_SCHEMA_VERSION:
                cls._model = loaded
        return cls._model

    @classmethod
    def is_available(cls) -> bool:
        return cls.load_model() is not None

    @classmethod
    def _get_feature_cols(cls) -> list:
        bundle = cls.load_model()
        if bundle:
            return bundle.get("feature_cols", FEATURE_COLS_V2)
        return FEATURE_COLS_V2

    @classmethod
    def build_feature_vector(cls, tech_data: dict, context_score: float = 0.0) -> pd.DataFrame:
        """Maps live technical_data dict to a feature vector for prediction."""
        features = {
            'Return_1d': tech_data.get('DailyReturn_Pct', 0) / 100,
            'Return_5d': tech_data.get('FiveDayReturn_Pct', 0) / 100,
            'Return_10d': tech_data.get('TenDayReturn_Pct', 0) / 100,
            'Return_20d': tech_data.get('MonthlyReturn_Pct', 0) / 100,
            'Return_40d': tech_data.get('Return_40d_Pct', tech_data.get('MonthlyReturn_Pct', 0)) / 100,
            'Return_60d': tech_data.get('Return_60d_Pct', tech_data.get('MonthlyReturn_Pct', 0)) / 100,
            'Dist_MA20': tech_data.get('Dist_MA20_Pct', 0) / 100,
            'Dist_MA50': tech_data.get('Dist_MA50_Pct', 0) / 100,
            'Dist_MA200': tech_data.get('Dist_MA200_Pct', 0) / 100,
            'MA20_MA50_Cross': 1.0 if tech_data.get('MA20', 0) > tech_data.get('MA50', 0) else 0.0,
            'Dist_52w_High': tech_data.get('Dist_52w_High', -0.1),
            'Dist_52w_Low': tech_data.get('Dist_52w_Low', 0.5),
            'Volatility_20d': tech_data.get('Volatility_20d', 0.01),
            'Volatility_5d': tech_data.get('Volatility_5d', tech_data.get('Volatility_20d', 0.01)),
            'Vol_Compression': tech_data.get('Vol_Compression', 1.0),
            'ATR_Ratio': tech_data.get('ATR_Ratio', 0.01),
            'RSI_14': tech_data.get('RSI_14', 50),
            'StochRSI': tech_data.get('StochRSI', 0.5),
            'WilliamsR': tech_data.get('WilliamsR', -50),
            'ADX_14': tech_data.get('ADX_14', 25),
            'MACD': tech_data.get('MACD', 0.0),
            'MACD_Signal': tech_data.get('MACD_Signal', 0.0),
            'MACD_Hist': tech_data.get('MACD_Hist', 0.0),
            'BB_Width': tech_data.get('BB_Width', 0.0),
            'BB_Position': tech_data.get('BB_Position', 0.5),
            'OBV': tech_data.get('OBV', 0.0),
            'OBV_MA20': tech_data.get('OBV_MA20', 0.0),
            'OBV_ROC': tech_data.get('OBV_ROC', 0.0),
            'Volume_Ratio': tech_data.get('Volume_Ratio', 1.0),
            'Macro_Copper_Ret': tech_data.get('Macro_Copper_Ret', 0.0),
            'Macro_SP500_Ret': tech_data.get('Macro_SP500_Ret', 0.0),
            'Macro_USDCLP_Ret': tech_data.get('Macro_USDCLP_Ret', 0.0),
            'Macro_Lithium_Ret': tech_data.get('Macro_Lithium_Ret', 0.0),
            'Macro_MSCI_EM_Ret': tech_data.get('Macro_MSCI_EM_Ret', 0.0),
            'Macro_VIX_Ret': tech_data.get('Macro_VIX_Ret', 0.0),
            'Macro_WTI_Oil_Ret': tech_data.get('Macro_WTI_Oil_Ret', 0.0),
            'Macro_DXY_Ret': tech_data.get('Macro_DXY_Ret', 0.0),
            'Macro_IronOre_Ret': tech_data.get('Macro_IronOre_Ret', 0.0),
            'Macro_IPSA_Ret': tech_data.get('Macro_IPSA_Ret', 0.0),
            'Macro_Gold_Ret': tech_data.get('Macro_Gold_Ret', 0.0),
            'Copper_Rolling_Corr_30d': tech_data.get('Copper_Rolling_Corr_30d', 0.0),
            'WTI_Oil_Corr_20d': tech_data.get('WTI_Oil_Corr_20d', 0.0),
            'Gold_Corr_20d': tech_data.get('Gold_Corr_20d', 0.0),
            'DXY_Corr_20d': tech_data.get('DXY_Corr_20d', 0.0),
            'IronOre_Corr_20d': tech_data.get('IronOre_Corr_20d', 0.0),
            'IPSA_Corr_20d': tech_data.get('IPSA_Corr_20d', 0.0),
            'DayOfWeek': tech_data.get('DayOfWeek', 0.5),
            'MonthOfYear': tech_data.get('MonthOfYear', 0.5),
            'IsMonday': tech_data.get('IsMonday', 0.0),
            'IsFriday': tech_data.get('IsFriday', 0.0),
            'Context_Score': context_score,
        }
        return pd.DataFrame([features])

    @classmethod
    def predict(cls, tech_data: dict, context_score: float = 0.0) -> dict:
        """
        Full multi-horizon prediction.

        Returns:
            {
                "horizons": {
                    "5d": {"probability": 0.65, "signal": "BUY", "expected_return": 0.023},
                    "10d": {...},
                    "20d": {...},
                },
                "best_horizon": "10d",
                "best_probability": 0.72,
                "suggested_hold_days": 8,
                "exit_category": "medium",
                "composite_signal": "BUY",
                "composite_confidence": 0.68,
            }
        """
        bundle = cls.load_model()
        if not bundle:
            return cls._fallback_prediction()

        X = cls.build_feature_vector(tech_data, context_score)
        model_feature_cols = bundle.get("feature_cols", FEATURE_COLS_V2)
        for col in model_feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model_feature_cols]

        horizons_result = {}
        best_horizon = None
        best_prob = 0.0

        for horizon in bundle.get("horizons", HORIZONS):
            h_key = f"{horizon}d"
            dir_model = bundle.get("direction_models", {}).get(h_key)
            mag_model = bundle.get("magnitude_models", {}).get(h_key)
            threshold = bundle.get("thresholds", {}).get(h_key, 0.40)

            if dir_model is None:
                continue

            try:
                prob = float(dir_model.predict_proba(X)[0][1])
            except Exception:
                prob = 0.5

            signal = "BUY" if prob >= threshold else ("SELL" if prob < 0.30 else "HOLD")

            exp_ret = 0.0
            if mag_model is not None:
                try:
                    exp_ret = float(mag_model.predict(X)[0])
                except Exception:
                    pass

            horizons_result[h_key] = {
                "probability": prob,
                "signal": signal,
                "expected_return": exp_ret,
                "threshold": threshold,
            }

            if prob > best_prob:
                best_prob = prob
                best_horizon = h_key

        # Exit timing model
        suggested_hold_days = 10
        exit_category = "medium"
        exit_model = bundle.get("exit_model")
        exit_regressor = bundle.get("exit_regressor")

        if exit_regressor is not None:
            try:
                suggested_hold_days = max(1, min(20, int(round(exit_regressor.predict(X)[0]))))
            except Exception:
                suggested_hold_days = 10

        if exit_model is not None:
            try:
                exit_pred = int(exit_model.predict(X)[0])
                exit_category = {0: "short", 1: "medium", 2: "long"}.get(exit_pred, "medium")
            except Exception:
                exit_category = "medium"

        # Composite signal: weighted vote across horizons
        buy_votes = sum(1 for h in horizons_result.values() if h["signal"] == "BUY")
        sell_votes = sum(1 for h in horizons_result.values() if h["signal"] == "SELL")
        total = len(horizons_result) or 1

        if buy_votes / total >= 0.5:
            composite = "BUY"
        elif sell_votes / total >= 0.5:
            composite = "SELL"
        else:
            composite = "HOLD"

        # Composite confidence: weighted average probability
        probs = [h["probability"] for h in horizons_result.values()]
        composite_confidence = np.mean(probs) if probs else 0.5

        return {
            "horizons": horizons_result,
            "best_horizon": best_horizon,
            "best_probability": best_prob,
            "suggested_hold_days": suggested_hold_days,
            "exit_category": exit_category,
            "composite_signal": composite,
            "composite_confidence": float(composite_confidence),
        }

    @classmethod
    def predict_probability(cls, tech_data: dict, context_score: float = 0.0) -> float:
        """Backward-compatible: returns single probability (best horizon)."""
        result = cls.predict(tech_data, context_score)
        return result.get("best_probability", 0.5)

    @classmethod
    def _fallback_prediction(cls) -> dict:
        return {
            "horizons": {},
            "best_horizon": None,
            "best_probability": 0.5,
            "suggested_hold_days": 10,
            "exit_category": "medium",
            "composite_signal": "HOLD",
            "composite_confidence": 0.5,
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    result = train_v2_model()
    if result and result.get("status") == "ok":
        print("\n🎉 V2 Model training completed successfully!")
        print(f"   Model: {result.get('model_file')}")
        print(f"   Report: {result.get('training_report_file')}")
    else:
        print("\n❌ Training failed.")
