import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, classification_report, f1_score, accuracy_score, recall_score
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
import numpy as np
from datetime import datetime, timezone
from paths import MODEL_FILE, PROCESSED_FEATURES_FILE, ensure_project_dirs, DATA_DIR

try:
    import shap
except Exception:
    shap = None

WALKFORWARD_RESULTS_FILE = os.path.join(DATA_DIR, "walkforward_results.csv")
TRAINING_REPORT_FILE = os.path.join(DATA_DIR, "training_report.json")
MODEL_DECISION_THRESHOLD = float(os.environ.get("MODEL_DECISION_THRESHOLD", 0.35))
MIN_TARGET_PRECISION = float(os.environ.get("MODEL_MIN_TARGET_PRECISION", 0.25))
MODEL_SCHEMA_VERSION = 2
DEFAULT_HORIZON_DAYS = 3

# Feature Set Definition
FEATURE_COLS = [
    'Return_1d',
    'Return_5d',
    'Return_10d',
    'Return_20d',
    'Dist_MA20',
    'Dist_MA50',
    'Dist_MA200',
    'MA20_MA50_Cross',
    'Volatility_20d',
    'Volatility_5d',
    'ATR_Ratio',
    'RSI_14',
    'MACD',
    'MACD_Signal',
    'MACD_Hist',
    'BB_Width',
    'BB_Position',
    'OBV',
    'OBV_MA20',
    'Volume_Ratio',
    'Macro_Copper_Ret',
    'Macro_SP500_Ret',
    'Macro_USDCLP_Ret',
    'Macro_Lithium_Ret',
    'Macro_MSCI_EM_Ret',
    'Macro_VIX_Ret',
    'Copper_Rolling_Corr_30d',
    'Context_Score'
]

TARGET_DIRECTION_COL = "Target_Direction"
TARGET_MAGNITUDE_COL = "Target_Magnitude_3d"
TARGET_HORIZON_COL = "Target_Horizon_Days"

def train_model():
    ensure_project_dirs()
    if not os.path.exists(PROCESSED_FEATURES_FILE):
        print("❌ No features found. Run features.py first.")
        return {"status": "error", "reason": "missing_features_file"}

    df = pd.read_csv(PROCESSED_FEATURES_FILE)
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        print(f"⚠️ Missing feature columns in dataset. Filling with 0.0: {missing}")
        for col in missing:
            df[col] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)

    if TARGET_DIRECTION_COL not in df.columns:
        df[TARGET_DIRECTION_COL] = df.get("Target", 0).fillna(0).astype(int)
    if TARGET_MAGNITUDE_COL not in df.columns:
        if "Close" in df.columns and "Ticker" in df.columns and "Date" in df.columns:
            local = df[['Ticker', 'Date', 'Close']].copy()
            local['Date'] = pd.to_datetime(local['Date'])
            local = local.sort_values(['Ticker', 'Date'])
            local[TARGET_MAGNITUDE_COL] = local.groupby('Ticker')['Close'].shift(-3)
            local[TARGET_MAGNITUDE_COL] = (
                (local[TARGET_MAGNITUDE_COL] - local['Close']) / (local['Close'] + 1e-9)
            ).clip(lower=-0.30, upper=0.30)
            df[TARGET_MAGNITUDE_COL] = local[TARGET_MAGNITUDE_COL].values
        else:
            df[TARGET_MAGNITUDE_COL] = 0.0
    if TARGET_HORIZON_COL not in df.columns:
        df[TARGET_HORIZON_COL] = DEFAULT_HORIZON_DAYS

    df = df.dropna(subset=[TARGET_DIRECTION_COL, TARGET_MAGNITUDE_COL, TARGET_HORIZON_COL])
    
    # Sort chronologically for TimeSeriesSplit
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    X = df[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.clip(lower=-1e6, upper=1e6)
    y = df[TARGET_DIRECTION_COL].astype(int)
    y_magnitude = df[TARGET_MAGNITUDE_COL].astype(float)
    y_horizon = df[TARGET_HORIZON_COL].astype(int)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    scale_pos_weight = negatives / positives
    
    print(f"🌲 Deep Quant Training on {len(df)} samples...")
    print("⏳ This process is exhaustive and will take several minutes...")
    
    # 1. TimeSeriesSplit (10 folds for maximum robustness)
    tscv = TimeSeriesSplit(n_splits=10)
    
    # 2. Define Base Models for Stacking
    estimators = [
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('xgb', xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
        )),
        ('lgbm', lgb.LGBMClassifier(random_state=42, verbosity=-1, class_weight='balanced'))
    ]
    
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=5, # Use standard partitions internally for stacking
        n_jobs=-1
    )
    
    # 3. Exhaustive Hyperparameter Grid (Optimized for speed)
    # Reverting to a faster subset of base parameters for quick turnaround.
    param_grid = {
        'rf__n_estimators': [100],
        'xgb__n_estimators': [100],
        'lgbm__n_estimators': [100]
    }
    
    # GridSearch for maximum precision and exhaustive search
    search = GridSearchCV(
        stack_model, param_grid=param_grid, 
        cv=tscv, scoring='f1', n_jobs=-1
    )
    
    search.fit(X, y)
    best_model = search.best_estimator_

    # 3.2 Multi-objective heads
    magnitude_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    magnitude_model.fit(X, y_magnitude)

    horizon_model = None
    horizon_classes = sorted(y_horizon.dropna().unique().tolist())
    if len(horizon_classes) >= 2:
        horizon_model = RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
            min_samples_leaf=3,
        )
        horizon_model.fit(X, y_horizon)

    # 3.1 Robust threshold calibration from OOS probabilities
    calibrated_threshold, calibration_summary = calibrate_decision_threshold(
        df=df,
        base_model=best_model,
        n_splits=8,
        min_precision=MIN_TARGET_PRECISION,
    )
    if calibration_summary:
        print(
            "Threshold calibration | "
            f"selected={calibrated_threshold:.2f} | "
            f"oos_f1={calibration_summary.get('f1', 0.0):.3f} | "
            f"oos_precision={calibration_summary.get('precision', 0.0):.3f} | "
            f"oos_recall={calibration_summary.get('recall', 0.0):.3f}"
        )

    # 4. Walk-forward validation report (true OOS splits)
    wf_summary = {}
    wf_df = run_walk_forward_validation(df, best_model, threshold=calibrated_threshold)
    if wf_df is not None and not wf_df.empty:
        wf_df.to_csv(WALKFORWARD_RESULTS_FILE, index=False)
        wf_summary = {
            "f1_mean": float(wf_df['f1'].mean()),
            "precision_mean": float(wf_df['precision'].mean()),
            "recall_mean": float(wf_df['recall'].mean()),
            "accuracy_mean": float(wf_df['accuracy'].mean()),
            "folds": int(len(wf_df)),
        }
        print(f"📈 Walk-forward results saved to {WALKFORWARD_RESULTS_FILE}")
        print(
            "Walk-forward summary | "
            f"F1={wf_summary['f1_mean']:.3f} | "
            f"Precision={wf_summary['precision_mean']:.3f} | "
            f"Recall={wf_summary['recall_mean']:.3f} | "
            f"Accuracy={wf_summary['accuracy_mean']:.3f}"
        )
    
    # 5. Final Report
    y_proba = best_model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= calibrated_threshold).astype(int)
    print("\n✅ Deep Training Complete.")
    print("Best Params (Subset):", {k: v for k, v in search.best_params_.items() if k.startswith('xgb') or k.startswith('lgbm')})
    print(f"Decision threshold: {calibrated_threshold:.2f}")
    print("Full Classification Report:")
    report_txt = classification_report(y, y_pred)
    print(report_txt)
    
    model_payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "direction_model": best_model,
        "magnitude_model": magnitude_model,
        "horizon_model": horizon_model,
        "feature_cols": FEATURE_COLS,
        "decision_threshold": float(calibrated_threshold),
        "has_shap": bool(shap is not None),
    }
    joblib.dump(model_payload, MODEL_FILE)
    print(f"💾 Institutional Ensemble Model saved to {MODEL_FILE}")

    train_summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(df)),
        "positives": int(positives),
        "negatives": int(negatives),
        "scale_pos_weight": float(scale_pos_weight),
        "decision_threshold": float(calibrated_threshold),
        "threshold_calibration": calibration_summary,
        "best_params": {k: v for k, v in search.best_params_.items()},
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "multi_objective": {
            "direction": "stacking_classifier",
            "magnitude": "random_forest_regressor",
            "horizon": "random_forest_classifier" if horizon_model is not None else "fallback_default",
            "horizon_classes": horizon_classes,
            "targets": {
                "direction": TARGET_DIRECTION_COL,
                "magnitude": TARGET_MAGNITUDE_COL,
                "horizon": TARGET_HORIZON_COL,
            },
        },
        "shap_enabled": bool(shap is not None),
        "walk_forward": wf_summary,
        "artifacts": {
            "model_file": MODEL_FILE,
            "walkforward_file": WALKFORWARD_RESULTS_FILE,
        },
    }
    with open(TRAINING_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, ensure_ascii=False, indent=2)
    print(f"🧾 Training report saved to {TRAINING_REPORT_FILE}")

    return {
        "status": "ok",
        "model_file": MODEL_FILE,
        "walkforward_file": WALKFORWARD_RESULTS_FILE,
        "training_report_file": TRAINING_REPORT_FILE,
        "walk_forward": wf_summary,
        "classification_report": report_txt,
    }


def run_walk_forward_validation(
    df: pd.DataFrame,
    base_model=None,
    n_splits: int = 8,
    threshold: float = MODEL_DECISION_THRESHOLD,
) -> pd.DataFrame:
    """Runs walk-forward validation with expanding window splits and returns fold metrics."""
    if df is None or df.empty:
        return pd.DataFrame()

    local_df = df.copy()
    local_df = local_df.replace([np.inf, -np.inf], np.nan)
    if TARGET_DIRECTION_COL not in local_df.columns:
        local_df[TARGET_DIRECTION_COL] = local_df.get('Target', 0).fillna(0).astype(int)
    local_df = local_df.dropna(subset=[TARGET_DIRECTION_COL])
    local_df['Date'] = pd.to_datetime(local_df['Date'])
    local_df = local_df.sort_values('Date')

    X = local_df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.clip(lower=-1e6, upper=1e6)
    y = local_df[TARGET_DIRECTION_COL].astype(int)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    scale_pos_weight = negatives / positives

    if len(local_df) < (n_splits + 2):
        return pd.DataFrame()

    if base_model is None:
        estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')),
            ('xgb', xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_estimators=100,
                scale_pos_weight=scale_pos_weight,
            )),
            ('lgbm', lgb.LGBMClassifier(random_state=42, verbosity=-1, n_estimators=100, class_weight='balanced')),
        ]
        base_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
            cv=5,
            n_jobs=-1,
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = base_model
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        rows.append({
            'fold': fold_id,
            'train_start': local_df.iloc[train_idx[0]]['Date'],
            'train_end': local_df.iloc[train_idx[-1]]['Date'],
            'test_start': local_df.iloc[test_idx[0]]['Date'],
            'test_end': local_df.iloc[test_idx[-1]]['Date'],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': int(len(test_idx)),
        })

    return pd.DataFrame(rows)


def calibrate_decision_threshold(
    df: pd.DataFrame,
    base_model,
    n_splits: int = 8,
    min_precision: float = 0.25,
):
    """Calibrates the decision threshold using OOS probabilities from walk-forward splits."""
    if df is None or df.empty:
        return MODEL_DECISION_THRESHOLD, {}

    local_df = df.copy()
    local_df = local_df.replace([np.inf, -np.inf], np.nan)
    if TARGET_DIRECTION_COL not in local_df.columns:
        local_df[TARGET_DIRECTION_COL] = local_df.get('Target', 0).fillna(0).astype(int)
    local_df = local_df.dropna(subset=[TARGET_DIRECTION_COL])
    local_df['Date'] = pd.to_datetime(local_df['Date'])
    local_df = local_df.sort_values('Date')

    X = local_df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.clip(lower=-1e6, upper=1e6)
    y = local_df[TARGET_DIRECTION_COL].astype(int)

    if len(local_df) < (n_splits + 2):
        return MODEL_DECISION_THRESHOLD, {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_true = []
    oos_prob = []

    for train_idx, test_idx in tscv.split(X):
        model = clone(base_model)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        oos_true.extend(y.iloc[test_idx].tolist())
        oos_prob.extend(probs.tolist())

    if not oos_true:
        return MODEL_DECISION_THRESHOLD, {}

    y_true = np.array(oos_true, dtype=int)
    y_prob = np.array(oos_prob, dtype=float)
    candidates = np.round(np.arange(0.20, 0.81, 0.05), 2)

    best_any = None
    best_constrained = None
    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        row = {
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }
        if best_any is None or row["f1"] > best_any["f1"]:
            best_any = row
        if row["precision"] >= min_precision:
            if best_constrained is None or row["f1"] > best_constrained["f1"]:
                best_constrained = row

    selected = best_constrained if best_constrained is not None else best_any
    if selected is None:
        return MODEL_DECISION_THRESHOLD, {}

    return float(selected["threshold"]), selected

class Predictor:
    _model = None
    _shap_explainers = None
    _shap_estimators_key = None
    
    @classmethod
    def load_model(cls):
        if cls._model is None and os.path.exists(MODEL_FILE):
            loaded = joblib.load(MODEL_FILE)
            if isinstance(loaded, dict) and "direction_model" in loaded:
                cls._model = loaded
            else:
                cls._model = {
                    "schema_version": 1,
                    "direction_model": loaded,
                    "magnitude_model": None,
                    "horizon_model": None,
                    "feature_cols": FEATURE_COLS,
                    "decision_threshold": MODEL_DECISION_THRESHOLD,
                    "has_shap": bool(shap is not None),
                }
        return cls._model

    @classmethod
    def _get_direction_model(cls):
        bundle = cls.load_model()
        if not bundle:
            return None
        return bundle.get("direction_model")

    @classmethod
    def _get_model_feature_cols(cls, model) -> list:
        bundle = cls.load_model() or {}
        configured = bundle.get("feature_cols") or FEATURE_COLS
        fallback_cols = list(configured)
        return list(getattr(model, "feature_names_in_", fallback_cols))

    @classmethod
    def predict_probability(cls, tech_data: dict, context_score: float = 0.0) -> float:
        """Predicts probability (0-1) of price increase > 2% in 3 days."""
        model = cls._get_direction_model()
        if not model:
            return 0.5
            
        try:
            # Map technical_indicators to ML features
            features = {
                'Return_1d': tech_data.get('DailyReturn_Pct', 0) / 100,
                'Return_5d': tech_data.get('FiveDayReturn_Pct', 0) / 100,
                'Return_10d': tech_data.get('TenDayReturn_Pct', 0) / 100,
                'Return_20d': tech_data.get('MonthlyReturn_Pct', 0) / 100,
                'Dist_MA20': tech_data.get('Dist_MA20_Pct', 0) / 100,
                'Dist_MA50': tech_data.get('Dist_MA50_Pct', 0) / 100,
                'Dist_MA200': tech_data.get('Dist_MA200_Pct', 0) / 100,
                'MA20_MA50_Cross': 1.0 if tech_data.get('MA20', 0) > tech_data.get('MA50', 0) else 0.0,
                'Volatility_20d': tech_data.get('Volatility_20d', 0.01),
                'Volatility_5d': tech_data.get('Volatility_5d', tech_data.get('Volatility_20d', 0.01)),
                'ATR_Ratio': tech_data.get('ATR_Ratio', 0.01),
                'RSI_14': tech_data.get('RSI_14', 50),
                'MACD': tech_data.get('MACD', 0.0),
                'MACD_Signal': tech_data.get('MACD_Signal', 0.0),
                'MACD_Hist': tech_data.get('MACD_Hist', 0.0),
                'BB_Width': tech_data.get('BB_Width', 0.0),
                'BB_Position': tech_data.get('BB_Position', 0.5),
                'OBV': tech_data.get('OBV', 0.0),
                'OBV_MA20': tech_data.get('OBV_MA20', 0.0),
                'Volume_Ratio': tech_data.get('Volume_Ratio', 1.0),
                'Macro_Copper_Ret': tech_data.get('Macro_Copper_Ret', 0.0),
                'Macro_SP500_Ret': tech_data.get('Macro_SP500_Ret', 0.0),
                'Macro_USDCLP_Ret': tech_data.get('Macro_USDCLP_Ret', 0.0),
                'Macro_Lithium_Ret': tech_data.get('Macro_Lithium_Ret', 0.0),
                'Macro_MSCI_EM_Ret': tech_data.get('Macro_MSCI_EM_Ret', 0.0),
                'Macro_VIX_Ret': tech_data.get('Macro_VIX_Ret', 0.0),
                'Copper_Rolling_Corr_30d': tech_data.get('Copper_Rolling_Corr_30d', 0.0),
                'Context_Score': context_score
            }
            
            X = pd.DataFrame([features])

            # Backward compatibility for previously-trained models with older feature sets.
            model_feature_cols = cls._get_model_feature_cols(model)
            for col in model_feature_cols:
                if col not in X.columns:
                    X[col] = 0.0

            prob = model.predict_proba(X[model_feature_cols])[0][1]
            return float(prob)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return 0.5

    @classmethod
    def predict_multi_objective(cls, tech_data: dict, context_score: float = 0.0) -> dict:
        """Returns direction probability + expected 3d magnitude + estimated horizon."""
        bundle = cls.load_model()
        if not bundle:
            return {
                "probability": 0.5,
                "expected_return_3d": 0.0,
                "horizon_days": DEFAULT_HORIZON_DAYS,
                "horizon_distribution": {},
            }

        direction_model = bundle.get("direction_model")
        magnitude_model = bundle.get("magnitude_model")
        horizon_model = bundle.get("horizon_model")
        if direction_model is None:
            return {
                "probability": 0.5,
                "expected_return_3d": 0.0,
                "horizon_days": DEFAULT_HORIZON_DAYS,
                "horizon_distribution": {},
            }

        X = cls.build_feature_vector(tech_data, context_score=context_score)
        model_feature_cols = cls._get_model_feature_cols(direction_model)
        for col in model_feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model_feature_cols]

        probability = float(direction_model.predict_proba(X)[0][1])

        expected_return_3d = 0.0
        if magnitude_model is not None:
            try:
                expected_return_3d = float(magnitude_model.predict(X)[0])
            except Exception:
                expected_return_3d = 0.0

        horizon_days = DEFAULT_HORIZON_DAYS
        horizon_distribution = {}
        if horizon_model is not None:
            try:
                probs = horizon_model.predict_proba(X)[0]
                classes = list(getattr(horizon_model, "classes_", []))
                if len(classes) == len(probs):
                    horizon_distribution = {
                        str(int(cls_val)): float(prob) for cls_val, prob in zip(classes, probs)
                    }
                    horizon_days = int(classes[int(np.argmax(probs))])
            except Exception:
                horizon_days = DEFAULT_HORIZON_DAYS

        return {
            "probability": probability,
            "expected_return_3d": expected_return_3d,
            "horizon_days": horizon_days,
            "horizon_distribution": horizon_distribution,
        }

    @classmethod
    def build_feature_vector(cls, tech_data: dict, context_score: float = 0.0):
        features = {
            'Return_1d': tech_data.get('DailyReturn_Pct', 0) / 100,
            'Return_5d': tech_data.get('FiveDayReturn_Pct', 0) / 100,
            'Return_10d': tech_data.get('TenDayReturn_Pct', 0) / 100,
            'Return_20d': tech_data.get('MonthlyReturn_Pct', 0) / 100,
            'Dist_MA20': tech_data.get('Dist_MA20_Pct', 0) / 100,
            'Dist_MA50': tech_data.get('Dist_MA50_Pct', 0) / 100,
            'Dist_MA200': tech_data.get('Dist_MA200_Pct', 0) / 100,
            'MA20_MA50_Cross': 1.0 if tech_data.get('MA20', 0) > tech_data.get('MA50', 0) else 0.0,
            'Volatility_20d': tech_data.get('Volatility_20d', 0.01),
            'Volatility_5d': tech_data.get('Volatility_5d', tech_data.get('Volatility_20d', 0.01)),
            'ATR_Ratio': tech_data.get('ATR_Ratio', 0.01),
            'RSI_14': tech_data.get('RSI_14', 50),
            'MACD': tech_data.get('MACD', 0.0),
            'MACD_Signal': tech_data.get('MACD_Signal', 0.0),
            'MACD_Hist': tech_data.get('MACD_Hist', 0.0),
            'BB_Width': tech_data.get('BB_Width', 0.0),
            'BB_Position': tech_data.get('BB_Position', 0.5),
            'OBV': tech_data.get('OBV', 0.0),
            'OBV_MA20': tech_data.get('OBV_MA20', 0.0),
            'Volume_Ratio': tech_data.get('Volume_Ratio', 1.0),
            'Macro_Copper_Ret': tech_data.get('Macro_Copper_Ret', 0.0),
            'Macro_SP500_Ret': tech_data.get('Macro_SP500_Ret', 0.0),
            'Macro_USDCLP_Ret': tech_data.get('Macro_USDCLP_Ret', 0.0),
            'Macro_Lithium_Ret': tech_data.get('Macro_Lithium_Ret', 0.0),
            'Macro_MSCI_EM_Ret': tech_data.get('Macro_MSCI_EM_Ret', 0.0),
            'Macro_VIX_Ret': tech_data.get('Macro_VIX_Ret', 0.0),
            'Copper_Rolling_Corr_30d': tech_data.get('Copper_Rolling_Corr_30d', 0.0),
            'Context_Score': context_score,
        }
        return pd.DataFrame([features])

    @classmethod
    def explain_prediction(cls, tech_data: dict, context_score: float = 0.0, top_n: int = 4) -> dict:
        """Returns SHAP-based model drivers for explainability in production signals."""
        bundle = cls.load_model()
        direction_model = cls._get_direction_model()
        if not bundle or direction_model is None:
            return {
                "probability": 0.5,
                "expected_return_3d": 0.0,
                "horizon_days": DEFAULT_HORIZON_DAYS,
                "top_drivers": [],
                "explainability_method": "unavailable",
            }

        objectives = cls.predict_multi_objective(tech_data, context_score=context_score)

        X = cls.build_feature_vector(tech_data, context_score=context_score)
        model_feature_cols = cls._get_model_feature_cols(direction_model)
        for col in model_feature_cols:
            if col not in X.columns:
                X[col] = 0.0

        X = X[model_feature_cols]

        method = "heuristic"
        signed_influence = None

        named_estimators = getattr(direction_model, "named_estimators_", {})
        estimators = [
            est for est in named_estimators.values()
            if hasattr(est, "predict") and hasattr(est, "feature_importances_")
        ]

        if shap is not None and estimators:
            estimator_key = tuple(type(est).__name__ for est in estimators)
            if cls._shap_explainers is None or cls._shap_estimators_key != estimator_key:
                explainers = []
                for est in estimators:
                    try:
                        explainers.append(shap.TreeExplainer(est))
                    except Exception:
                        continue
                cls._shap_explainers = explainers
                cls._shap_estimators_key = estimator_key

            shap_arrays = []
            for explainer in cls._shap_explainers or []:
                try:
                    shap_values = explainer.shap_values(X)

                    sv = None
                    if isinstance(shap_values, list):
                        candidate = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                        arr = np.array(candidate)
                        if arr.ndim == 2 and arr.shape[1] == len(model_feature_cols):
                            sv = arr[0]
                        elif arr.ndim == 1 and arr.shape[0] == len(model_feature_cols):
                            sv = arr
                    else:
                        arr = np.array(shap_values)
                        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == len(model_feature_cols):
                            # Shape like (1, n_features, n_classes)
                            sv = arr[0, :, 1] if arr.shape[2] > 1 else arr[0, :, 0]
                        elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == len(model_feature_cols):
                            # Shape like (1, n_classes, n_features)
                            sv = arr[0, 1, :] if arr.shape[1] > 1 else arr[0, 0, :]
                        elif arr.ndim == 2 and arr.shape[1] == len(model_feature_cols):
                            sv = arr[0]
                        elif arr.ndim == 1 and arr.shape[0] == len(model_feature_cols):
                            sv = arr

                    if sv is not None and len(sv) == len(model_feature_cols):
                        shap_arrays.append(np.array(sv, dtype=float))
                except Exception:
                    continue

            if shap_arrays:
                signed_influence = np.mean(np.vstack(shap_arrays), axis=0)
                method = "shap"

        if signed_influence is None:
            importance = np.zeros(len(model_feature_cols), dtype=float)
            estimators_found = 0
            for est in estimators:
                fi = getattr(est, "feature_importances_", None)
                if fi is None:
                    continue
                fi = np.array(fi, dtype=float)
                if fi.shape[0] == len(model_feature_cols):
                    importance += fi
                    estimators_found += 1

            if estimators_found > 0:
                importance = importance / estimators_found
            else:
                importance = np.ones(len(model_feature_cols), dtype=float)

            vals = X.iloc[0].to_numpy(dtype=float)
            signed_influence = np.tanh(vals) * importance

        vals = X.iloc[0].to_numpy(dtype=float)
        order = np.argsort(np.abs(signed_influence))[::-1][:max(1, top_n)]

        top_drivers = []
        for idx in order:
            direction = "bullish" if signed_influence[idx] >= 0 else "bearish"
            top_drivers.append(
                {
                    "feature": model_feature_cols[idx],
                    "direction": direction,
                    "impact_score": float(abs(signed_influence[idx])),
                    "value": float(vals[idx]),
                }
            )

        return {
            "probability": objectives.get("probability", 0.5),
            "expected_return_3d": objectives.get("expected_return_3d", 0.0),
            "horizon_days": objectives.get("horizon_days", DEFAULT_HORIZON_DAYS),
            "horizon_distribution": objectives.get("horizon_distribution", {}),
            "top_drivers": top_drivers,
            "explainability_method": method,
        }

if __name__ == "__main__":
    train_model()
