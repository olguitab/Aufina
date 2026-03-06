import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import numpy as np
from paths import MODEL_FILE, PROCESSED_FEATURES_FILE, ensure_project_dirs

# Feature Set Definition
FEATURE_COLS = [
    'Return_1d', 'Return_5d', 'Dist_MA20', 'Volatility_20d', 
    'RSI_14', 'Macro_Copper_Ret', 'Macro_SP500_Ret', 'Macro_USDCLP_Ret', 
    'Context_Score'
]

def train_model():
    ensure_project_dirs()
    if not os.path.exists(PROCESSED_FEATURES_FILE):
        print("❌ No features found. Run features.py first.")
        return

    df = pd.read_csv(PROCESSED_FEATURES_FILE)
    df = df.dropna(subset=FEATURE_COLS + ['Target'])
    
    # Sort chronologically for TimeSeriesSplit
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    X = df[FEATURE_COLS]
    y = df['Target']
    
    print(f"🌲 Deep Quant Training on {len(df)} samples...")
    print("⏳ This process is exhaustive and will take several minutes...")
    
    # 1. TimeSeriesSplit (10 folds for maximum robustness)
    tscv = TimeSeriesSplit(n_splits=10)
    
    # 2. Define Base Models for Stacking
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lgbm', lgb.LGBMClassifier(random_state=42, verbosity=-1))
    ]
    
    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
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
    
    # 4. Final Report
    y_pred = best_model.predict(X)
    print("\n✅ Deep Training Complete.")
    print("Best Params (Subset):", {k: v for k, v in search.best_params_.items() if k.startswith('xgb') or k.startswith('lgbm')})
    print("Full Classification Report:")
    print(classification_report(y, y_pred))
    
    joblib.dump(best_model, MODEL_FILE)
    print(f"💾 Institutional Ensemble Model saved to {MODEL_FILE}")

class Predictor:
    _model = None
    
    @classmethod
    def load_model(cls):
        if cls._model is None and os.path.exists(MODEL_FILE):
            cls._model = joblib.load(MODEL_FILE)
        return cls._model

    @classmethod
    def predict_probability(cls, tech_data: dict, context_score: float = 0.0) -> float:
        """Predicts probability (0-1) of price increase > 2% in 3 days."""
        model = cls.load_model()
        if not model:
            return 0.5
            
        try:
            # Map technical_indicators to ML features
            features = {
                'Return_1d': tech_data.get('DailyReturn_Pct', 0) / 100,
                'Return_5d': tech_data.get('FiveDayReturn_Pct', 0) / 100,
                'Dist_MA20': tech_data.get('Dist_MA20_Pct', 0) / 100,
                'Volatility_20d': tech_data.get('Volatility_20d', 0.01),
                'RSI_14': tech_data.get('RSI_14', 50) / 100,
                'Macro_Copper_Ret': tech_data.get('Macro_Copper_Ret', 0.0),
                'Macro_SP500_Ret': tech_data.get('Macro_SP500_Ret', 0.0),
                'Macro_USDCLP_Ret': tech_data.get('Macro_USDCLP_Ret', 0.0),
                'Context_Score': context_score
            }
            
            X = pd.DataFrame([features])
            prob = model.predict_proba(X[FEATURE_COLS])[0][1]
            return float(prob)
        except Exception as e:
            print(f"Prediction Error: {e}")
            return 0.5

if __name__ == "__main__":
    train_model()
