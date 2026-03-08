import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from paths import HISTORICAL_FEATURES_FILE, PROCESSED_FEATURES_FILE, ensure_project_dirs
from universe import get_training_watchlist

FEATURE_FILE = HISTORICAL_FEATURES_FILE

# Macro Indicators to track
MACRO_INDICATORS = {
    'HG=F': 'Copper',
    '^GSPC': 'SP500',
    'CLP=X': 'USDCLP',
    'ALB': 'Lithium',
    'EEM': 'MSCI_EM',
    '^VIX': 'VIX'
}


def _compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_obv(close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
    direction = np.sign(close_series.diff()).fillna(0.0)
    return (direction * volume_series.fillna(0.0)).cumsum()

def download_historical_data(tickers: list, years: int = 10):
    """Downloads historical daily data for all tickers and macro indicators."""
    ensure_project_dirs()
    print(f"📥 Downloading {years} years of historical data for {len(tickers)} tickers + Macros...")
    
    all_data = []
    
    # Download Tickers
    for ticker in tickers:
        try:
            print(f"  → {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{years}y")
            if hist.empty: continue
            hist['Ticker'] = ticker
            hist = hist[['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            all_data.append(hist)
        except Exception as e:
            print(f"  ❌ Error downloading {ticker}: {e}")

    # Download Macros separately to align later
    macro_dfs = {}
    for sym, name in MACRO_INDICATORS.items():
        try:
            print(f"  → Macro: {name} ({sym})...")
            m_stock = yf.Ticker(sym)
            m_hist = m_stock.history(period=f"{years}y")
            if not m_hist.empty:
                # Normalize index for alignment
                m_hist.index = pd.to_datetime(m_hist.index).tz_localize(None).normalize()
                macro_dfs[name] = m_hist['Close']
        except Exception as e:
            print(f"  ❌ Error downloading macro {name}: {e}")
            
    if not all_data:
        print("⚠️ No data downloaded.")
        return None, {}
        
    df = pd.concat(all_data)
    df.to_csv(FEATURE_FILE)
    print(f"✅ Data saved to {FEATURE_FILE} ({len(df)} rows)")
    return df, macro_dfs

def generate_features(df: pd.DataFrame, macro_dfs: dict):
    """Generates an expanded feature set (momentum, trend, volatility, volume, macro, context)."""
    print("🧠 Generating features...")

    if 'Date' not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values(['Ticker', 'Date'])
    
    features = []
    
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        group = group.sort_values('Date').set_index('Date')

        # 1) Momentum
        group['Return_1d'] = group['Close'].pct_change(1)
        group['Return_5d'] = group['Close'].pct_change(5)
        group['Return_10d'] = group['Close'].pct_change(10)
        group['Return_20d'] = group['Close'].pct_change(20)

        # 2) Trend / MA distances
        group['MA20'] = group['Close'].rolling(window=20).mean()
        group['MA50'] = group['Close'].rolling(window=50).mean()
        group['MA200'] = group['Close'].rolling(window=200).mean()
        group['Dist_MA20'] = (group['Close'] - group['MA20']) / group['MA20']
        group['Dist_MA50'] = (group['Close'] - group['MA50']) / group['MA50']
        group['Dist_MA200'] = (group['Close'] - group['MA200']) / group['MA200']
        group['MA20_MA50_Cross'] = (group['MA20'] > group['MA50']).astype(int)

        # 3) Volatility and ATR
        group['Volatility_20d'] = group['Return_1d'].rolling(window=20).std()
        group['Volatility_5d'] = group['Return_1d'].rolling(window=5).std()

        high_low = group['High'] - group['Low']
        high_close = (group['High'] - group['Close'].shift()).abs()
        low_close = (group['Low'] - group['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        group['ATR_14'] = true_range.rolling(14).mean()
        group['ATR_Ratio'] = group['ATR_14'] / (group['Close'] + 1e-9)

        # 4) RSI / MACD / Bollinger
        group['RSI_14'] = _compute_rsi(group['Close'], 14)

        ema12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = ema12 - ema26
        group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
        group['MACD_Hist'] = group['MACD'] - group['MACD_Signal']

        bb_mid = group['Close'].rolling(window=20).mean()
        bb_std = group['Close'].rolling(window=20).std()
        group['BB_Upper'] = bb_mid + (2 * bb_std)
        group['BB_Lower'] = bb_mid - (2 * bb_std)
        group['BB_Width'] = (group['BB_Upper'] - group['BB_Lower']) / (bb_mid + 1e-9)
        group['BB_Position'] = (group['Close'] - group['BB_Lower']) / ((group['BB_Upper'] - group['BB_Lower']) + 1e-9)

        # 5) Volume features
        group['OBV'] = _compute_obv(group['Close'], group['Volume'])
        group['OBV_MA20'] = group['OBV'].rolling(window=20).mean()
        group['Volume_Ratio'] = group['Volume'] / (group['Volume'].rolling(window=20).mean() + 1e-9)

        # 6) Macro alignment + returns
        for name, m_sr in macro_dfs.items():
            group[f'Macro_{name}'] = group.index.map(m_sr)
            group[f'Macro_{name}'] = group[f'Macro_{name}'].ffill()
            group[f'Macro_{name}_Ret'] = group[f'Macro_{name}'].pct_change(1)

        # 7) Dynamic copper correlation (only if both series exist)
        if 'Macro_Copper' in group.columns:
            group['Copper_Rolling_Corr_30d'] = group['Close'].rolling(30).corr(group['Macro_Copper'])
        else:
            group['Copper_Rolling_Corr_30d'] = np.nan

        # 8) Context proxy for training (live runtime uses ContextService)
        context_proxy = pd.Series(0.0, index=group.index)
        if 'Macro_USDCLP_Ret' in group.columns:
            context_proxy += group['Macro_USDCLP_Ret'].rolling(window=5).mean() * -1
        if 'Macro_Copper_Ret' in group.columns:
            context_proxy += group['Macro_Copper_Ret'].rolling(window=5).mean()
        if 'Macro_VIX_Ret' in group.columns:
            context_proxy += group['Macro_VIX_Ret'].rolling(window=5).mean() * -0.5
        group['Context_Score'] = context_proxy.clip(-1, 1)

        # 9) Targets for multi-objective training
        # Direction target: Did it go up > 2% in the next 3 days?
        future_close_1d = group['Close'].shift(-1)
        future_close_3d = group['Close'].shift(-3)
        future_close_5d = group['Close'].shift(-5)

        ret_1d_fwd = (future_close_1d - group['Close']) / (group['Close'] + 1e-9)
        ret_3d_fwd = (future_close_3d - group['Close']) / (group['Close'] + 1e-9)
        ret_5d_fwd = (future_close_5d - group['Close']) / (group['Close'] + 1e-9)

        group['Target_Direction'] = (ret_3d_fwd > 0.02).astype(int)
        group['Target_Magnitude_3d'] = ret_3d_fwd.clip(lower=-0.30, upper=0.30)

        # Horizon label (days to meaningful move): 1, 3, 5
        cond_1d = ret_1d_fwd > 0.02
        cond_3d = ret_3d_fwd > 0.02
        cond_5d = ret_5d_fwd > 0.02
        group['Target_Horizon_Days'] = np.select(
            [cond_1d, cond_3d, cond_5d],
            [1, 3, 5],
            default=5,
        ).astype(int)

        # Backward compatibility with existing training scripts
        group['Target'] = group['Target_Direction']

        group = group.dropna()
        if not group.empty:
            group['Ticker'] = ticker
            group = group.reset_index()
        features.append(group)

    if not features:
        print("⚠️ No features generated after cleaning.")
        return pd.DataFrame()

    final_df = pd.concat(features, ignore_index=True)
    ensure_project_dirs()
    final_df.to_csv(PROCESSED_FEATURES_FILE)
    print(f"✅ Features generated. Saved to {PROCESSED_FEATURES_FILE} ({len(final_df)} samples)")
    return final_df

if __name__ == "__main__":
    training_watchlist = get_training_watchlist(include_global=True)
    raw_df, macros = download_historical_data(training_watchlist)
    if raw_df is not None:
        generate_features(raw_df, macros)
