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

# V2 Extended Macro — additional global context
MACRO_INDICATORS_V2 = {
    'CL=F': 'WTI_Oil',
    'DX-Y.NYB': 'DXY',
    'TIO=F': 'IronOre',
    '^IPSA': 'IPSA',
    'GC=F': 'Gold',
}


def _compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_stochastic_rsi(close_series: pd.Series, rsi_window: int = 14, stoch_window: int = 14) -> pd.Series:
    """Stochastic RSI — RSI normalized to its own range."""
    rsi = _compute_rsi(close_series, rsi_window)
    rsi_min = rsi.rolling(stoch_window).min()
    rsi_max = rsi.rolling(stoch_window).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)


def _compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R — momentum oscillator (-100 to 0)."""
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    return -100 * (highest - close) / (highest - lowest + 1e-9)


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — trend strength (0-100)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window).mean() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(window).mean()


def _compute_obv(close_series: pd.Series, volume_series: pd.Series) -> pd.Series:
    direction = np.sign(close_series.diff()).fillna(0.0)
    return (direction * volume_series.fillna(0.0)).cumsum()


def _compute_max_drawdown_forward(close_series: pd.Series, window: int) -> pd.Series:
    """For each row, compute the max drawdown experienced during the next `window` days."""
    result = pd.Series(np.nan, index=close_series.index)
    values = close_series.values
    n = len(values)
    for i in range(n - window):
        entry = values[i]
        if entry <= 0:
            continue
        future_slice = values[i + 1: i + 1 + window]
        peak = entry
        max_dd = 0.0
        for p in future_slice:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd
        result.iloc[i] = max_dd
    return result


def _compute_optimal_exit_day(close_series: pd.Series, max_horizon: int = 20) -> pd.Series:
    """For each row, find the day (1..max_horizon) that maximizes risk-adjusted return."""
    result = pd.Series(np.nan, index=close_series.index)
    values = close_series.values
    n = len(values)
    for i in range(n - max_horizon):
        entry = values[i]
        if entry <= 0:
            continue
        best_score = -999.0
        best_day = max_horizon
        peak = entry
        for d in range(1, max_horizon + 1):
            future_price = values[i + d]
            ret = (future_price - entry) / entry
            # Track max drawdown up to this day
            peak = max(peak, future_price)
            dd = (peak - future_price) / peak if peak > 0 else 0
            # Sharpe-like score: return / (drawdown + small constant)
            score = ret / (dd + 0.005)
            if score > best_score:
                best_score = score
                best_day = d
        result.iloc[i] = best_day
    return result


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
    all_macros = {**MACRO_INDICATORS, **MACRO_INDICATORS_V2}
    for sym, name in all_macros.items():
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
    """Generates an expanded V2 feature set (momentum, trend, volatility, volume, macro, context, seasonality)."""
    print("🧠 Generating V2 features (60+ columns)...")

    if 'Date' not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values(['Ticker', 'Date'])
    
    features = []

    # Pre-compute sector returns for relative momentum
    sector_returns = {}
    
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        group = group.sort_values('Date').set_index('Date')

        # =============================================
        # 1) Momentum — multi-timeframe
        # =============================================
        group['Return_1d'] = group['Close'].pct_change(1)
        group['Return_5d'] = group['Close'].pct_change(5)
        group['Return_10d'] = group['Close'].pct_change(10)
        group['Return_20d'] = group['Close'].pct_change(20)
        group['Return_40d'] = group['Close'].pct_change(40)
        group['Return_60d'] = group['Close'].pct_change(60)

        # =============================================
        # 2) Trend / MA distances
        # =============================================
        group['MA20'] = group['Close'].rolling(window=20).mean()
        group['MA50'] = group['Close'].rolling(window=50).mean()
        group['MA200'] = group['Close'].rolling(window=200).mean()
        group['Dist_MA20'] = (group['Close'] - group['MA20']) / group['MA20']
        group['Dist_MA50'] = (group['Close'] - group['MA50']) / group['MA50']
        group['Dist_MA200'] = (group['Close'] - group['MA200']) / group['MA200']
        group['MA20_MA50_Cross'] = (group['MA20'] > group['MA50']).astype(int)

        # 52-week high/low distance
        group['High_52w'] = group['Close'].rolling(window=252).max()
        group['Low_52w'] = group['Close'].rolling(window=252).min()
        group['Dist_52w_High'] = (group['Close'] - group['High_52w']) / (group['High_52w'] + 1e-9)
        group['Dist_52w_Low'] = (group['Close'] - group['Low_52w']) / (group['Low_52w'] + 1e-9)

        # =============================================
        # 3) Volatility and ATR
        # =============================================
        group['Volatility_20d'] = group['Return_1d'].rolling(window=20).std()
        group['Volatility_5d'] = group['Return_1d'].rolling(window=5).std()
        group['Vol_Compression'] = group['Volatility_5d'] / (group['Volatility_20d'] + 1e-9)

        high_low = group['High'] - group['Low']
        high_close = (group['High'] - group['Close'].shift()).abs()
        low_close = (group['Low'] - group['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        group['ATR_14'] = true_range.rolling(14).mean()
        group['ATR_Ratio'] = group['ATR_14'] / (group['Close'] + 1e-9)

        # =============================================
        # 4) RSI / Stochastic RSI / Williams %R / MACD / Bollinger / ADX
        # =============================================
        group['RSI_14'] = _compute_rsi(group['Close'], 14)
        group['StochRSI'] = _compute_stochastic_rsi(group['Close'], 14, 14)
        group['WilliamsR'] = _compute_williams_r(group['High'], group['Low'], group['Close'], 14)
        group['ADX_14'] = _compute_adx(group['High'], group['Low'], group['Close'], 14)

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

        # =============================================
        # 5) Volume features
        # =============================================
        group['OBV'] = _compute_obv(group['Close'], group['Volume'])
        group['OBV_MA20'] = group['OBV'].rolling(window=20).mean()
        group['OBV_ROC'] = group['OBV'].pct_change(10)
        group['Volume_Ratio'] = group['Volume'] / (group['Volume'].rolling(window=20).mean() + 1e-9)

        # =============================================
        # 6) Macro alignment + returns (V1 + V2)
        # =============================================
        for name, m_sr in macro_dfs.items():
            group[f'Macro_{name}'] = group.index.map(m_sr)
            group[f'Macro_{name}'] = group[f'Macro_{name}'].ffill()
            group[f'Macro_{name}_Ret'] = group[f'Macro_{name}'].pct_change(1)

        # =============================================
        # 7) Dynamic inter-market correlations (20d rolling)
        # =============================================
        if 'Macro_Copper' in group.columns:
            group['Copper_Rolling_Corr_30d'] = group['Close'].rolling(30).corr(group['Macro_Copper'])
        else:
            group['Copper_Rolling_Corr_30d'] = np.nan

        for corr_name in ['WTI_Oil', 'Gold', 'DXY', 'IronOre', 'IPSA']:
            col = f'Macro_{corr_name}'
            if col in group.columns:
                group[f'{corr_name}_Corr_20d'] = group['Close'].rolling(20).corr(group[col])
            else:
                group[f'{corr_name}_Corr_20d'] = np.nan

        # =============================================
        # 8) Seasonality encoding
        # =============================================
        group['DayOfWeek'] = group.index.dayofweek / 4.0  # 0-1 normalized
        group['MonthOfYear'] = group.index.month / 12.0   # 0-1 normalized
        group['IsMonday'] = (group.index.dayofweek == 0).astype(float)
        group['IsFriday'] = (group.index.dayofweek == 4).astype(float)

        # =============================================
        # 9) Context proxy for training
        # =============================================
        context_proxy = pd.Series(0.0, index=group.index)
        if 'Macro_USDCLP_Ret' in group.columns:
            context_proxy += group['Macro_USDCLP_Ret'].rolling(window=5).mean() * -1
        if 'Macro_Copper_Ret' in group.columns:
            context_proxy += group['Macro_Copper_Ret'].rolling(window=5).mean()
        if 'Macro_VIX_Ret' in group.columns:
            context_proxy += group['Macro_VIX_Ret'].rolling(window=5).mean() * -0.5
        if 'Macro_WTI_Oil_Ret' in group.columns:
            context_proxy += group['Macro_WTI_Oil_Ret'].rolling(window=5).mean() * 0.3
        if 'Macro_DXY_Ret' in group.columns:
            context_proxy += group['Macro_DXY_Ret'].rolling(window=5).mean() * -0.4
        group['Context_Score'] = context_proxy.clip(-1, 1)

        # =============================================
        # 10) V1 Targets (backward compatibility)
        # =============================================
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

        # Backward compatibility
        group['Target'] = group['Target_Direction']

        # =============================================
        # 11) V2 Multi-Horizon Targets
        # =============================================
        future_close_10d = group['Close'].shift(-10)
        future_close_20d = group['Close'].shift(-20)

        ret_10d_fwd = (future_close_10d - group['Close']) / (group['Close'] + 1e-9)
        ret_20d_fwd = (future_close_20d - group['Close']) / (group['Close'] + 1e-9)

        # Direction targets per horizon
        group['Target_Direction_5d'] = (ret_5d_fwd > 0.02).astype(int)
        group['Target_Direction_10d'] = (ret_10d_fwd > 0.03).astype(int)
        group['Target_Direction_20d'] = (ret_20d_fwd > 0.05).astype(int)

        # Magnitude targets per horizon
        group['Target_Magnitude_5d'] = ret_5d_fwd.clip(lower=-0.30, upper=0.30)
        group['Target_Magnitude_10d'] = ret_10d_fwd.clip(lower=-0.40, upper=0.40)
        group['Target_Magnitude_20d'] = ret_20d_fwd.clip(lower=-0.50, upper=0.50)

        # Max drawdown during holding period (for exit model)
        group['Target_MaxDD_5d'] = _compute_max_drawdown_forward(group['Close'], 5)
        group['Target_MaxDD_10d'] = _compute_max_drawdown_forward(group['Close'], 10)
        group['Target_MaxDD_20d'] = _compute_max_drawdown_forward(group['Close'], 20)

        # Optimal exit day (maximizes risk-adjusted return within 20d)
        group['Target_OptimalExit'] = _compute_optimal_exit_day(group['Close'], 20)

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
    print(f"✅ V2 Features generated. Saved to {PROCESSED_FEATURES_FILE} ({len(final_df)} samples, {len(final_df.columns)} columns)")
    return final_df

if __name__ == "__main__":
    training_watchlist = get_training_watchlist(include_global=True)
    raw_df, macros = download_historical_data(training_watchlist)
    if raw_df is not None:
        generate_features(raw_df, macros)

