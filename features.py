import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from paths import HISTORICAL_FEATURES_FILE, PROCESSED_FEATURES_FILE, ensure_project_dirs

# List of IPSA 30 tickers (Chile)
IPSA_WATCHLIST = [
    'CHILE.SN', 'SQM-B.SN', 'CENCOSUD.SN', 'ENELAM.SN', 'FALABELLA.SN', 'LTM.SN',
    'BCI.SN', 'BSANTANDER.SN', 'COPEC.SN', 'CMPC.SN', 'AGUAS-A.SN', 'PARAUCO.SN',
    'ANDINA-B.SN', 'VAPORES.SN', 'CCU.SN', 'IAM.SN', 'SECURITY.SN', 'MALLPLAZA.SN',
    'SONDA.SN', 'ENTEL.SN', 'SMU.SN', 'RIPLEY.SN', 'CAP.SN', 'ILC.SN',
    'CENCOSHOPP.SN', 'CONCHATORO.SN', 'ENELCHILE.SN', 'COLBUN.SN', 'ORO_BLANCO.SN', 'VSPT.SN'
]

# US Tech & Mega Caps (Market Expansion)
GLOBAL_WATCHLIST = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'AVGO'
]

FULL_WATCHLIST = IPSA_WATCHLIST + GLOBAL_WATCHLIST

FEATURE_FILE = HISTORICAL_FEATURES_FILE

# Macro Indicators to track
MACRO_INDICATORS = {
    'HG=F': 'Copper',
    '^GSPC': 'SP500',
    'CLP=X': 'USDCLP'
}

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
    """Generates technical indicators as features for ML."""
    print("🧠 Generating features...")
    
    # Sort by ticker and date
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_values(['Ticker', 'Date'])
    
    features = []
    
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()
        # Normalize index to naive date for alignment
        group.index = pd.to_datetime(group.index).tz_localize(None).normalize()
        
        # 1. Momentum: Returns
        group['Return_1d'] = group['Close'].pct_change(1)
        group['Return_5d'] = group['Close'].pct_change(5)
        
        # 2. Moving Averages & Trend
        group['MA20'] = group['Close'].rolling(window=20).mean()
        group['MA50'] = group['Close'].rolling(window=50).mean()
        group['MA200'] = group['Close'].rolling(window=200).mean()
        group['Dist_MA20'] = (group['Close'] - group['MA20']) / group['MA20']
        
        # 3. Volatility
        group['Volatility_20d'] = group['Return_1d'].rolling(window=20).std()
        
        # 4. RSI (14)
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        group['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 5. Macros Alignment
        for name, m_sr in macro_dfs.items():
            # Align macro data to the ticker's dates
            group[f'Macro_{name}'] = group.index.map(m_sr)
            group[f'Macro_{name}'] = group[f'Macro_{name}'].ffill()
            # Calculate macro returns as features
            group[f'Macro_{name}_Ret'] = group[f'Macro_{name}'].pct_change(1)

        # 6. Context Score (Pro-level Sentiment Backfill)
        # In a full-scale institutional system, this would fetch from a database of analyzed news.
        # For this robust framework, we'll implement a correlation-based backfill for training.
        group['Context_Score'] = group['Macro_USDCLP_Ret'].rolling(window=5).mean() * -1 # USDCLP up is usually bad for CLP assets
        group['Context_Score'] += group['Macro_Copper_Ret'].rolling(window=5).mean()      # Copper up is good
        group['Context_Score'] = group['Context_Score'].clip(-1, 1)
        
        # 7. Target: Did it go up > 2% in the next 3 days?
        future_close = group['Close'].shift(-3)
        group['Target'] = ((future_close - group['Close']) / group['Close'] > 0.02).astype(int)
        
        # Drop rows with NaN
        group = group.dropna()
        features.append(group)
        
    final_df = pd.concat(features)
    ensure_project_dirs()
    final_df.to_csv(PROCESSED_FEATURES_FILE)
    print(f"✅ Features generated. Saved to {PROCESSED_FEATURES_FILE} ({len(final_df)} samples)")
    return final_df

if __name__ == "__main__":
    raw_df, macros = download_historical_data(FULL_WATCHLIST)
    if raw_df is not None:
        generate_features(raw_df, macros)
