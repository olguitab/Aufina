"""
Aureus Backtest Engine
Simulates the performance of the trading bot over the last 6 months.
Uses the trained Deep Quant ensemble and the dual-confirmation sell strategy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from models import Predictor, FEATURE_COLS
from features import IPSA_WATCHLIST, MACRO_INDICATORS
from paths import BACKTEST_RESULTS_FILE, BACKTEST_TRADES_FILE, MODEL_FILE, ensure_project_dirs

# --- Configuration ---
INITIAL_BALANCE = 10_000_000.0  # 10M CLP
TRADING_FEE = 0.003            # 0.3% per trade
SLIPPAGE = 0.001               # 0.1% slippage
START_DATE = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

class BacktestEngine:
    def __init__(self, watchlist: list = IPSA_WATCHLIST):
        self.watchlist = watchlist
        self.balance = INITIAL_BALANCE
        self.positions = {}        # {ticker: quantity}
        self.entry_prices = {}     # {ticker: price}
        self.peak_prices = {}      # {ticker: price}
        self.trade_history = []
        self.equity_curve = []
        self.last_known_prices = {}
        
        # Sizing / Strategy Params
        self.TRAILING_STOP_PCT = 0.05
        self.TAKE_PROFIT_PCT = 0.20    # Adjusted to 20% for faster rotation
        self.INVESTMENT_PER_TICKER_PCT = 0.50 
        self.BUY_THRESHOLD = 0.37      # Final High-Voltage Threshold

    def fetch_data(self):
        """Fetches 1 year of data to have enough for indicators in the 6-month window."""
        print(f"📥 Fetching historical data for {len(self.watchlist)} tickers...")
        all_dfs = {}
        
        # Fetching for 1 year to ensure RSI/MA buffers
        start_fetch = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        for ticker in self.watchlist:
            try:
                data = yf.download(ticker, start=start_fetch, end=END_DATE, interval="1d", progress=False)
                if not data.empty:
                    # Flatten multi-index if necessary (newer yfinance)
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    all_dfs[ticker] = data
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        # Fetch Macro data
        macros = {}
        for sym in MACRO_INDICATORS.keys():
            try:
                m_data = yf.download(sym, start=start_fetch, end=END_DATE, interval="1d", progress=False)
                if isinstance(m_data.columns, pd.MultiIndex):
                    m_data.columns = m_data.columns.get_level_values(0)
                macros[sym] = m_data['Close']
            except: pass
            
        return all_dfs, macros

    def generate_features_daily(self, prices_df: pd.DataFrame, macro_closes: dict):
        """Generates ML features for a specific stock dataframe."""
        df = prices_df.copy()
        
        # Momentum
        df['Return_1d'] = df['Close'].pct_change(1)
        df['Return_5d'] = df['Close'].pct_change(5)
        
        # Trend
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Dist_MA20'] = (df['Close'] - df['MA20']) / df['MA20']
        
        # Volatility
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Macro alignment
        for sym, name in MACRO_INDICATORS.items():
            if sym in macro_closes:
                df[f'Macro_{name}'] = macro_closes[sym]
                df[f'Macro_{name}'] = df[f'Macro_{name}'].ffill()
                df[f'Macro_{name}_Ret'] = df[f'Macro_{name}'].pct_change(1)
        
        df['Context_Score'] = 0.0 # Neutral context for backtest
        return df.dropna()

    def run(self):
        all_data, macros = self.fetch_data()
        
        # Prepare feature dataframes for all tickers
        ticker_features = {}
        for t in all_data:
            try:
                ticker_features[t] = self.generate_features_daily(all_data[t], macros)
            except Exception as e:
                print(f"Skipping {t} features due to error: {e}")

        # Get the unified timeline within the 6-month window
        timeline = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
        print(f"🚀 Starting simulation from {START_DATE} to {END_DATE}...")

        for current_date in timeline:
            # 1. Update Portfolio Valuation
            current_equity = self.balance
            for ticker, qty in self.positions.items():
                if qty > 0:
                    # Use current price if available, else last known peak (or entry)
                    if current_date in ticker_features[ticker].index:
                        price = ticker_features[ticker].loc[current_date, 'Close']
                        self.last_known_prices[ticker] = price
                    
                    price = self.last_known_prices.get(ticker, self.entry_prices.get(ticker, 0))
                    current_equity += qty * price
            self.equity_curve.append({'Date': current_date, 'Equity': current_equity})

            # 2. Check Exits (Trailing Stop + ML Confirmation)
            to_sell = []
            for ticker, qty in list(self.positions.items()):
                if qty <= 0: continue
                if current_date not in ticker_features[ticker].index: continue
                
                curr_price = ticker_features[ticker].loc[current_date, 'Close']
                avg_cost = self.entry_prices[ticker]
                
                # Update Peak Price
                self.peak_prices[ticker] = max(self.peak_prices.get(ticker, 0), curr_price)
                drop_from_peak = (curr_price - self.peak_prices[ticker]) / self.peak_prices[ticker]
                pnl_pct = (curr_price - avg_cost) / avg_cost
                
                # Case A: Hard Take Profit (+15%)
                if pnl_pct >= self.TAKE_PROFIT_PCT:
                    to_sell.append((ticker, curr_price, f"Take Profit (+{pnl_pct:.1%})"))
                
                # Case B: Smart Trailing Stop (-5% from peak + ML check)
                elif drop_from_peak <= -self.TRAILING_STOP_PCT:
                    # ML Check
                    row = ticker_features[ticker].loc[current_date]
                    X = pd.DataFrame([row[FEATURE_COLS]])
                    ml_prob = Predictor.load_model().predict_proba(X)[0][1]
                    
                    if ml_prob < 0.45:
                        to_sell.append((ticker, curr_price, f"Confirmed Trailing Stop (Peak Drop {drop_from_peak:.1%}, ML {ml_prob:.1%})"))
            
            for ticker, price, reason in to_sell:
                qty = self.positions[ticker]
                gross_val = qty * price
                fee = gross_val * (TRADING_FEE + SLIPPAGE)
                net_val = gross_val - fee
                
                self.balance += net_val
                pnl_clp = net_val - (qty * self.entry_prices[ticker])
                
                self.trade_history.append({
                    'Date': current_date, 'Ticker': ticker, 'Action': 'SELL',
                    'Price': price, 'Qty': qty, 'P&L CLP': pnl_clp, 'Reason': reason
                })
                self.positions[ticker] = 0
                print(f"[{current_date.date()}] 💸 SELL {ticker} @ {price:.0f} | {reason} | P&L: ${pnl_clp:,.0f}")

            # 3. Check Entries (Probability Scan)
            candidates = []
            for ticker in self.watchlist:
                if ticker in self.positions and self.positions[ticker] > 0: continue
                if ticker not in ticker_features or current_date not in ticker_features[ticker].index: continue
                
                row = ticker_features[ticker].loc[current_date]
                X = pd.DataFrame([row[FEATURE_COLS]])
                try:
                    ml_prob = Predictor.load_model().predict_proba(X)[0][1]
                except: ml_prob = 0.0
                
                if ml_prob > self.BUY_THRESHOLD:
                    candidates.append((ticker, ml_prob, row['Close']))
            
            # Buy best candidates with available balance
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:10] # Top 10 per day
            for ticker, prob, price in candidates:
                if self.balance > price:
                    invest_amount = self.balance * self.INVESTMENT_PER_TICKER_PCT
                    qty = int(invest_amount / (price * (1 + TRADING_FEE + SLIPPAGE)))
                    if qty > 0:
                        cost = qty * price * (1 + TRADING_FEE + SLIPPAGE)
                        self.balance -= cost
                        self.positions[ticker] = qty
                        self.entry_prices[ticker] = price
                        self.peak_prices[ticker] = price
                        
                        self.trade_history.append({
                            'Date': current_date, 'Ticker': ticker, 'Action': 'BUY',
                            'Price': price, 'Qty': qty, 'Prob': prob, 'Reason': 'ML Signal'
                        })
                        print(f"[{current_date.date()}] 🟢 BUY {ticker} @ {price:.0f} | Prob: {prob:.1%}")

        self.report()

    def report(self):
        ensure_project_dirs()
        print("\n\n" + "="*50)
        print("📊 BACKTEST PERFORMANCE REPORT (6 MONTHS)")
        print("="*50)
        
        final_equity = self.equity_curve[-1]['Equity'] if self.equity_curve else INITIAL_BALANCE
        total_roi = ((final_equity - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        # Benchmark vs IPSA (Safe fallback)
        ipsa_roi = 0.0
        try:
            ipsa = yf.download("^IPSA", start=START_DATE, end=END_DATE, progress=False)
            if not ipsa.empty:
                i_start = ipsa['Close'].iloc[0]
                i_end = ipsa['Close'].iloc[-1]
                v_start = float(i_start) if not isinstance(i_start, pd.Series) else float(i_start.iloc[0])
                v_end = float(i_end) if not isinstance(i_end, pd.Series) else float(i_end.iloc[0])
                ipsa_roi = ((v_end - v_start) / v_start) * 100
        except: pass

        print(f"Initial Capital: ${INITIAL_BALANCE:,.0f} CLP")
        print(f"Final Capital:   ${final_equity:,.0f} CLP")
        print(f"Total ROI:       {total_roi:.2f}%")
        print(f"IPSA ROI:        {ipsa_roi:.2f}% (Benchmark)")
        print(f"Alpha:           {total_roi - ipsa_roi:.2f}%")
        
        trades_df = pd.DataFrame(self.trade_history)
        if not trades_df.empty:
            sells = trades_df[trades_df['Action'] == 'SELL']
            win_rate = (len(sells[sells['P&L CLP'] > 0]) / len(sells)) * 100 if len(sells) > 0 else 0
            print(f"Total Trades:    {len(trades_df)}")
            print(f"Win Rate:        {win_rate:.1f}%")
            print(f"Avg P&L/Trade:   ${sells['P&L CLP'].mean():,.0f} CLP")
            trades_df.to_csv(BACKTEST_TRADES_FILE, index=False)
        
        # Save equity curve for UI
        pd.DataFrame(self.equity_curve).to_csv(BACKTEST_RESULTS_FILE, index=False)
        print(f"\n✅ Results saved to {BACKTEST_RESULTS_FILE} and {BACKTEST_TRADES_FILE}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Trained model not found. Backtest requires {MODEL_FILE}")
    else:
        engine = BacktestEngine()
        engine.run()
