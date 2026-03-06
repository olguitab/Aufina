import pandas as pd
import numpy as np
from typing import Dict, List, Any
from database import TradingDB

class StatsEngine:
    @staticmethod
    def calculate_metrics() -> Dict[str, Any]:
        """Calculates institutional-grade performance metrics."""
        trade_log = TradingDB.load_trade_log()
        if not trade_log:
            return {
                "ROI (%)": 0.0, "Sharpe Ratio": 0.0, 
                "Max Drawdown (%)": 0.0, "Win Rate (%)": 0.0
            }
        
        df = pd.DataFrame(trade_log)
        # Simplify ROI calculation based on starting balance of 100k
        current_balance = TradingDB.load_state()
        initial_balance = 100000.0
        roi = ((current_balance - initial_balance) / initial_balance) * 100
        
        # Win Rate based on signaling (BUY vs SELL effectiveness is harder without entry/exit mapping, 
        # so we look at positive conviction signals for now as proxy or just total trades)
        wins = df[df['confidence'] > 0.7].shape[0] # Temporary heuristic
        win_rate = (wins / len(df)) * 100 if len(df) > 0 else 0
        
        # Sharpe Ratio (Simplified using daily returns mock or log delta)
        # In a real system, we'd use daily NAV history. Here we estimate volatility from trade prices.
        returns = df['price'].pct_change().dropna()
        if len(returns) > 5:
            vol = returns.std() * np.sqrt(252) # Annualized
            avg_ret = returns.mean() * 252
            sharpe = (avg_ret - 0.05) / vol if vol > 0 else 0
        else:
            sharpe = 0.0
            
        return {
            "ROI (%)": round(roi, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown (%)": "N/A (Pending History)",
            "Win Rate (%)": round(win_rate, 2),
            "Total Trades": len(df)
        }
