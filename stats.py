from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from database import TradingDB


class StatsEngine:
    @staticmethod
    def _to_nav_df(portfolio: str = "paper") -> pd.DataFrame:
        nav = TradingDB.load_nav_history(portfolio=portfolio, limit=5000)
        if not nav:
            return pd.DataFrame(columns=["timestamp", "equity"])
        df = pd.DataFrame(nav)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
        return df

    @staticmethod
    def _daily_returns_from_nav(nav_df: pd.DataFrame) -> pd.Series:
        if nav_df.empty:
            return pd.Series(dtype=float)
        daily = nav_df.set_index("timestamp")["equity"].resample("1D").last().ffill().dropna()
        return daily.pct_change().dropna()

    @staticmethod
    def _max_drawdown(nav_df: pd.DataFrame) -> float:
        if nav_df.empty:
            return 0.0
        eq = nav_df.set_index("timestamp")["equity"].astype(float)
        running_peak = eq.cummax()
        dd = (eq / (running_peak + 1e-9)) - 1.0
        return float(dd.min()) if not dd.empty else 0.0

    @staticmethod
    def _profit_factor(trades_df: pd.DataFrame, predictions_df: pd.DataFrame = None) -> float:
        if trades_df.empty:
            if predictions_df is None or predictions_df.empty:
                return 0.0
            resolved = predictions_df[predictions_df["resolved"] == 1].copy()
            if resolved.empty:
                return 0.0
            rr = pd.to_numeric(resolved["realized_return"], errors="coerce").fillna(0.0)
            gross_profit = float(rr[rr > 0].sum())
            gross_loss = float(abs(rr[rr < 0].sum()))
            if gross_loss <= 0:
                return float("inf") if gross_profit > 0 else 0.0
            return gross_profit / gross_loss
        sells = trades_df[trades_df["signal"].str.upper() == "SELL"].copy()
        if sells.empty:
            if predictions_df is None or predictions_df.empty:
                return 0.0
            resolved = predictions_df[predictions_df["resolved"] == 1].copy()
            if resolved.empty:
                return 0.0
            rr = pd.to_numeric(resolved["realized_return"], errors="coerce").fillna(0.0)
            gross_profit = float(rr[rr > 0].sum())
            gross_loss = float(abs(rr[rr < 0].sum()))
            if gross_loss <= 0:
                return float("inf") if gross_profit > 0 else 0.0
            return gross_profit / gross_loss
        if "metadata" in sells.columns:
            pnl = pd.to_numeric(sells["metadata"].apply(lambda x: (x or {}).get("pnl_clp", 0.0)), errors="coerce").fillna(0.0)
        else:
            pnl = pd.Series([0.0] * len(sells))
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(abs(pnl[pnl < 0].sum()))
        if gross_loss <= 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _alpha_vs_ipsa(nav_returns: pd.Series) -> float:
        if nav_returns.empty:
            return 0.0
        start = nav_returns.index.min().strftime("%Y-%m-%d")
        end = (nav_returns.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        benchmark = None
        for symbol in ["^IPSA", "SPIPSA.SN"]:
            try:
                data = yf.download(symbol, start=start, end=end, progress=False)
                if data is not None and not data.empty and "Close" in data.columns:
                    benchmark = data["Close"].pct_change().dropna()
                    break
            except Exception:
                continue

        if benchmark is None or benchmark.empty:
            return 0.0

        aligned = pd.concat([nav_returns.rename("nav"), benchmark.rename("bench")], axis=1).dropna()
        if aligned.empty:
            return 0.0
        alpha_daily = aligned["nav"].mean() - aligned["bench"].mean()
        return float(alpha_daily * 252.0)

    @staticmethod
    def calculate_metrics(portfolio: str = "paper") -> Dict[str, Any]:
        nav_df = StatsEngine._to_nav_df(portfolio=portfolio)
        returns = StatsEngine._daily_returns_from_nav(nav_df)

        trade_log = TradingDB.load_trade_log()
        trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(columns=["signal"])
        if not trades_df.empty:
            trades_df["signal"] = trades_df["signal"].astype(str)

        if nav_df.empty:
            return {
                "ROI (%)": 0.0,
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "Max Drawdown (%)": 0.0,
                "Profit Factor": 0.0,
                "Alpha vs IPSA (%)": 0.0,
                "Total Trades": int(len(trades_df)),
                "Prediction Accuracy T+3 (%)": 0.0,
            }

        start_equity = float(nav_df["equity"].iloc[0])
        end_equity = float(nav_df["equity"].iloc[-1])
        roi = ((end_equity - start_equity) / (start_equity + 1e-9)) * 100.0

        rf_annual = 0.03
        rf_daily = rf_annual / 252.0
        excess = returns - rf_daily

        sharpe = 0.0
        if len(excess) >= 2 and float(excess.std()) > 0:
            sharpe = float((excess.mean() / excess.std()) * np.sqrt(252.0))

        downside = excess[excess < 0]
        sortino = 0.0
        if len(downside) >= 2 and float(downside.std()) > 0:
            sortino = float((excess.mean() / downside.std()) * np.sqrt(252.0))

        max_dd = StatsEngine._max_drawdown(nav_df) * 100.0
        preds = TradingDB.load_predictions(limit=5000)
        pred_df = pd.DataFrame(preds) if preds else pd.DataFrame()

        profit_factor = StatsEngine._profit_factor(trades_df, pred_df)
        alpha_annual = StatsEngine._alpha_vs_ipsa(returns) * 100.0
        pred_acc = 0.0
        if not pred_df.empty:
            resolved = pred_df[pred_df["resolved"] == 1].copy()
            if not resolved.empty:
                resolved["pred_label"] = (pd.to_numeric(resolved["predicted_prob"], errors="coerce") >= 0.5).astype(int)
                resolved["realized_label"] = pd.to_numeric(resolved["realized_label"], errors="coerce").fillna(0).astype(int)
                pred_acc = float((resolved["pred_label"] == resolved["realized_label"]).mean() * 100.0)

        return {
            "ROI (%)": round(roi, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Max Drawdown (%)": round(max_dd, 2),
            "Profit Factor": round(float(profit_factor), 3) if np.isfinite(profit_factor) else "inf",
            "Alpha vs IPSA (%)": round(alpha_annual, 2),
            "Total Trades": int(len(trades_df)),
            "Prediction Accuracy T+3 (%)": round(pred_acc, 2),
            "Updated At": datetime.utcnow().isoformat() + "Z",
        }
