"""
Aureus V2 Backtest Engine — Multi-Horizon Validation
=====================================================
Simulates trading performance using V2 multi-horizon predictions.
Tests holding periods of 5, 10, and 20 days against IPSA benchmark.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from paths import BACKTEST_RESULTS_FILE, DATA_DIR, ensure_project_dirs

try:
    from models_v2 import MultiHorizonPredictor
    HAS_V2 = MultiHorizonPredictor.is_available()
except ImportError:
    HAS_V2 = False

from models import Predictor
from features import generate_features, download_historical_data, PROCESSED_FEATURES_FILE
from universe import get_training_watchlist

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Multi-Horizon Backtest: simulates trading with V2 model over historical data.
    
    Strategy:
    - Entry: V2 composite_signal == "BUY" with best_probability >= threshold
    - Holding: Use suggested_hold_days from V2; hold for at least min_hold_days
    - Exit: Take profit, trailing stop after protection, or hold period expiry
    - Position sizing: Equal weight across concurrent positions
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        buy_threshold: float = 0.50,
        max_positions: int = 8,
        take_profit_pct: float = 0.12,
        trailing_stop_pct: float = 0.05,
        min_hold_days: int = 1,
        backtest_months: int = 6,
    ):
        self.initial_capital = initial_capital
        self.buy_threshold = buy_threshold
        self.max_positions = max_positions
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_hold_days = min_hold_days
        self.backtest_months = backtest_months

    def run(self):
        """Execute multi-horizon backtest."""
        ensure_project_dirs()

        # Load features
        if not os.path.exists(PROCESSED_FEATURES_FILE):
            logger.info("Generating features for backtest...")
            watchlist = get_training_watchlist(include_global=True)
            raw_df, macros = download_historical_data(watchlist, years=10)
            if raw_df is None:
                logger.error("Failed to download data for backtest.")
                return None
            generate_features(raw_df, macros)

        df = pd.read_csv(PROCESSED_FEATURES_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Define backtest window
        end_date = df['Date'].max()
        start_date = end_date - timedelta(days=self.backtest_months * 30)
        logger.info(f"📊 Backtest: {start_date.date()} → {end_date.date()} ({self.backtest_months} months)")

        # Filter to backtest window
        bt_df = df[df['Date'] >= start_date].copy()
        dates = sorted(bt_df['Date'].unique())

        # Portfolio state
        cash = self.initial_capital
        positions = {}  # ticker -> {qty, entry_price, entry_date, peak, hold_target}
        trade_log = []
        equity_curve = []

        # Track IPSA benchmark
        ipsa_start = None
        ipsa_col = 'Macro_IPSA' if 'Macro_IPSA' in bt_df.columns else None

        for date in dates:
            day_data = bt_df[bt_df['Date'] == date]
            tickers_today = day_data['Ticker'].unique()

            # Track IPSA
            if ipsa_col and ipsa_start is None:
                row0 = day_data.iloc[0]
                val = row0.get(ipsa_col, 0)
                if pd.notna(val) and float(val) > 0:
                    ipsa_start = float(val)

            # Check exits for existing positions
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                ticker_row = day_data[day_data['Ticker'] == ticker]

                if ticker_row.empty:
                    continue

                current_price = float(ticker_row['Close'].iloc[0])
                entry_price = pos['entry_price']
                days_held = (date - pos['entry_date']).days
                pnl_pct = (current_price - entry_price) / entry_price
                pos['peak'] = max(pos['peak'], current_price)
                drop_from_peak = (current_price - pos['peak']) / pos['peak']
                hold_target = pos.get('hold_target', 10)

                sell_reason = None

                # Take profit
                if pnl_pct >= self.take_profit_pct:
                    sell_reason = f"Take Profit +{pnl_pct:.1%}"

                # Hold period expired
                elif days_held >= hold_target:
                    sell_reason = f"Hold period expired ({days_held}d >= {hold_target}d), P&L={pnl_pct:.1%}"

                # Trailing stop (only after min hold)
                elif days_held >= self.min_hold_days and drop_from_peak <= -self.trailing_stop_pct:
                    # Protection window: first half of hold target
                    protection_days = max(self.min_hold_days, hold_target // 2)
                    if days_held >= protection_days:
                        sell_reason = f"Trailing stop {drop_from_peak:.1%} from peak after {days_held}d"

                if sell_reason:
                    proceeds = pos['qty'] * current_price
                    cash += proceeds
                    pnl = (current_price - entry_price) * pos['qty']
                    trade_log.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'price': current_price,
                        'qty': pos['qty'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'days_held': days_held,
                        'hold_target': hold_target,
                        'reason': sell_reason,
                    })
                    del positions[ticker]

            # Check entries
            if len(positions) < self.max_positions and cash > 0:
                candidates = []
                for _, row in day_data.iterrows():
                    ticker = row['Ticker']
                    if ticker in positions:
                        continue

                    # Build feature dict for prediction
                    feature_dict = {}
                    for col in day_data.columns:
                        if col not in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']:
                            if not col.startswith('Target'):
                                val = row.get(col, 0)
                                feature_dict[col] = float(val) if pd.notna(val) else 0.0

                    # Get V2 prediction
                    if HAS_V2:
                        pred = MultiHorizonPredictor.predict(feature_dict, context_score=feature_dict.get('Context_Score', 0))
                        signal = pred.get('composite_signal', 'HOLD')
                        prob = pred.get('best_probability', 0.0)
                        hold_days = pred.get('suggested_hold_days', 10)
                    else:
                        prob = Predictor.predict_probability(feature_dict)
                        signal = 'BUY' if prob >= self.buy_threshold else 'HOLD'
                        hold_days = 5

                    if signal == 'BUY' and prob >= self.buy_threshold:
                        candidates.append({
                            'ticker': ticker,
                            'prob': prob,
                            'hold_days': hold_days,
                            'price': float(row['Close']),
                        })

                # Sort by probability, take best
                candidates.sort(key=lambda x: x['prob'], reverse=True)
                slots = self.max_positions - len(positions)

                for cand in candidates[:slots]:
                    if cash <= 0:
                        break
                    alloc = cash / (slots - len([c for c in candidates[:slots] if c['ticker'] in positions]))
                    alloc = min(alloc, cash * 0.5)  # Max 50% per position
                    qty = int(alloc / cand['price'])
                    if qty <= 0:
                        continue

                    cost = qty * cand['price']
                    if cost > cash:
                        qty = int(cash / cand['price'])
                        cost = qty * cand['price']
                    if qty <= 0:
                        continue

                    cash -= cost
                    positions[cand['ticker']] = {
                        'qty': qty,
                        'entry_price': cand['price'],
                        'entry_date': date,
                        'peak': cand['price'],
                        'hold_target': cand['hold_days'],
                    }
                    trade_log.append({
                        'date': date,
                        'ticker': cand['ticker'],
                        'action': 'BUY',
                        'price': cand['price'],
                        'qty': qty,
                        'pnl': 0,
                        'pnl_pct': 0,
                        'days_held': 0,
                        'hold_target': cand['hold_days'],
                        'reason': f"V2 signal prob={cand['prob']:.2%}",
                    })

            # Record equity
            invested = 0
            for tk, pos in positions.items():
                p_row = day_data[day_data['Ticker'] == tk]
                if not p_row.empty:
                    invested += pos['qty'] * float(p_row['Close'].iloc[0])
                else:
                    invested += pos['qty'] * pos['entry_price'] # fallback to entry if missing data today

            equity = cash + invested

            ipsa_value = 0.0
            if ipsa_col:
                row0 = day_data.iloc[0]
                val = row0.get(ipsa_col, None)
                if pd.notna(val) and ipsa_start and float(val) > 0:
                    ipsa_value = (float(val) / ipsa_start - 1)

            equity_curve.append({
                'date': date,
                'equity': equity,
                'cash': cash,
                'invested': invested,
                'n_positions': len(positions),
                'ipsa_return': ipsa_value,
            })

        # Generate report
        report = self._generate_report(equity_curve, trade_log)
        return report

    def _generate_report(self, equity_curve: list, trade_log: list) -> dict:
        """Generate comprehensive backtest report."""
        if not equity_curve:
            return {"status": "error", "reason": "no_data"}

        ec_df = pd.DataFrame(equity_curve)
        tl_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

        initial = self.initial_capital
        final = ec_df['equity'].iloc[-1]
        
        # Total profit from trades
        if not tl_df.empty:
            total_pnl = tl_df[tl_df['action'] == 'SELL']['pnl'].sum()
        else:
            total_pnl = 0
            
        total_return = (final - initial) / initial

        # Trade statistics
        sells = tl_df[tl_df['action'] == 'SELL'] if not tl_df.empty else pd.DataFrame()
        n_trades = len(sells)
        n_wins = len(sells[sells['pnl'] > 0]) if not sells.empty else 0
        n_losses = len(sells[sells['pnl'] <= 0]) if not sells.empty else 0
        win_rate = n_wins / max(n_trades, 1)
        avg_pnl_pct = float(sells['pnl_pct'].mean()) if not sells.empty else 0
        avg_hold_days = float(sells['days_held'].mean()) if not sells.empty else 0

        # Max drawdown
        ec_df['peak_equity'] = ec_df['equity'].cummax()
        ec_df['drawdown'] = (ec_df['equity'] - ec_df['peak_equity']) / ec_df['peak_equity']
        max_drawdown = float(ec_df['drawdown'].min())

        # IPSA comparison
        ipsa_final_return = equity_curve[-1].get('ipsa_return', 0.0) or 0.0

        report = {
            "status": "ok",
            "period": f"{equity_curve[0]['date']} to {equity_curve[-1]['date']}",
            "initial_capital": initial,
            "final_equity": final,
            "total_return": total_return,
            "total_trades": n_trades,
            "win_rate": win_rate,
            "wins": n_wins,
            "losses": n_losses,
            "avg_pnl_pct": avg_pnl_pct,
            "avg_holding_days": avg_hold_days,
            "max_drawdown": max_drawdown,
            "ipsa_return": ipsa_final_return,
            "alpha_vs_ipsa": total_return - ipsa_final_return,
            "model_version": "V2" if HAS_V2 else "V1",
        }

        # Save results
        tl_df.to_csv(BACKTEST_RESULTS_FILE, index=False)
        logger.info(f"📄 Trade log saved to {BACKTEST_RESULTS_FILE}")

        # Print report
        print(f"\n{'='*60}")
        print(f"📊 AUREUS BACKTEST REPORT ({'V2 Multi-Horizon' if HAS_V2 else 'V1 Legacy'})")
        print(f"{'='*60}")
        print(f"Period:           {report['period']}")
        print(f"Capital Inicial:  ${initial:,.0f} CLP")
        print(f"Equity Final:     ${final:,.0f} CLP")
        print(f"Retorno Total:    {total_return:.2%}")
        print(f"IPSA Benchmark:   {ipsa_final_return:.2%}")
        print(f"Alpha vs IPSA:    {report['alpha_vs_ipsa']:.2%}")
        print(f"{'─'*40}")
        print(f"Trades Totales:   {n_trades}")
        print(f"Win Rate:         {win_rate:.1%}")
        print(f"P&L Promedio:     {avg_pnl_pct:.2%}")
        print(f"Hold Promedio:    {avg_hold_days:.1f} days")
        print(f"Max Drawdown:     {max_drawdown:.2%}")
        print(f"{'='*60}\n")

        return report


if __name__ == "__main__":
    engine = BacktestEngine()
    result = engine.run()
    if result and result.get("status") == "ok":
        print("✅ Backtest completed successfully.")
    else:
        print("❌ Backtest failed.")
