#!/usr/bin/env python3
"""
Aureus V2 Demo — 10MM CLP Continuous Recommendation Engine
===========================================================
Runs continuously, scanning the Chilean market every cycle and printing
buy/sell/hold recommendations with horizon-aware analysis.

Usage:
    source venv/bin/activate
    python run_demo.py
"""

import os
import sys
import time
import logging
from datetime import datetime

# Ensure project dirs exist
from paths import ensure_project_dirs
ensure_project_dirs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Check V2 model
from models_v2 import MultiHorizonPredictor
if not MultiHorizonPredictor.is_available():
    print("❌ V2 model not found. Run train_sentinel.py first.")
    sys.exit(1)

from market_data import MarketData
from models import Predictor
from universe import get_trading_watchlist
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
DEMO_CAPITAL = 10_000_000  # 10MM CLP
SCAN_INTERVAL_SECONDS = int(os.environ.get("DEMO_INTERVAL", 120))
MAX_TICKERS = int(os.environ.get("DEMO_MAX_TICKERS", 24))
MAX_WORKERS = 4

def scan_market():
    """Single market scan: fetch data, predict, and print recommendations."""
    market_data = MarketData()
    watchlist = get_trading_watchlist(include_global=False)[:MAX_TICKERS]
    
    print(f"\n{'='*70}")
    print(f"🔍 AUREUS V2 — SCAN DEL MERCADO | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Capital demo: ${DEMO_CAPITAL:,.0f} CLP | Tickers: {len(watchlist)}")
    print(f"{'='*70}")
    
    # Fetch market data in parallel
    ticker_data = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fut_map = {executor.submit(market_data.get_comprehensive_data, t): t for t in watchlist}
        for fut in as_completed(fut_map):
            tk = fut_map[fut]
            try:
                ticker_data[tk] = fut.result()
            except Exception as e:
                logger.debug(f"Error fetching {tk}: {e}")
    
    recommendations = []
    
    for ticker, data in ticker_data.items():
        price = float(data.get("current_price", 0) or 0)
        if price <= 0:
            continue
        
        tech = data.get("technical_data", {}) or {}
        
        # V2 multi-horizon prediction
        try:
            pred = MultiHorizonPredictor.predict(tech, context_score=0.0)
        except Exception:
            continue
        
        signal = pred.get("composite_signal", "HOLD")
        prob = pred.get("best_probability", 0.5)
        best_h = pred.get("best_horizon", "?")
        hold_days = pred.get("suggested_hold_days", 10)
        exit_cat = pred.get("exit_category", "medium")
        
        # Per-horizon details
        horizons_detail = []
        for h_key, h_val in pred.get("horizons", {}).items():
            horizons_detail.append(f"{h_key}:{h_val['signal']}({h_val['probability']:.0%})")
        
        recommendations.append({
            "ticker": ticker,
            "price": price,
            "signal": signal,
            "prob": prob,
            "best_horizon": best_h,
            "hold_days": hold_days,
            "exit_category": exit_cat,
            "horizons": " | ".join(horizons_detail),
        })
    
    # Sort: BUY first, then by probability
    signal_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    recommendations.sort(key=lambda r: (signal_order.get(r["signal"], 1), -r["prob"]))
    
    # Print recommendations
    buy_count = sum(1 for r in recommendations if r["signal"] == "BUY")
    sell_count = sum(1 for r in recommendations if r["signal"] == "SELL")
    hold_count = sum(1 for r in recommendations if r["signal"] == "HOLD")
    
    print(f"\n📊 Señales: {buy_count} BUY | {hold_count} HOLD | {sell_count} SELL")
    print(f"{'─'*70}")
    
    for rec in recommendations:
        if rec["signal"] == "BUY":
            emoji = "🟢"
        elif rec["signal"] == "SELL":
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        # Allocation suggestion for BUYs
        alloc_msg = ""
        if rec["signal"] == "BUY" and buy_count > 0:
            alloc = DEMO_CAPITAL / buy_count
            qty = int(alloc / rec["price"])
            alloc_msg = f" → Invertir ${alloc:,.0f} ({qty} acciones)"
        
        print(
            f"{emoji} {rec['ticker']:15s} | ${rec['price']:>10,.0f} | "
            f"{rec['signal']:4s} prob={rec['prob']:.0%} | "
            f"horizonte={rec['best_horizon']} hold={rec['hold_days']}d ({rec['exit_category']})"
            f"{alloc_msg}"
        )
        print(f"   └─ {rec['horizons']}")
    
    print(f"\n{'='*70}")
    return recommendations


def run_continuous():
    """Main loop: scan every SCAN_INTERVAL_SECONDS."""
    print(f"\n🚀 AUREUS V2 DEMO — Motor de Recomendación Continuo")
    print(f"💰 Capital: ${DEMO_CAPITAL:,.0f} CLP")
    print(f"⏱️  Intervalo: {SCAN_INTERVAL_SECONDS}s")
    print(f"📊 V2 Multi-Horizon: 5d/10d/20d")
    print(f"{'='*70}\n")
    
    cycle = 0
    while True:
        cycle += 1
        try:
            logger.info(f"--- Cycle {cycle} starting ---")
            recs = scan_market()
            logger.info(f"--- Cycle {cycle} complete: {len(recs)} tickers analyzed ---")
        except KeyboardInterrupt:
            print("\n⏹️  Demo detenida por el usuario.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}")
        
        try:
            print(f"\n⏳ Próximo scan en {SCAN_INTERVAL_SECONDS}s... (Ctrl+C para detener)")
            time.sleep(SCAN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n⏹️  Demo detenida por el usuario.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        scan_market()
    else:
        run_continuous()
