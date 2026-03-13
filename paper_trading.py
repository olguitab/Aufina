"""
Paper Trading Engine for Aureus Demo Mode.
Simulates trades with real market prices but virtual capital of 10,000,000 CLP.
This is completely isolated from the live trading database.
"""

import sqlite3
import json
import os
from typing import Dict, List, Any
from datetime import datetime
import pytz
from paths import PAPER_TRADING_DB_FILE, ensure_project_dirs

PAPER_DB_PATH = PAPER_TRADING_DB_FILE
INITIAL_BALANCE_CLP = 10_000_000.0  # 10 million CLP
CHILE_TZ = pytz.timezone("America/Santiago")

class PaperTradingDB:
    @staticmethod
    def init_db():
        """Initializes a separate, clean DB for paper trading."""
        ensure_project_dirs()
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_positions (
                ticker TEXT PRIMARY KEY,
                quantity REAL,
                avg_cost REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                signal TEXT,
                price REAL,
                quantity REAL,
                reasoning TEXT,
                confidence REAL,
                value_clp REAL
            )
        ''')

        # Set initial balance if it doesn't exist
        cursor.execute("SELECT value FROM paper_state WHERE key = 'balance'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO paper_state (key, value) VALUES ('balance', ?)", (str(INITIAL_BALANCE_CLP),))

        conn.commit()
        conn.close()

    @staticmethod
    def save_state(balance: float):
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('balance', ?)", (str(balance),))
        conn.commit()
        conn.close()

    @staticmethod
    def load_state() -> float:
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM paper_state WHERE key = 'balance'")
        row = cursor.fetchone()
        conn.close()
        return float(row[0]) if row else INITIAL_BALANCE_CLP

    @staticmethod
    def save_position(ticker: str, quantity: float, avg_cost: float):
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        if quantity <= 0:
            cursor.execute("DELETE FROM paper_positions WHERE ticker = ?", (ticker,))
        else:
            cursor.execute("INSERT OR REPLACE INTO paper_positions (ticker, quantity, avg_cost) VALUES (?, ?, ?)",
                           (ticker, quantity, avg_cost))
        conn.commit()
        conn.close()

    @staticmethod
    def load_positions() -> Dict[str, float]:
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, quantity FROM paper_positions")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def load_position_costs() -> Dict[str, float]:
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, avg_cost FROM paper_positions")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def log_trade(trade_data: Dict[str, Any]):
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO paper_trades (timestamp, ticker, signal, price, quantity, reasoning, confidence, value_clp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get("timestamp"),
            trade_data.get("ticker"),
            trade_data.get("signal"),
            trade_data.get("price"),
            trade_data.get("quantity"),
            trade_data.get("reasoning"),
            trade_data.get("confidence"),
            trade_data.get("value_clp", 0)
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def load_trade_log() -> List[Dict[str, Any]]:
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, ticker, signal, price, quantity, reasoning, confidence, value_clp FROM paper_trades ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "Hora": r[0], "Ticker": r[1], "Signal": r[2],
                "Precio": r[3], "Cantidad": r[4], "Razón": r[5],
                "Confianza": f"{r[6]:.1%}" if r[6] else "N/A",
                "Valor CLP": f"${r[7]:,.0f}" if r[7] else "N/A"
            } for r in rows
        ]

    @staticmethod
    def reset(new_balance: float = None):
        """Reset paper trading to a fresh state with the given balance (defaults to INITIAL_BALANCE_CLP)."""
        balance = new_balance if new_balance is not None else INITIAL_BALANCE_CLP
        ensure_project_dirs()
        conn = sqlite3.connect(PAPER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM paper_trades")
        cursor.execute("DELETE FROM paper_positions")
        cursor.execute("INSERT OR REPLACE INTO paper_state (key, value) VALUES ('balance', ?)", (str(balance),))
        conn.commit()
        conn.close()


class PaperPortfolio:
    """A paper trading portfolio that mirrors PortfolioManager but uses PaperTradingDB."""

    def __init__(self):
        PaperTradingDB.init_db()
        self.balance = PaperTradingDB.load_state()
        self.positions = PaperTradingDB.load_positions()
        self.position_costs = PaperTradingDB.load_position_costs()
        self.trade_log = PaperTradingDB.load_trade_log()
        self.initial_balance = INITIAL_BALANCE_CLP

    def sync_from_db(self):
        """Reload all state from DB. Call at the start of each bot cycle to pick up UI resets."""
        self.balance = PaperTradingDB.load_state()
        self.positions = PaperTradingDB.load_positions()
        self.position_costs = PaperTradingDB.load_position_costs()

    def calculate_position_size(self, price: float, confidence: float = 0.5, aggressive: bool = False) -> int:
        """
        Sizing for demo paper portfolio.
        Aggressive mode: double the allocation for faster portfolio growth.
        """
        if aggressive:
            # Aggressive: double the allocation
            if confidence > 0.60:
                risk_pct = 0.80  # 80% for high conviction aggressive
            elif confidence > 0.45:
                risk_pct = 0.50  # 50%
            else:
                risk_pct = 0.30  # 30%
        else:
            # Normal mode
            if confidence > 0.60:
                risk_pct = 0.40
            elif confidence > 0.45:
                risk_pct = 0.25
            else:
                risk_pct = 0.15

        risk_cap_from_env = float(os.environ.get("PAPER_MAX_POSITION_PCT", 0.0) or 0.0)
        if risk_cap_from_env <= 0:
            # Keep a small buffer below portfolio hard limit (default risk max position = 20%)
            configured_risk_limit = float(os.environ.get("RISK_MAX_POSITION_PCT", 0.24) or 0.24)
            risk_cap_from_env = max(0.05, configured_risk_limit - 0.02)

        # For tiny demo accounts, concentrate capital to maximize chance of meaningful gains.
        small_portfolio_clp = float(os.environ.get("PAPER_SMALL_PORTFOLIO_CLP", 100000) or 100000)
        if self.balance <= small_portfolio_clp:
            risk_cap_from_env = max(risk_cap_from_env, 1.0)
            risk_pct = max(risk_pct, 0.95)

        risk_pct = min(risk_pct, risk_cap_from_env)

        amount = self.balance * risk_pct
        size = int(amount / price)
        if size == 0 and self.balance >= price:
            size = 1
        return size

    def execute_order(
        self,
        ticker: str,
        signal: str,
        price: float,
        reasoning: str,
        confidence: float = 0.5,
        amount_to_invest: float = None,
        adv_20d: float = 0,
        aggressive: bool = False,
    ):
        """Executes a paper trade using real market prices but virtual money."""
        timestamp = datetime.now(CHILE_TZ).strftime("%Y-%m-%d %H:%M:%S")

        if signal == "HOLD":
            return

        if signal == "BUY":
            if amount_to_invest is not None:
                size = int(float(amount_to_invest) / price)
                if size == 0 and self.balance >= price and float(amount_to_invest) >= price:
                    size = 1
            else:
                size = self.calculate_position_size(price, confidence, aggressive=aggressive)
            cost = size * price

            if size > 0 and self.balance >= cost:
                self.balance -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + size
                self.position_costs[ticker] = price
                PaperTradingDB.save_state(self.balance)
                PaperTradingDB.save_position(ticker, self.positions[ticker], price)
                PaperTradingDB.log_trade({
                    "timestamp": timestamp, "ticker": ticker, "signal": "BUY",
                    "price": price, "quantity": size, "reasoning": reasoning,
                    "confidence": confidence, "value_clp": cost
                })
                print(f"[PAPER] ✅ BUY {size} x {ticker} @ ${price:.2f} = ${cost:,.0f} CLP")

        elif signal == "SELL":
            size = self.positions.get(ticker, 0)
            if size > 0:
                revenue = size * price
                buy_cost = self.position_costs.get(ticker, price)
                pnl = (price - buy_cost) * size
                self.balance += revenue
                self.positions[ticker] = 0
                PaperTradingDB.save_state(self.balance)
                PaperTradingDB.save_position(ticker, 0, price)
                PaperTradingDB.log_trade({
                    "timestamp": timestamp, "ticker": ticker, "signal": "SELL",
                    "price": price, "quantity": size, "reasoning": reasoning,
                    "confidence": confidence, "value_clp": revenue
                })
                print(f"[PAPER] 💸 SELL {size} x {ticker} @ ${price:.2f} = ${revenue:,.0f} CLP | P&L: ${pnl:,.0f}")

    def get_total_value(self, market_data_engine) -> float:
        """Calculates total portfolio value (cash + positions at current prices)."""
        total = self.balance
        for ticker, qty in self.positions.items():
            if qty > 0:
                try:
                    data = market_data_engine.get_comprehensive_data(ticker)
                    price = data.get("current_price", 0.0) or data.get("close_price", 0.0)
                    if price and price > 0:
                        total += qty * price
                    else:
                        total += qty * float(self.position_costs.get(ticker, 0.0) or 0.0)
                except Exception:
                    total += qty * float(self.position_costs.get(ticker, 0.0) or 0.0)
        return total

    def get_roi(self) -> float:
        return ((self.balance - self.initial_balance) / self.initial_balance) * 100

