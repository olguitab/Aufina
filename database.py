import sqlite3
import json
import os
from typing import Dict, List, Any
from paths import TRADING_DB_FILE, ensure_project_dirs

DB_PATH = TRADING_DB_FILE

class TradingDB:
    @staticmethod
    def init_db():
        """Initializes the SQLite database with institutional tables."""
        ensure_project_dirs()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. State Table (Balance, Settings)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # 2. Portfolio Table (Ticker, Quantity, AvgCost)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY,
                quantity REAL,
                avg_cost REAL
            )
        ''')
        
        # 3. Trade Log Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                signal TEXT,
                price REAL,
                quantity REAL,
                reasoning TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    @staticmethod
    def save_state(balance: float):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO global_state (key, value) VALUES ('balance', ?)", (str(balance),))
        conn.commit()
        conn.close()

    @staticmethod
    def load_state() -> float:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM global_state WHERE key = 'balance'")
        row = cursor.fetchone()
        conn.close()
        return float(row[0]) if row else 100000.0

    @staticmethod
    def save_position(ticker: str, quantity: float, avg_cost: float):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if quantity <= 0:
            cursor.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
        else:
            cursor.execute("INSERT OR REPLACE INTO positions (ticker, quantity, avg_cost) VALUES (?, ?, ?)", 
                           (ticker, quantity, avg_cost))
        conn.commit()
        conn.close()

    @staticmethod
    def load_positions() -> Dict[str, float]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, quantity FROM positions")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def log_trade(trade_data: Dict[str, Any]):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (timestamp, ticker, signal, price, quantity, reasoning, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get("timestamp"),
            trade_data.get("ticker"),
            trade_data.get("signal"),
            trade_data.get("price"),
            trade_data.get("quantity"),
            trade_data.get("reasoning"),
            trade_data.get("confidence")
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def load_trade_log() -> List[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, ticker, signal, price, quantity, reasoning, confidence FROM trades ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "timestamp": r[0], "ticker": r[1], "signal": r[2], 
                "price": r[3], "quantity": r[4], "reasoning": r[5], "confidence": r[6]
            } for r in rows
        ]
