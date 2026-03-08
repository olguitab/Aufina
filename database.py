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

        # 4. NAV History
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nav_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                portfolio TEXT,
                equity REAL,
                cash REAL,
                invested REAL,
                note TEXT
            )
        ''')

        # 5. Predictions Tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                prediction_horizon_days INTEGER,
                predicted_prob REAL,
                predicted_return REAL,
                realized_return REAL,
                realized_label INTEGER,
                resolved INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')

        # 6. Context Tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                global_score REAL,
                event_type TEXT,
                impact_level TEXT,
                summary TEXT,
                raw_payload TEXT
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

    @staticmethod
    def log_nav(snapshot: Dict[str, Any]):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO nav_history (timestamp, portfolio, equity, cash, invested, note)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.get("timestamp"),
            snapshot.get("portfolio", "real"),
            snapshot.get("equity", 0.0),
            snapshot.get("cash", 0.0),
            snapshot.get("invested", 0.0),
            snapshot.get("note", ""),
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def load_nav_history(portfolio: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if portfolio:
            cursor.execute(
                "SELECT timestamp, portfolio, equity, cash, invested, note FROM nav_history WHERE portfolio = ? ORDER BY id DESC LIMIT ?",
                (portfolio, int(limit)),
            )
        else:
            cursor.execute(
                "SELECT timestamp, portfolio, equity, cash, invested, note FROM nav_history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "timestamp": r[0],
                "portfolio": r[1],
                "equity": r[2],
                "cash": r[3],
                "invested": r[4],
                "note": r[5],
            }
            for r in rows
        ]

    @staticmethod
    def log_prediction(payload: Dict[str, Any]):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, ticker, prediction_horizon_days, predicted_prob,
                predicted_return, realized_return, realized_label, resolved, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            payload.get("timestamp"),
            payload.get("ticker"),
            int(payload.get("prediction_horizon_days", 3)),
            float(payload.get("predicted_prob", 0.0)),
            float(payload.get("predicted_return", 0.0)),
            payload.get("realized_return"),
            payload.get("realized_label"),
            int(payload.get("resolved", 0)),
            json.dumps(payload.get("metadata", {}), ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def load_predictions(limit: int = 2000, unresolved_only: bool = False) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if unresolved_only:
            cursor.execute(
                "SELECT id, timestamp, ticker, prediction_horizon_days, predicted_prob, predicted_return, realized_return, realized_label, resolved, metadata FROM predictions WHERE resolved = 0 ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        else:
            cursor.execute(
                "SELECT id, timestamp, ticker, prediction_horizon_days, predicted_prob, predicted_return, realized_return, realized_label, resolved, metadata FROM predictions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        rows = cursor.fetchall()
        conn.close()
        out = []
        for r in rows:
            try:
                metadata = json.loads(r[9] or "{}")
            except Exception:
                metadata = {}
            out.append(
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "ticker": r[2],
                    "prediction_horizon_days": r[3],
                    "predicted_prob": r[4],
                    "predicted_return": r[5],
                    "realized_return": r[6],
                    "realized_label": r[7],
                    "resolved": r[8],
                    "metadata": metadata,
                }
            )
        return out

    @staticmethod
    def resolve_prediction(prediction_id: int, realized_return: float, realized_label: int):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE predictions SET realized_return = ?, realized_label = ?, resolved = 1 WHERE id = ?",
            (float(realized_return), int(realized_label), int(prediction_id)),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def log_context(snapshot: Dict[str, Any]):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO context_history (timestamp, global_score, event_type, impact_level, summary, raw_payload)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.get("timestamp"),
            float(snapshot.get("global_score", 0.0)),
            snapshot.get("event_type", "unknown"),
            snapshot.get("impact_level", "unknown"),
            snapshot.get("summary", ""),
            json.dumps(snapshot.get("raw_payload", {}), ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    @staticmethod
    def load_context_history(limit: int = 1000) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, global_score, event_type, impact_level, summary, raw_payload FROM context_history ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        rows = cursor.fetchall()
        conn.close()
        out = []
        for r in rows:
            try:
                payload = json.loads(r[5] or "{}")
            except Exception:
                payload = {}
            out.append(
                {
                    "timestamp": r[0],
                    "global_score": r[1],
                    "event_type": r[2],
                    "impact_level": r[3],
                    "summary": r[4],
                    "raw_payload": payload,
                }
            )
        return out
