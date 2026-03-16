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

        # 7. Orders (Order Book trading)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                ticker TEXT,
                side TEXT,
                offer_price REAL,
                quantity REAL,
                commission_pct REAL DEFAULT 0.0014,
                commission_clp REAL,
                total_cost REAL,
                status TEXT DEFAULT 'PENDING',
                resolved_at TEXT,
                reasoning TEXT,
                confidence REAL,
                metadata TEXT
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

    # ── Order Book methods ──────────────────────────────────────────────

    @staticmethod
    def create_order(order_data: Dict[str, Any]) -> int:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO orders (
                created_at, ticker, side, offer_price, quantity,
                commission_pct, commission_clp, total_cost,
                status, reasoning, confidence, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)
        ''', (
            order_data.get("created_at"),
            order_data.get("ticker"),
            order_data.get("side"),
            float(order_data.get("offer_price", 0)),
            float(order_data.get("quantity", 0)),
            float(order_data.get("commission_pct", 0.0014)),
            float(order_data.get("commission_clp", 0)),
            float(order_data.get("total_cost", 0)),
            order_data.get("reasoning", ""),
            float(order_data.get("confidence", 0)),
            json.dumps(order_data.get("metadata", {}), ensure_ascii=False),
        ))
        order_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return order_id

    @staticmethod
    def load_pending_orders(ticker: str = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if ticker:
            cursor.execute(
                "SELECT id, created_at, ticker, side, offer_price, quantity, commission_pct, "
                "commission_clp, total_cost, status, resolved_at, reasoning, confidence, metadata "
                "FROM orders WHERE status = 'PENDING' AND ticker = ? ORDER BY id DESC",
                (ticker,),
            )
        else:
            cursor.execute(
                "SELECT id, created_at, ticker, side, offer_price, quantity, commission_pct, "
                "commission_clp, total_cost, status, resolved_at, reasoning, confidence, metadata "
                "FROM orders WHERE status = 'PENDING' ORDER BY id DESC"
            )
        rows = cursor.fetchall()
        conn.close()
        return TradingDB._rows_to_order_dicts(rows)

    @staticmethod
    def confirm_order(order_id: int):
        from datetime import datetime, timezone
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE orders SET status = 'CONFIRMED', resolved_at = ? WHERE id = ? AND status = 'PENDING'",
            (datetime.now(timezone.utc).isoformat(), int(order_id)),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def cancel_order(order_id: int):
        from datetime import datetime, timezone
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE orders SET status = 'CANCELLED', resolved_at = ? WHERE id = ? AND status = 'PENDING'",
            (datetime.now(timezone.utc).isoformat(), int(order_id)),
        )
        conn.commit()
        conn.close()

    @staticmethod
    def expire_stale_orders(timeout_seconds: int = 600) -> List[Dict[str, Any]]:
        """Marks PENDING orders older than timeout_seconds as EXPIRED. Returns the expired orders."""
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).isoformat()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, created_at, ticker, side, offer_price, quantity, commission_pct, "
            "commission_clp, total_cost, status, resolved_at, reasoning, confidence, metadata "
            "FROM orders WHERE status = 'PENDING' AND created_at < ?",
            (cutoff,),
        )
        stale_rows = cursor.fetchall()
        if stale_rows:
            stale_ids = [r[0] for r in stale_rows]
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.executemany(
                "UPDATE orders SET status = 'EXPIRED', resolved_at = ? WHERE id = ?",
                [(now_iso, oid) for oid in stale_ids],
            )
            conn.commit()
        conn.close()
        return TradingDB._rows_to_order_dicts(stale_rows)

    @staticmethod
    def load_orders(limit: int = 200, status_filter: str = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if status_filter:
            cursor.execute(
                "SELECT id, created_at, ticker, side, offer_price, quantity, commission_pct, "
                "commission_clp, total_cost, status, resolved_at, reasoning, confidence, metadata "
                "FROM orders WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status_filter, int(limit)),
            )
        else:
            cursor.execute(
                "SELECT id, created_at, ticker, side, offer_price, quantity, commission_pct, "
                "commission_clp, total_cost, status, resolved_at, reasoning, confidence, metadata "
                "FROM orders ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
        rows = cursor.fetchall()
        conn.close()
        return TradingDB._rows_to_order_dicts(rows)

    @staticmethod
    def _rows_to_order_dicts(rows) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            try:
                metadata = json.loads(r[13] or "{}")
            except Exception:
                metadata = {}
            out.append({
                "id": r[0], "created_at": r[1], "ticker": r[2], "side": r[3],
                "offer_price": r[4], "quantity": r[5], "commission_pct": r[6],
                "commission_clp": r[7], "total_cost": r[8], "status": r[9],
                "resolved_at": r[10], "reasoning": r[11], "confidence": r[12],
                "metadata": metadata,
            })
        return out
