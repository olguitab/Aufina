from datetime import datetime
import sqlite3
from database import DB_PATH
from risk_engine import RiskEngine

class PortfolioManager:
    def __init__(self, initial_balance: float = 100000.0):
        from database import TradingDB
        TradingDB.init_db()
        self.balance = TradingDB.load_state()
        self.positions = TradingDB.load_positions()
        self.trade_log = TradingDB.load_trade_log()
        self.risk_engine = RiskEngine()

    def _get_position_reference_prices(self) -> dict:
        if not self.positions:
            return {}
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT ticker, avg_cost FROM positions").fetchall()
        conn.close()
        prices = {ticker: float(avg_cost) for ticker, avg_cost in rows if avg_cost and avg_cost > 0}
        return prices
        
    def calculate_position_size(self, price: float, confidence: float = 0.5) -> int:
        """Calculate position size based on ML conviction (Ultra-Aggressive).
        - High (>60%): 40% of balance (Maximize Gains)
        - Med (45-60%): 25% of balance
        - Low (<45%): 15% of balance
        """
        if confidence > 0.60:
            risk_pct = 0.40
        elif confidence > 0.45:
            risk_pct = 0.25
        else:
            risk_pct = 0.15
            
        amount_to_risk = self.balance * risk_pct
        size = int(amount_to_risk / price)
        if size == 0 and self.balance >= price:
            size = 1
        return size

    def execute_order(self, ticker: str, signal: str, price: float, reasoning: str, confidence: float = 0.5, amount_to_invest: float = None, adv_20d: float = 0):
        from database import TradingDB
        print(f"\n--- EXECUTING {signal} ORDER FOR {ticker} (Confidence: {confidence:.2%}) ---")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if signal == "HOLD":
            # ... (unchanged)
            pass

        if signal == "BUY":
            # 1. Base Sizing from Confidence or explicit amount
            if amount_to_invest is not None:
                cost_limit = amount_to_invest
                size = int(cost_limit / price)
            else:
                size = self.calculate_position_size(price, confidence=confidence)
                cost_limit = size * price
            
            # 2. Sentinel AI Liquidity Rule: Max 5% of ADV
            if adv_20d > 0:
                max_size = int(adv_20d * 0.05)
                if size > max_size:
                    print(f"⚠️ LIQUIDITY ALERT: Size {size} exceeds 5% of ADV ({max_size}). Capping order.")
                    size = max_size
                    if size == 0:
                        print(f"❌ SKIPPED {ticker}: Liquidity is too low for 1 share.")
                        return

            cost = size * price

            if size > 0:
                market_prices = self._get_position_reference_prices()
                market_prices[ticker] = price
                approved, reason = self.risk_engine.validate_buy(
                    ticker=ticker,
                    proposed_qty=size,
                    proposed_price=price,
                    cash_balance=self.balance,
                    positions=self.positions,
                    ticker_prices=market_prices,
                )
                if not approved:
                    log = f"[{timestamp}] Action: BLOCKED BUY {ticker}. Risk rule: {reason}"
                    print(log)
                    return
            
            if size > 0 and self.balance >= cost:
                self.balance -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + size
                TradingDB.save_state(self.balance)
                TradingDB.save_position(ticker, self.positions[ticker], price)
                log = f"[{timestamp}] Action: BUY {size} shares of {ticker} at ${price:.2f}. Total Cost: ${cost:.2f}"
            elif size == 0:
                log = f"[{timestamp}] Action: SKIPPED BUY {ticker}. Monto disponible insuficiente para 1 unidad."
                print(log)
                return
            else:
                log = f"[{timestamp}] Action: FAILED BUY {ticker}. Insufficient balance (${self.balance:.2f}) for cost ${cost:.2f}"
                print(log)
                return
                
        elif signal == "SELL":
            size = self.positions.get(ticker, 0)
            if size > 0:
                revenue = size * price
                self.balance += revenue
                self.positions[ticker] = 0
                TradingDB.save_state(self.balance)
                TradingDB.save_position(ticker, 0, price)
                log = f"[{timestamp}] Action: SELL {size} shares of {ticker} at ${price:.2f}. Revenue: ${revenue:.2f}"
            else:
                log = f"[{timestamp}] Action: FAILED SELL {ticker}. No existing position."
                return
                
        trade_entry = {
            "Hora": timestamp,
            "Activo": ticker, "Acción": signal, "Precio": price, "Razón": reasoning[:60]
        }
        self.trade_log.insert(0, trade_entry)
        TradingDB.log_trade({
            "timestamp": timestamp, "ticker": ticker, "signal": signal,
            "price": price, "quantity": size if signal != "SELL" else size,
            "reasoning": reasoning, "confidence": confidence
        })
        print(log)
        print(f"Current Balance: ${self.balance:.2f} | Positions: {self.positions}\n")

    def get_portfolio_distribution(self, market_data_engine) -> dict:
        """Returns a dictionary of Asset: Value in dollars for the portfolio pie chart.
        Uses parallelism to ensure the dashboard doesn't hang.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        distribution = {"Cash": self.balance}
        active_tickers = [t for t, s in self.positions.items() if s > 0]
        
        if not active_tickers:
            return distribution

        def fetch_price(ticker):
            # Use the new comprehensive logic which is faster and safer
            data = market_data_engine.get_comprehensive_data(ticker)
            return ticker, data.get("current_price", 0)

        with ThreadPoolExecutor(max_workers=len(active_tickers)) as executor:
            future_to_ticker = {executor.submit(fetch_price, t): t for t in active_tickers}
            for future in as_completed(future_to_ticker):
                ticker, price = future.result()
                if price > 0:
                    distribution[ticker] = self.positions[ticker] * price
                else:
                    # Fail-safe: if price can't be found, keep ticker at 0 to avoid crash
                    distribution[ticker] = 0
                    
        return distribution

    def manual_entry(self, ticker: str, quantity: float, price: float):
        from database import TradingDB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if quantity <= 0 or price <= 0:
            msg = "Cantidad y precio deben ser mayores a 0."
            print(f"[{timestamp}] Action: FAILED MANUAL ENTRY {ticker}. {msg}")
            return False, msg

        cost = quantity * price
        if self.balance < cost:
            msg = f"Capital insuficiente: disponible ${self.balance:.2f}, requerido ${cost:.2f}."
            print(f"[{timestamp}] Action: FAILED MANUAL ENTRY {ticker}. {msg}")
            return False, msg

        prev_qty = self.positions.get(ticker, 0)
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT avg_cost FROM positions WHERE ticker = ?", (ticker,)).fetchone()
        conn.close()
        prev_avg_cost = row[0] if row else price

        new_qty = prev_qty + quantity
        new_avg_cost = ((prev_qty * prev_avg_cost) + (quantity * price)) / new_qty

        self.balance -= cost
        self.positions[ticker] = new_qty
        TradingDB.save_state(self.balance)
        TradingDB.save_position(ticker, new_qty, new_avg_cost)
        log = f"[{timestamp}] Action: MANUAL ENTRY {quantity} shares of {ticker} at ${price:.2f}."
        trade_entry = {
            "Hora": timestamp,
            "Activo": ticker, "Acción": "MANUAL BUY", "Precio": price, "Razón": "Entrada manual por Telegram"
        }
        self.trade_log.insert(0, trade_entry)
        TradingDB.log_trade({
            "timestamp": timestamp, "ticker": ticker, "signal": "BUY",
            "price": price, "quantity": quantity,
            "reasoning": "Entrada manual por Telegram", "confidence": 1.0
        })
        print(log)
        return True, f"Compra registrada: {quantity}x {ticker} a ${price:,.2f}."

    def manual_exit(self, ticker: str, quantity: float = None, price: float = None):
        from database import TradingDB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_qty = self.positions.get(ticker, 0)

        if current_qty <= 0:
            msg = f"{ticker} no existe en portafolio."
            print(f"[{timestamp}] Action: FAILED MANUAL EXIT. {msg}")
            return False, msg

        qty_to_sell = current_qty if quantity is None else quantity
        if qty_to_sell <= 0:
            msg = "La cantidad a vender debe ser mayor a 0."
            print(f"[{timestamp}] Action: FAILED MANUAL EXIT {ticker}. {msg}")
            return False, msg

        if qty_to_sell > current_qty:
            msg = f"Cantidad inválida: tienes {current_qty} y quieres vender {qty_to_sell}."
            print(f"[{timestamp}] Action: FAILED MANUAL EXIT {ticker}. {msg}")
            return False, msg

        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT avg_cost FROM positions WHERE ticker = ?", (ticker,)).fetchone()
        conn.close()
        avg_cost = row[0] if row else 0

        if price is not None and price > 0:
            revenue = qty_to_sell * price
            self.balance += revenue
            TradingDB.save_state(self.balance)
        else:
            revenue = None

        remaining_qty = current_qty - qty_to_sell
        self.positions[ticker] = remaining_qty
        TradingDB.save_position(ticker, remaining_qty, avg_cost)

        if revenue is not None:
            log = f"[{timestamp}] Action: MANUAL EXIT {qty_to_sell} shares of {ticker} at ${price:.2f}. Revenue: ${revenue:.2f}"
        else:
            log = f"[{timestamp}] Action: MANUAL EXIT {qty_to_sell} shares of {ticker} (sin precio informado)."

            trade_entry = {
                "Hora": timestamp,
                "Activo": ticker, "Acción": "MANUAL SELL", "Precio": price if price is not None else 0,
                "Razón": "Salida manual por Telegram"
            }
            self.trade_log.insert(0, trade_entry)
            TradingDB.log_trade({
                "timestamp": timestamp, "ticker": ticker, "signal": "SELL",
                "price": price if price is not None else 0, "quantity": qty_to_sell,
                "reasoning": "Salida manual por Telegram", "confidence": 1.0
            })
            print(log)
            if price is not None and price > 0:
                return True, f"Venta registrada: {qty_to_sell}x {ticker} a ${price:,.2f}."
            return True, f"Venta registrada: {qty_to_sell}x {ticker}."
