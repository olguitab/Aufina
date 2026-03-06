import os
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import MVP components
from market_data import MarketData
from execution import PortfolioManager
from intelligence import IntelligenceLayer
from models import Predictor
from paper_trading import PaperPortfolio
import requests as http_requests
from paths import BOT_LOG_FILE, TRADING_DB_FILE, ensure_project_dirs

# --- Setup Logging ---
ensure_project_dirs()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(BOT_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
# --- Sentinel AI Segmented Watchlists ---
BLUE_CHIPS = ['SQM-B.SN', 'CHILE.SN', 'BSANTANDER.SN', 'LTM.SN', 'ENELCHILE.SN']
COMMODITIES = ['CAP.SN', 'VAPORES.SN', 'CMPC.SN', 'COPEC.SN']
SMALL_CAPS = ['SALFACORP.SN', 'HITES.SN', 'SONDA.SN', 'BESALCO.SN']

# Complete Universe (IPSA/IGPA + Coverage)
WATCHLIST = list(set(BLUE_CHIPS + COMMODITIES + SMALL_CAPS + [
    'CENCOSUD.SN', 'ENELAM.SN', 'FALABELLA.SN', 'BCI.SN', 'AGUAS-A.SN', 'PARAUCO.SN',
    'ANDINA-B.SN', 'CCU.SN', 'IAM.SN', 'MALLPLAZA.SN', 'ENTEL.SN', 'SMU.SN', 'RIPLEY.SN',
    'ILC.SN', 'CONCHATORO.SN', 'COLBUN.SN', 'VSPT.SN'
]))

INTERVAL_SECONDS = int(os.environ.get("TRADING_INTERVAL_SECONDS", 60)) # Default 60 seconds for stable continuous loop
LLM_BATCH_SIZE = max(1, int(os.environ.get("LLM_BATCH_SIZE", 10)))

def send_telegram(message: str, chat_id_override: str = None):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id_override or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        http_requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")

class AutonomousBot:
    def __init__(self):
        self.portfolio = PortfolioManager()
        self.paper_portfolio = PaperPortfolio()  # 10M CLP demo mode
        self.intelligence = IntelligenceLayer()
        self.market_data = MarketData()
        # Trailing Stop: tracks the highest price seen since entry for each position
        self._price_peaks: dict = {}          # {ticker: highest_price_seen}
        self.TRAILING_STOP_PCT = 0.05         # Sell if price drops 5% from peak
        self.TAKE_PROFIT_PCT = 0.15            # Hard take-profit at +15% (massive move)
        self.MIN_HOLD_CYCLES = 3               # Hold at least 3 cycles before considering sell
        self._last_buy_alerts: dict = {}
        self.telegram_alerts_enabled = True
        self.last_update_id = 0
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if token:
            try:
                resp = http_requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=5).json()
                if resp.get("ok") and resp.get("result"):
                    self.last_update_id = resp["result"][-1]["update_id"]
            except Exception:
                pass

    def _send_automatic_telegram(self, message: str, chat_id_override: str = None):
        """Sends Telegram alerts only when automatic notifications are enabled."""
        if not self.telegram_alerts_enabled:
            return
        send_telegram(message, chat_id_override)

    def _execute_paper_signal(self, ticker: str, signal: str, price: float, reasoning: str, confidence: float = 0.5):
        """Executes autonomous paper-trading orders for the 10M demo portfolio."""
        if price <= 0:
            return

        if signal == "BUY":
            if self.paper_portfolio.positions.get(ticker, 0) > 0:
                return
        elif signal == "SELL":
            if self.paper_portfolio.positions.get(ticker, 0) <= 0:
                return
        else:
            return

        self.paper_portfolio.execute_order(
            ticker=ticker,
            signal=signal,
            price=price,
            reasoning=reasoning,
            confidence=confidence,
        )

    def _get_buy_recommendation_message(self, capital: float) -> str:
        """Builds a BUY allocation plan based on latest market scan and AI signals."""
        try:
            context_snapshot = self.intelligence.context_service.analyze_context()
            ticker_data = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                fut_to_ticker = {executor.submit(self.market_data.get_comprehensive_data, t): t for t in WATCHLIST}
                for fut in as_completed(fut_to_ticker):
                    ticker = fut_to_ticker[fut]
                    ticker_data[ticker] = fut.result()

            promising_batch = {}
            for ticker, data in ticker_data.items():
                tech = data.get("technical_data", {})
                ml_prob = tech.get("ml_confidence", 0.3)
                if ml_prob > 0.30:
                    promising_batch[ticker] = data

            if not promising_batch:
                return "ℹ️ No encontré acciones candidatas en este momento."

            final_analyses = {}
            batch_list = list(promising_batch.keys())
            for i in range(0, len(batch_list), LLM_BATCH_SIZE):
                chunk = {k: promising_batch[k] for k in batch_list[i : i + LLM_BATCH_SIZE]}
                final_analyses.update(self.intelligence.bulk_analyze(chunk, context_data=context_snapshot))

            buy_candidates = []
            for ticker, analysis in final_analyses.items():
                if analysis.signal != "BUY":
                    continue
                if self.portfolio.positions.get(ticker, 0) > 0:
                    continue
                price = ticker_data.get(ticker, {}).get("current_price", 0)
                if price <= 0:
                    continue
                conf = max(float(getattr(analysis, "ml_confidence", 0.0)), 0.01)
                buy_candidates.append((ticker, analysis, price, conf))

            if not buy_candidates:
                return "ℹ️ No hay señales de compra activas para recomendar en este momento."

            buy_candidates.sort(key=lambda item: item[3], reverse=True)
            top_candidates = buy_candidates[:8]
            total_weight = sum(item[3] for item in top_candidates)

            lines = [
                "🧾 *Plan de compra sugerido*",
                f"💰 Capital informado: ${capital:,.0f} CLP",
                f"📌 Acciones sugeridas: {len(top_candidates)}",
                ""
            ]

            total_used = 0
            for ticker, _analysis, price, conf in top_candidates:
                weight = conf / total_weight if total_weight > 0 else (1 / len(top_candidates))
                alloc_clp = capital * weight
                qty = int(alloc_clp / price)
                used_clp = qty * price
                total_used += used_clp

                if qty <= 0:
                    lines.append(f"• {ticker}: sin capital suficiente para 1 acción (precio ${price:,.0f})")
                else:
                    lines.append(
                        f"• *Invierte* ${used_clp:,.0f} CLP en *{ticker}* "
                        f"(comprar {qty} acciones @ ${price:,.0f}, peso {weight:.1%})"
                    )

            lines.extend([
                "",
                f"✅ Total estimado a invertir: ${total_used:,.0f} CLP",
                f"🏦 Caja remanente estimada: ${max(capital - total_used, 0):,.0f} CLP",
                "",
                "_Confirma cada ejecución con `/comprar TICKER CANTIDAD PRECIO`_"
            ])

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error generating buy recommendation plan: {e}")
            return "❌ No pude generar la recomendación en este momento. Intenta de nuevo en unos minutos."

    def _check_telegram_commands(self):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token: return
        
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            params = {"offset": self.last_update_id + 1, "timeout": 5}
            resp = http_requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not data.get("ok"): return
            
            for item in data.get("result", []):
                self.last_update_id = item["update_id"]
                msg = item.get("message", {})
                text = msg.get("text", "").strip()
                chat_id = msg.get("chat", {}).get("id")
                
                if not text or not chat_id: continue
                
                parts = text.split()
                cmd = parts[0].lower()
                
                if cmd == "/capital":
                    if len(parts) >= 2:
                        try:
                            new_cap = float(parts[1].replace(".", "").replace(",", ""))
                            from database import TradingDB
                            self.portfolio.balance = new_cap
                            TradingDB.save_state(new_cap)
                            send_telegram(
                                f"✅ Capital de la cuenta real actualizado a ${new_cap:,.0f} CLP.\n"
                                f"⏳ Calculando plan de compra sugerido...",
                                str(chat_id)
                            )
                            recommendation_msg = self._get_buy_recommendation_message(new_cap)
                            send_telegram(recommendation_msg, str(chat_id))
                        except ValueError:
                            send_telegram("❌ Error: Formato incorrecto. Usa: `/capital 5000000`", str(chat_id))

                elif cmd == "/stop":
                    self.telegram_alerts_enabled = False
                    send_telegram(
                        "⏸️ Notificaciones automáticas pausadas.\n"
                        "Usa `/start` para reanudarlas.",
                        str(chat_id)
                    )

                elif cmd == "/start":
                    self.telegram_alerts_enabled = True
                    send_telegram(
                        "▶️ Notificaciones automáticas reactivadas.",
                        str(chat_id)
                    )
                
                elif cmd == "/comprar":
                    if len(parts) >= 4:
                        ticker = parts[1].upper()
                        try:
                            qty = float(parts[2])
                            price = float(parts[3])
                            ok, detail = self.portfolio.manual_entry(ticker, qty, price)
                            if ok:
                                send_telegram(
                                    f"✅ Compra confirmada: {qty}x {ticker} a ${price:,.0f}.\n"
                                    f"💼 Capital libre actualizado: ${self.portfolio.balance:,.0f} CLP.",
                                    str(chat_id)
                                )
                            else:
                                send_telegram(f"❌ No se pudo registrar la compra: {detail}", str(chat_id))
                        except ValueError:
                            send_telegram("❌ Error: Cantidad o precio inválido.", str(chat_id))
                    else:
                        send_telegram("❌ Formato: `/comprar [TICKER] [CANTIDAD] [PRECIO]`", str(chat_id))
                        
                elif cmd == "/vender":
                    if len(parts) >= 4:
                        ticker = parts[1].upper()
                        try:
                            qty = float(parts[2])
                            price = float(parts[3])
                            ok, detail = self.portfolio.manual_exit(ticker, qty, price)
                            if ok:
                                if self.portfolio.positions.get(ticker, 0) <= 0 and f"REAL_{ticker}" in self._price_peaks:
                                    del self._price_peaks[f"REAL_{ticker}"]
                                send_telegram(
                                    f"✅ Venta confirmada: {qty}x {ticker} a ${price:,.0f}.\n"
                                    f"💼 Capital libre actualizado: ${self.portfolio.balance:,.0f} CLP.",
                                    str(chat_id)
                                )
                            else:
                                send_telegram(f"❌ No se pudo registrar la venta: {detail}", str(chat_id))
                        except ValueError:
                            send_telegram("❌ Error: Cantidad o precio inválido.", str(chat_id))
                    else:
                        send_telegram("❌ Formato: `/vender [TICKER] [CANTIDAD] [PRECIO]`", str(chat_id))
                        
                elif cmd == "/estado":
                    msg_txt = f"📊 *Estado Portafolio Real*\nCapital Libre: ${self.portfolio.balance:,.0f} CLP\n"
                    estado_notif = "🟢 Activas" if self.telegram_alerts_enabled else "⏸️ Pausadas"
                    msg_txt += f"Notificaciones automáticas: {estado_notif}\n"
                    active = {k: v for k, v in self.portfolio.positions.items() if v > 0}
                    if active:
                        msg_txt += "\n*Posiciones Activas:*\n"
                        for tk, q in active.items():
                            msg_txt += f"• {tk}: {q} acciones\n"
                    else:
                        msg_txt += "\n_Sin posiciones activas._"
                    send_telegram(msg_txt, str(chat_id))
        
        except Exception as e:
            logger.error(f"Error checking Telegram commands: {e}")

    def _get_active_tickers(self) -> set:
        """Returns set of tickers with active positions in either portfolio."""
        return set(list(self.portfolio.positions.keys()) + list(self.paper_portfolio.positions.keys()))

    def _update_trailing_stops(self, ticker_data: dict):
        """
        Dual-Confirmation Sell System (Maximiza Ganancias):
        
        REGLA 1 — Take Profit Duro (+15%): Vende SIEMPRE de inmediato. Ganancia excepcional asegurada.
        
        REGLA 2 — Trailing Stop Inteligente (-5% desde pico):
          - Detecta la posible reversión por precio.
          - PERO consulta al modelo ML institucional (entrenado 17h).
          - Si ML dice probabilidad > 45%: Espera. Puede ser ruido. No vende.
          - Si ML dice probabilidad < 45%: Confirma debilidad. VENDE.
          - Así evitamos salir de una acción que aún tiene momentum real.
        """
        import sqlite3

        def _process_position(ticker, qty, avg_cost, current_price, portfolio_label, ticker_data_entry):
            if qty <= 0 or current_price <= 0:
                return None

            pnl_pct = (current_price - avg_cost) / avg_cost

            # Track price peak since entry
            peak_key = f"{portfolio_label}_{ticker}"
            prev_peak = self._price_peaks.get(peak_key, avg_cost)
            new_peak = max(prev_peak, current_price)
            self._price_peaks[peak_key] = new_peak
            drop_from_peak = (current_price - new_peak) / new_peak

            # --- REGLA 1: Hard Take-Profit +15% (sin condiciones, asegura ganancia excepcional) ---
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                return ("SELL", current_price, f"🎯 Take Profit: +{pnl_pct:.1%} — Ganancia excepcional asegurada")

            # --- REGLA 2: Trailing Stop + Confirmación ML ---
            if new_peak > avg_cost and drop_from_peak <= -self.TRAILING_STOP_PCT:
                # Price fell from peak. Now ask the ML model: is the trend still bullish?
                tech = ticker_data_entry.get("technical_data", {})
                ml_prob = Predictor.predict_probability(tech)

                if ml_prob >= 0.45:
                    # ML says: still bullish probability. Likely noise/temporary dip. HOLD.
                    logger.info(
                        f"[{portfolio_label}] {ticker}: Trailing activado pero ML={ml_prob:.1%} (≥45%) → MANTENIENDO. "
                        f"Puede ser ruido temporal. P&L: {pnl_pct:.1%}"
                    )
                    return None
                else:
                    # ML confirms weakness. Real reversal detected. SELL.
                    reason = (
                        f"📉 Trailing Stop CONFIRMADO por ML: Cayó {drop_from_peak:.1%} desde pico "
                        f"+ ML prob={ml_prob:.1%} (debilidad confirmada). "
                        f"Ganancia neta: {pnl_pct:.1%}"
                    )
                    return ("SELL", current_price, reason)

            logger.info(
                f"[{portfolio_label}] {ticker}: P&L {pnl_pct:.1%} | "
                f"Peak: ${new_peak:.2f} | Drop: {drop_from_peak:.1%} → HOLD (en tendencia)"
            )
            return None

        # --- Real Portfolio ---
        for ticker, qty in list(self.portfolio.positions.items()):
            current_price = ticker_data.get(ticker, {}).get("current_price", 0)
            conn = sqlite3.connect(TRADING_DB_FILE)
            row = conn.execute("SELECT avg_cost FROM positions WHERE ticker = ?", (ticker,)).fetchone()
            conn.close()
            avg_cost = row[0] if row else current_price

            result = _process_position(ticker, qty, avg_cost, current_price, "REAL", ticker_data.get(ticker, {}))
            if result:
                signal, price, reason = result
                logger.info(f"[REAL] {signal} {ticker}: {reason}")
                self._send_automatic_telegram(
                    f"📉 *ALERTA DE VENTA (REAL)* 📉\n"
                    f"Acción: {ticker}\n"
                    f"Cantidad estimada en cartera: {qty} acciones\n"
                    f"Precio Actual: ${price:,.0f}\n"
                    f"Motivo: {reason}\n\n"
                    f"_Si ejecutas la venta, confirma con_ `/vender {ticker} {qty} {price:,.0f}`"
                )

        # --- Paper Portfolio (Demo 10M): ejecución automática de ventas ---
        for ticker, qty in list(self.paper_portfolio.positions.items()):
            current_price = ticker_data.get(ticker, {}).get("current_price", 0)
            avg_cost = self.paper_portfolio.position_costs.get(ticker, current_price)

            result = _process_position(ticker, qty, avg_cost, current_price, "PAPER", ticker_data.get(ticker, {}))
            if result:
                signal, price, reason = result
                logger.info(f"[PAPER] {signal} {ticker}: {reason}")
                self._execute_paper_signal(
                    ticker=ticker,
                    signal=signal,
                    price=price,
                    reasoning=reason,
                    confidence=0.9,
                )
                if signal == "SELL":
                    peak_key = f"PAPER_{ticker}"
                    if peak_key in self._price_peaks:
                        del self._price_peaks[peak_key]

    def _check_circuit_breakers(self, ticker_data: dict):
        """
        SENTINEL AI CIRCUIT BREAKER:
        If a position drops > 2% AND there is relevant news detected, execute IMMEDIATE stop-loss.
        """
        import sqlite3

        # Real portfolio: alerta manual
        for ticker, qty in self.portfolio.positions.items():
            if qty <= 0:
                continue

            data = ticker_data.get(ticker, {})
            current_price = data.get("current_price", 0)
            if current_price <= 0:
                continue

            conn = sqlite3.connect(TRADING_DB_FILE)
            row = conn.execute("SELECT avg_cost FROM positions WHERE ticker = ?", (ticker,)).fetchone()
            conn.close()
            avg_cost = row[0] if row else current_price

            pnl_pct = (current_price - avg_cost) / avg_cost
            news = data.get("news_text", "")

            if pnl_pct <= -0.02 and "No news found" not in news:
                reason = f"🚨 CIRCUIT BREAKER: Drop of {pnl_pct:.1%} with news context. Emergency Liquidate."
                logger.warning(f"[REAL] {ticker}: {reason}")
                self._send_automatic_telegram(
                    f"🚨 *ALERTA DE VENTA URGENTE (CIRCUIT BREAKER)* 🚨\n"
                    f"Acción: {ticker}\n"
                    f"Cantidad estimada en cartera: {qty} acciones\n"
                    f"Precio Actual: ${current_price:,.0f}\n"
                    f"Motivo: {reason}\n\n"
                    f"_Si ejecutas la venta, confirma con_ `/vender {ticker} {qty} {current_price:,.0f}`"
                )

        # Paper portfolio: ejecución automática para rotación continua demo
        for ticker, qty in self.paper_portfolio.positions.items():
            if qty <= 0:
                continue

            data = ticker_data.get(ticker, {})
            current_price = data.get("current_price", 0)
            if current_price <= 0:
                continue

            avg_cost = self.paper_portfolio.position_costs.get(ticker, current_price)
            pnl_pct = (current_price - avg_cost) / avg_cost
            news = data.get("news_text", "")

            if pnl_pct <= -0.02 and "No news found" not in news:
                reason = f"🚨 CIRCUIT BREAKER (PAPER): Drop of {pnl_pct:.1%} with news context. Emergency Liquidate."
                logger.warning(f"[PAPER] {ticker}: {reason}")
                self._execute_paper_signal(
                    ticker=ticker,
                    signal="SELL",
                    price=current_price,
                    reasoning=reason,
                    confidence=0.95,
                )
                peak_key = f"PAPER_{ticker}"
                if peak_key in self._price_peaks:
                    del self._price_peaks[peak_key]

    def run_cycle(self):
        logger.info("Starting market scan cycle (Sentinel AI Engine)...")
        self._check_telegram_commands()
        
        # Snapshot global context once to avoid repeated LLM calls across chunks
        context_snapshot = self.intelligence.context_service.analyze_context()
        
        # 1. Fetch Market Data ...
        ticker_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            fut_to_ticker = {executor.submit(self.market_data.get_comprehensive_data, t): t for t in WATCHLIST}
            for fut in as_completed(fut_to_ticker):
                t = fut_to_ticker[fut]
                ticker_data[t] = fut.result()

        # 1.5 Run Protections
        self._check_circuit_breakers(ticker_data)
        self._update_trailing_stops(ticker_data)

        # 2. Preparation
        promising_batch = {}
        for ticker, data in ticker_data.items():
            tech = data.get("technical_data", {})
            ml_prob = tech.get("ml_confidence", 0.3) # Fallback
            
            if ml_prob > 0.30 or ticker in self._get_active_tickers():
                promising_batch[ticker] = data

        # 3. AI Analysis
        final_analyses = {}
        batch_list = list(promising_batch.keys())
        for i in range(0, len(batch_list), LLM_BATCH_SIZE):
            chunk = {k: promising_batch[k] for k in batch_list[i : i + LLM_BATCH_SIZE]}
            final_analyses.update(self.intelligence.bulk_analyze(chunk, context_data=context_snapshot))

        # 4. Execution
        buy_candidates = []
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for ticker, analysis in final_analyses.items():
            price = ticker_data.get(ticker, {}).get("current_price", 0)
            confidence = float(getattr(analysis, "ml_confidence", getattr(analysis, "confidence", 0.5)) or 0.5)
            signal = getattr(analysis, "signal", "HOLD")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

            if analysis.signal == "BUY":
                buy_candidates.append((ticker, analysis))
                self._execute_paper_signal(
                    ticker=ticker,
                    signal="BUY",
                    price=price,
                    reasoning=analysis.reasoning,
                    confidence=confidence,
                )
            elif analysis.signal == "SELL":
                self._execute_paper_signal(
                    ticker=ticker,
                    signal="SELL",
                    price=price,
                    reasoning=analysis.reasoning,
                    confidence=confidence,
                )
                # Solo alerta para cuenta real (ejecución manual)
                if ticker in self.portfolio.positions and self.portfolio.positions[ticker] > 0:
                    price = ticker_data[ticker]["current_price"]
                    qty = self.portfolio.positions[ticker]
                    self._send_automatic_telegram(
                        f"📉 *ALERTA DE VENTA (IA)* 📉\n"
                        f"Acción: {ticker}\n"
                        f"Cantidad estimada en cartera: {qty} acciones\n"
                        f"Precio Actual: ${price:,.0f}\n"
                        f"Razón: {analysis.reasoning}\n\n"
                        f"_Si ejecutas la venta, confirma con_ `/vender {ticker} {qty} {price:,.0f}`"
                    )

        logger.info(
            "Model signal summary: "
            f"BUY={signal_counts.get('BUY', 0)} | "
            f"SELL={signal_counts.get('SELL', 0)} | "
            f"HOLD={signal_counts.get('HOLD', 0)}"
        )

        if buy_candidates:
            # Alertas para REAL portfolio (solo recomendaciones Telegram)
            current_time = time.time()
            cooldown_hours = 4
            actionable_buys = []

            for ticker, analysis in buy_candidates:
                if ticker in self.portfolio.positions and self.portfolio.positions[ticker] > 0:
                    continue

                last_alert = self._last_buy_alerts.get(ticker, 0)
                if (current_time - last_alert) <= (cooldown_hours * 3600):
                    continue

                price = ticker_data[ticker]["current_price"]
                if price <= 0:
                    continue

                actionable_buys.append((ticker, analysis, price))

            if actionable_buys:
                capital_disponible = max(self.portfolio.balance, 0)
                capital_por_accion = capital_disponible / len(actionable_buys)

                for ticker, analysis, price in actionable_buys:
                    suggested_shares = int(capital_por_accion / price)
                    suggested_size_clp = suggested_shares * price

                    if suggested_shares <= 0:
                        msg = (f"🟢 *ALERTA DE COMPRA* 🟢\n"
                               f"📌 *Acción:* {ticker}\n"
                               f"💰 *Precio Recomendado:* ${price:,.0f}\n"
                               f"🏦 *Capital Disponible:* ${capital_disponible:,.0f} CLP\n"
                               f"⚠️ *Sin capital suficiente* para comprar 1 acción al precio actual.\n"
                               f"💡 *Razón:* {analysis.reasoning}")
                    else:
                        msg = (f"🟢 *ALERTA DE COMPRA* 🟢\n"
                               f"📌 *Acción:* {ticker}\n"
                               f"💰 *Precio Recomendado:* ${price:,.0f}\n"
                               f"🏦 *Capital Disponible:* ${capital_disponible:,.0f} CLP\n"
                               f"💼 *Sugerencia:* Comprar {suggested_shares} acciones por ~${suggested_size_clp:,.0f} CLP\n"
                               f"💡 *Razón:* {analysis.reasoning}\n\n"
                               f"👉 _Si ejecutas la compra, confirma con_ `/comprar {ticker} {suggested_shares} {price:,.0f}`\n"
                               f"📝 _Si no ejecutas, no se registra ningún cambio._")

                    self._send_automatic_telegram(msg)
                    self._last_buy_alerts[ticker] = current_time
        
        logger.info("Cycle completed.")

    def start(self, once=False):
        if not os.environ.get("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY not found in environment. Exiting.")
            return

        logger.info("=== Autonomous Trading Bot Started ===")
        if once:
            self.run_cycle()
            return

        while True:
            try:
                self._check_telegram_commands()
                if self.market_data.is_santiago_market_open():
                    self.run_cycle()
                else:
                    wait_seconds = self.market_data.seconds_until_next_santiago_open()
                    logger.info(
                        "Market closed (Bolsa de Santiago). "
                        f"Skipping market queries. Next open in ~{max(wait_seconds // 60, 1)} min."
                    )
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
            
            logger.info(f"Sleeping for {INTERVAL_SECONDS} seconds...")
            time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    import sys
    bot = AutonomousBot()
    test_mode = "--test-run" in sys.argv
    bot.start(once=test_mode)
