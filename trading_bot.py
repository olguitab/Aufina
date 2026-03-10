import os
import time
import json
import logging
import threading
import fcntl
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Import MVP components
from market_data import MarketData
from execution import PortfolioManager
from intelligence import IntelligenceLayer
from models import Predictor
from paper_trading import PaperPortfolio
import requests as http_requests
from paths import BOT_LOG_FILE, TRADING_DB_FILE, ensure_project_dirs
from universe import get_trading_watchlist
from risk_engine import RiskEngine
from afp_tracker import AFPTracker
from he_analyzer import HEAnalyzer
from regime_detector import RegimeDetector
from database import TradingDB
from broker_interface import PaperBrokerAdapter, ManualRealBrokerAdapter

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


def _is_hosted_runtime() -> bool:
    return any(
        os.environ.get(flag)
        for flag in ("RENDER", "RENDER_SERVICE_ID", "RAILWAY_ENVIRONMENT", "K_SERVICE")
    )


def _env_int(name: str, local_default: int, hosted_default: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip() != "":
        return int(raw)
    if hosted_default is not None and _is_hosted_runtime():
        return int(hosted_default)
    return int(local_default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _extra_tickers_from_env() -> list[str]:
    raw = os.environ.get("TRADING_EXTRA_TICKERS", "")
    if not raw or not raw.strip():
        return []
    cleaned = []
    for part in raw.split(","):
        tk = part.strip().upper()
        if tk.startswith("$"):
            tk = tk[1:]
        if tk:
            cleaned.append(tk)
    return list(dict.fromkeys(cleaned))


# Complete Universe (IPSA + expanded Chile coverage + optional global/extra via env)
_include_global_default = True if _is_hosted_runtime() else True
_include_global = _env_flag("TRADING_INCLUDE_GLOBAL", default=_include_global_default)
_base_watchlist = get_trading_watchlist(include_global=_include_global)
_max_watchlist_size = max(1, _env_int("TRADING_MAX_TICKERS", local_default=24, hosted_default=28))
WATCHLIST = list(dict.fromkeys(_base_watchlist + _extra_tickers_from_env()))[:_max_watchlist_size]

INTERVAL_SECONDS = _env_int("TRADING_INTERVAL_SECONDS", local_default=120, hosted_default=90)
LLM_BATCH_SIZE = max(1, _env_int("LLM_BATCH_SIZE", local_default=6, hosted_default=6))
MARKET_DATA_MAX_WORKERS = max(1, _env_int("MARKET_DATA_MAX_WORKERS", local_default=4, hosted_default=3))

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
    _START_GUARD_LOCK = threading.Lock()
    _IS_RUNNING = False

    def __init__(self):
        self.portfolio = PortfolioManager()
        self.paper_portfolio = PaperPortfolio()  # 10M CLP demo mode
        self.intelligence = IntelligenceLayer()
        self.market_data = MarketData()
        self.risk_engine = RiskEngine()
        self.paper_broker = PaperBrokerAdapter(self.paper_portfolio)
        self.real_broker = ManualRealBrokerAdapter(self.portfolio)
        self.afp_tracker = AFPTracker()
        self.he_analyzer = HEAnalyzer()
        self.regime_detector = RegimeDetector()
        # Trailing Stop: tracks the highest price seen since entry for each position
        self._price_peaks: dict = {}          # {ticker: highest_price_seen}
        self.TRAILING_STOP_PCT = 0.05         # Sell if price drops 5% from peak
        self.TAKE_PROFIT_PCT = 0.15            # Hard take-profit at +15% (massive move)
        self.MIN_HOLD_CYCLES = 3               # Hold at least 3 cycles before considering sell
        self._last_buy_alerts: dict = {}
        self.telegram_alerts_enabled = True
        self.last_update_id = 0
        self.prediction_log_cooldown_seconds = int(os.environ.get("PREDICTION_LOG_COOLDOWN_SECONDS", 1800))
        self._last_prediction_logged_at: dict = {}
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if token:
            try:
                resp = http_requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=5).json()
                if resp.get("ok") and resp.get("result"):
                    self.last_update_id = resp["result"][-1]["update_id"]
            except Exception:
                pass
        logger.info(
            "Universe config: "
            f"tickers={len(WATCHLIST)} | "
            f"include_global={_include_global} | "
            f"extra_tickers={len(_extra_tickers_from_env())}"
        )
        self._process_lock_fd = None
        self._process_lock_path = os.path.join(
            os.path.dirname(TRADING_DB_FILE),
            "autonomous_bot.lock",
        )

    def _acquire_process_lock(self) -> bool:
        """Ensures only one bot loop runs across processes (Streamlit reloads/workers)."""
        if self._process_lock_fd is not None:
            return True
        try:
            fd = os.open(self._process_lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            self._process_lock_fd = fd
            return True
        except OSError:
            return False

    def _release_process_lock(self) -> None:
        if self._process_lock_fd is None:
            return
        try:
            fcntl.flock(self._process_lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._process_lock_fd)
        except OSError:
            pass
        self._process_lock_fd = None

    def _send_automatic_telegram(self, message: str, chat_id_override: str = None):
        """Sends Telegram alerts only when automatic notifications are enabled."""
        if not self.telegram_alerts_enabled:
            return
        send_telegram(message, chat_id_override)

    def _log_context_snapshot(self, context_snapshot):
        try:
            now_ts = datetime.utcnow().isoformat() + "Z"
            TradingDB.log_context(
                {
                    "timestamp": now_ts,
                    "global_score": float(getattr(context_snapshot, "global_score", 0.0) or 0.0),
                    "event_type": getattr(context_snapshot, "event_type", "unknown"),
                    "impact_level": getattr(context_snapshot, "impact_level", "unknown"),
                    "summary": getattr(context_snapshot, "summary", ""),
                    "raw_payload": context_snapshot.model_dump() if hasattr(context_snapshot, "model_dump") else {},
                }
            )
        except Exception as e:
            logger.debug(f"Context logging skipped: {e}")

    def _log_prediction_candidate(self, ticker: str, analysis, ticker_data: dict):
        now = time.time()
        last = self._last_prediction_logged_at.get(ticker, 0.0)
        if (now - last) < self.prediction_log_cooldown_seconds:
            return
        self._last_prediction_logged_at[ticker] = now

        tech = ticker_data.get("technical_data", {})
        full_tech = dict(tech)
        context_score = 0.0
        ml_out = Predictor.predict_multi_objective(full_tech, context_score=context_score)
        TradingDB.log_prediction(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "ticker": ticker,
                "prediction_horizon_days": int(ml_out.get("horizon_days", 3)),
                "predicted_prob": float(ml_out.get("probability", getattr(analysis, "ml_confidence", 0.5) or 0.5)),
                "predicted_return": float(ml_out.get("expected_return_3d", 0.0)),
                "resolved": 0,
                "metadata": {
                    "signal": getattr(analysis, "signal", "HOLD"),
                    "reasoning": getattr(analysis, "reasoning", ""),
                    "entry_price": float(ticker_data.get("current_price", 0.0) or 0.0),
                },
            }
        )

    def _resolve_pending_predictions(self):
        unresolved = TradingDB.load_predictions(limit=500, unresolved_only=True)
        if not unresolved:
            return

        now_utc = datetime.utcnow()
        for p in unresolved:
            try:
                ts = datetime.fromisoformat(str(p.get("timestamp", "")).replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue

            horizon = int(p.get("prediction_horizon_days", 3) or 3)
            if now_utc < (ts + timedelta(days=horizon)):
                continue

            ticker = p.get("ticker")
            if not ticker:
                continue

            meta = p.get("metadata", {}) or {}
            entry_price = float(meta.get("entry_price", 0.0) or 0.0)
            if entry_price <= 0:
                continue

            try:
                hist = yf.Ticker(ticker).history(period="10d")
                if hist is None or hist.empty:
                    continue
                realized_price = float(hist["Close"].iloc[-1])
                realized_ret = (realized_price - entry_price) / (entry_price + 1e-9)
                realized_label = 1 if realized_ret > 0.02 else 0
                TradingDB.resolve_prediction(int(p.get("id")), realized_ret, realized_label)
            except Exception:
                continue

    def _log_nav_snapshots(self, ticker_data: dict):
        try:
            real_prices = {
                tk: ticker_data.get(tk, {}).get("current_price", 0.0)
                for tk, qty in self.portfolio.positions.items()
                if qty > 0
            }
            real_snapshot = self.risk_engine.portfolio_risk_snapshot(
                cash_balance=self.portfolio.balance,
                positions=self.portfolio.positions,
                ticker_prices=real_prices,
            )
            TradingDB.log_nav(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "portfolio": "real",
                    "equity": float(real_snapshot.get("equity", self.portfolio.balance)),
                    "cash": float(self.portfolio.balance),
                    "invested": float(real_snapshot.get("invested_value", 0.0)),
                    "note": "cycle",
                }
            )

            paper_prices = {
                tk: ticker_data.get(tk, {}).get("current_price", 0.0)
                for tk, qty in self.paper_portfolio.positions.items()
                if qty > 0
            }
            paper_snapshot = self.risk_engine.portfolio_risk_snapshot(
                cash_balance=self.paper_portfolio.balance,
                positions=self.paper_portfolio.positions,
                ticker_prices=paper_prices,
            )
            TradingDB.log_nav(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "portfolio": "paper",
                    "equity": float(paper_snapshot.get("equity", self.paper_portfolio.balance)),
                    "cash": float(self.paper_portfolio.balance),
                    "invested": float(paper_snapshot.get("invested_value", 0.0)),
                    "note": "cycle",
                }
            )
        except Exception as e:
            logger.debug(f"NAV logging skipped: {e}")

    @staticmethod
    def _phase4_opportunity_score(analysis, data: dict, regime: dict, afp_info: dict, he_info: dict) -> float:
        ml_prob = float(getattr(analysis, "ml_confidence", 0.5) or 0.5)
        quant_score = float(getattr(analysis, "quant_score", 5) or 5) / 10.0

        tech = data.get("technical_data", {})
        copper_corr = float(tech.get("Realtime_Copper_Corr_20d", 0.0) or 0.0)
        copper_ret = float(tech.get("Macro_Copper_Ret", tech.get("Copper_Return", 0.0)) or 0.0)
        copper_boost = copper_corr * copper_ret * 8.0

        liquidity_score = float(tech.get("Liquidity_Score", 0.5) or 0.5)
        regime_name = regime.get("regime", "sideways")
        # AGRESIVO: bonifica operar en bear
        regime_boost = 0.08 if regime_name == "bull" else (0.10 if regime_name == "bear" else 0.0)

        afp_pressure = float(afp_info.get("pressure_score", 0.0) or 0.0)
        he_impact = float(he_info.get("impact_score", 0.0) or 0.0)

        score = (
            0.45 * ml_prob
            + 0.15 * quant_score
            + 0.12 * afp_pressure
            + 0.10 * he_impact
            + 0.10 * liquidity_score
            + 0.08 * copper_boost
            + regime_boost
        )
        return float(score)

    def _execute_paper_signal(self, ticker: str, signal: str, price: float, reasoning: str, confidence: float = 0.5):
        """Executes autonomous paper-trading orders for the 10M demo portfolio."""
        if price <= 0:
            return

        if signal == "BUY":
            if self.paper_portfolio.positions.get(ticker, 0) > 0:
                return

            suggested_qty = self.paper_portfolio.calculate_position_size(price, confidence=confidence)
            if suggested_qty <= 0:
                return

            market_prices = {
                tk: self.paper_portfolio.position_costs.get(tk, 0.0)
                for tk, qty in self.paper_portfolio.positions.items()
                if qty > 0
            }
            min_order_value = float(getattr(self.risk_engine, "min_order_clp", 0.0) or 0.0)
            min_qty = max(1, int(min_order_value / price)) if min_order_value > 0 else 1
            proposed_qty = int(suggested_qty)
            ok = False
            reason = ""
            while proposed_qty >= min_qty:
                ok, reason = self.risk_engine.validate_buy(
                    ticker=ticker,
                    proposed_qty=proposed_qty,
                    proposed_price=price,
                    cash_balance=self.paper_portfolio.balance,
                    positions=self.paper_portfolio.positions,
                    ticker_prices=market_prices,
                )
                if ok:
                    break
                reduced_qty = int(proposed_qty * 0.75)
                if reduced_qty == proposed_qty:
                    reduced_qty = proposed_qty - 1
                proposed_qty = reduced_qty

            if not ok:
                logger.info(f"[PAPER][RISK BLOCK] BUY {ticker} bloqueado: {reason}")
                return

            if proposed_qty < suggested_qty:
                logger.info(
                    f"[PAPER][RISK ADJUST] BUY {ticker}: qty ajustada {suggested_qty} -> {proposed_qty} para cumplir riesgo"
                )

            max_invest = proposed_qty * price
            self.paper_portfolio.execute_order(
                ticker=ticker,
                signal=signal,
                price=price,
                reasoning=reasoning,
                confidence=confidence,
                amount_to_invest=max_invest,
            )
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
            with ThreadPoolExecutor(max_workers=MARKET_DATA_MAX_WORKERS) as executor:
                fut_to_ticker = {executor.submit(self.market_data.get_comprehensive_data, t): t for t in WATCHLIST}
                for fut in as_completed(fut_to_ticker):
                    ticker = fut_to_ticker[fut]
                    ticker_data[ticker] = fut.result()

            promising_batch = {}
            for ticker, data in ticker_data.items():
                price = float(data.get("current_price", 0.0) or 0.0)
                if bool(data.get("is_active", False)) and price > 0:
                    promising_batch[ticker] = data

            if not promising_batch:
                return "ℹ️ No encontré acciones candidatas en este momento."

            final_analyses = self.intelligence.bulk_analyze(promising_batch, context_data=context_snapshot)

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
        market_regime = self.regime_detector.detect()
        
        # Snapshot global context once to avoid repeated LLM calls across chunks
        context_snapshot = self.intelligence.context_service.analyze_context()
        self._log_context_snapshot(context_snapshot)
        self._resolve_pending_predictions()
        
        # 1. Fetch Market Data ...
        ticker_data = {}
        with ThreadPoolExecutor(max_workers=MARKET_DATA_MAX_WORKERS) as executor:
            fut_to_ticker = {executor.submit(self.market_data.get_comprehensive_data, t): t for t in WATCHLIST}
            for fut in as_completed(fut_to_ticker):
                t = fut_to_ticker[fut]
                ticker_data[t] = fut.result()

        total_tickers = len(ticker_data)
        priced_tickers = sum(1 for d in ticker_data.values() if float(d.get("current_price", 0.0) or 0.0) > 0)
        active_tickers = sum(1 for d in ticker_data.values() if bool(d.get("is_active", False)))
        errored_tickers = sum(1 for d in ticker_data.values() if d.get("error"))
        logger.info(
            "Market data snapshot: "
            f"total={total_tickers} | priced={priced_tickers} | active={active_tickers} | errors={errored_tickers}"
        )

        # 1.5 Run Protections
        self._check_circuit_breakers(ticker_data)
        self._update_trailing_stops(ticker_data)

        # 2. Preparation
        promising_batch = {}
        for ticker, data in ticker_data.items():
            price = float(data.get("current_price", 0.0) or 0.0)
            is_candidate = (bool(data.get("is_active", False)) and price > 0)
            if is_candidate or ticker in self._get_active_tickers():
                promising_batch[ticker] = data

        if not promising_batch:
            fallback_limit = max(1, int(os.environ.get("FALLBACK_ANALYSIS_TICKERS", 10)))
            fallback_candidates = []
            for ticker, data in ticker_data.items():
                price = float(data.get("current_price", 0.0) or 0.0)
                if price <= 0:
                    continue
                tech = data.get("technical_data", {})
                momentum = abs(float(tech.get("DailyReturn_Pct", 0.0) or 0.0))
                volatility = abs(float(tech.get("Volatility_20d", 0.0) or 0.0))
                fallback_candidates.append((ticker, data, momentum + volatility))

            fallback_candidates.sort(key=lambda row: row[2], reverse=True)
            for ticker, data, _ in fallback_candidates[:fallback_limit]:
                promising_batch[ticker] = data

            if promising_batch:
                logger.info(
                    "No active tickers passed gatekeeper; using fallback analysis set "
                    f"of {len(promising_batch)} tickers with valid pricing."
                )
            else:
                error_examples = []
                for ticker, data in ticker_data.items():
                    msg = str(data.get("error", "")).strip()
                    if msg:
                        error_examples.append(f"{ticker}:{msg[:80]}")
                    if len(error_examples) >= 4:
                        break
                logger.warning(
                    "Promising batch remains empty after fallback; no valid prices available. "
                    f"Sample errors={error_examples if error_examples else 'none'}"
                )

        # 3. AI Analysis
        final_analyses = self.intelligence.bulk_analyze(promising_batch, context_data=context_snapshot)

        if not final_analyses and promising_batch:
            context_score = float(getattr(context_snapshot, "global_score", 0.0) or 0.0)
            repaired = {}
            for ticker, data in promising_batch.items():
                tech = data.get("technical_data", {}) or {}
                ml_outputs = Predictor.predict_multi_objective(tech, context_score=context_score)
                fallback_item = {
                    "ticker": ticker,
                    "ml_prob": float(ml_outputs.get("probability", 0.5) or 0.5),
                    "ml_expected_return_3d": float(ml_outputs.get("expected_return_3d", 0.0) or 0.0),
                }
                repaired[ticker] = self.intelligence._build_ml_fallback_analysis(fallback_item, context_score)
            final_analyses = repaired
            logger.warning(
                "bulk_analyze returned empty results; applied deterministic ML fallback "
                f"for {len(final_analyses)} tickers."
            )

        # 4. Execution
        buy_candidates = []
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for ticker, analysis in final_analyses.items():
            price = ticker_data.get(ticker, {}).get("current_price", 0)
            confidence = float(getattr(analysis, "ml_confidence", getattr(analysis, "confidence", 0.5)) or 0.5)
            signal = getattr(analysis, "signal", "HOLD")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

            afp_info = self.afp_tracker.estimate_pressure(
                ticker,
                ticker_data.get(ticker, {}).get("technical_data", {}),
            )
            he_info = self.he_analyzer.analyze(ticker, ticker_data.get(ticker, {}).get("news_text", ""))
            phase4_score = self._phase4_opportunity_score(
                analysis,
                ticker_data.get(ticker, {}),
                market_regime,
                afp_info,
                he_info,
            )

            if analysis.signal == "BUY":
                self._log_prediction_candidate(ticker, analysis, ticker_data.get(ticker, {}))
                buy_candidates.append((ticker, analysis, phase4_score, afp_info, he_info))
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
        logger.info(
            "Regime detector: "
            f"regime={market_regime.get('regime')} | "
            f"confidence={market_regime.get('confidence', 0.0):.2f} | "
            f"ret20d={market_regime.get('ret_20d', 0.0):.2%}"
        )

        paper_prices = {
            tk: ticker_data.get(tk, {}).get("current_price", 0.0)
            for tk, qty in self.paper_portfolio.positions.items()
            if qty > 0
        }
        risk_snapshot = self.risk_engine.portfolio_risk_snapshot(
            cash_balance=self.paper_portfolio.balance,
            positions=self.paper_portfolio.positions,
            ticker_prices=paper_prices,
        )

        paper_vols = {
            tk: ticker_data.get(tk, {}).get("technical_data", {}).get("Volatility_20d", 0.02)
            for tk, qty in self.paper_portfolio.positions.items()
            if qty > 0
        }
        paper_returns_history = {
            tk: ticker_data.get(tk, {}).get("technical_data", {}).get("Recent_Returns_30d", [])
            for tk, qty in self.paper_portfolio.positions.items()
            if qty > 0
        }
        corr_matrix = self.risk_engine.build_correlation_matrix(paper_returns_history)
        var_snapshot = self.risk_engine.estimate_parametric_var(
            cash_balance=self.paper_portfolio.balance,
            positions=self.paper_portfolio.positions,
            ticker_prices=paper_prices,
            returns_vol=paper_vols,
            correlation_matrix=corr_matrix,
            confidence=0.95,
        )
        stress_snapshot = self.risk_engine.run_stress_tests(
            cash_balance=self.paper_portfolio.balance,
            positions=self.paper_portfolio.positions,
            ticker_prices=paper_prices,
        )

        logger.info(
            "Risk snapshot (PAPER): "
            f"invested={risk_snapshot['invested_pct']:.1%} | "
            f"max_pos={risk_snapshot['max_single_position_pct']:.1%} | "
            f"max_sector={risk_snapshot['max_sector_pct']:.1%} | "
            f"open_positions={int(risk_snapshot['open_positions'])} | "
            f"VaR95_1d={var_snapshot['var_1d_pct']:.1%} | "
            f"VaR_method={var_snapshot.get('method', 'diagonal')}"
        )
        logger.info(
            "Stress snapshot (PAPER): "
            f"mild={stress_snapshot['shock_mild_pct']:.1%} | "
            f"moderate={stress_snapshot['shock_moderate_pct']:.1%} | "
            f"severe={stress_snapshot['shock_severe_pct']:.1%}"
        )
        self._log_nav_snapshots(ticker_data)

        if buy_candidates:
            buy_candidates.sort(key=lambda row: row[2], reverse=True)
            # Alertas para REAL portfolio (solo recomendaciones Telegram)
            current_time = time.time()
            cooldown_hours = 4
            actionable_buys = []

            for ticker, analysis, phase4_score, afp_info, he_info in buy_candidates:
                if ticker in self.portfolio.positions and self.portfolio.positions[ticker] > 0:
                    continue

                last_alert = self._last_buy_alerts.get(ticker, 0)
                if (current_time - last_alert) <= (cooldown_hours * 3600):
                    continue

                price = ticker_data[ticker]["current_price"]
                if price <= 0:
                    continue

                actionable_buys.append((ticker, analysis, price, phase4_score, afp_info, he_info))

            if actionable_buys:
                capital_disponible = max(self.portfolio.balance, 0)
                capital_por_accion = capital_disponible / len(actionable_buys)

                current_prices = {
                    tk: ticker_data.get(tk, {}).get("current_price", 0.0)
                    for tk, qty in self.portfolio.positions.items()
                    if qty > 0
                }

                for ticker, analysis, price, phase4_score, afp_info, he_info in actionable_buys:
                    suggested_shares = int(capital_por_accion / price)
                    suggested_size_clp = suggested_shares * price
                    urgency = he_info.get("urgency", "low")
                    urgency_emoji = "🚨" if urgency == "high" else ("🟠" if urgency == "medium" else "🟢")
                    urgency_label = "Evento material urgente" if urgency == "high" else (
                        "Evento material monitoreado" if urgency == "medium" else "Sin urgencia material"
                    )
                    afp_label = (
                        "presión compradora AFP"
                        if afp_info.get("pressure_type") == "buying"
                        else "presión vendedora AFP"
                        if afp_info.get("pressure_type") == "selling"
                        else "flujo AFP neutral"
                    )

                    risk_ok = False
                    risk_reason = ""
                    if suggested_shares > 0:
                        risk_ok, risk_reason = self.risk_engine.validate_buy(
                            ticker=ticker,
                            proposed_qty=suggested_shares,
                            proposed_price=price,
                            cash_balance=self.portfolio.balance,
                            positions=self.portfolio.positions,
                            ticker_prices=current_prices,
                        )

                    if suggested_shares <= 0:
                           msg = (f"{urgency_emoji} *ALERTA DE COMPRA* {urgency_emoji}\n"
                               f"📌 *Acción:* {ticker}\n"
                               f"💰 *Precio Recomendado:* ${price:,.0f}\n"
                               f"🎯 *Score Fase 4:* {phase4_score:.3f}\n"
                               f"🧭 *Régimen de mercado:* {market_regime.get('regime', 'sideways')}\n"
                               f"🏛️ *Señal AFP:* {afp_label} ({float(afp_info.get('pressure_score', 0.0)):+.2f})\n"
                               f"🗞️ *HE Analyzer:* {urgency_label} ({float(he_info.get('impact_score', 0.0)):+.2f})\n"
                               f"🏦 *Capital Disponible:* ${capital_disponible:,.0f} CLP\n"
                               f"⚠️ *Sin capital suficiente* para comprar 1 acción al precio actual.\n"
                               f"💡 *Razón:* {analysis.reasoning}")
                    elif not risk_ok:
                        msg = (f"🟡 *SEÑAL BUY BLOQUEADA POR RIESGO* 🟡\n"
                               f"📌 *Acción:* {ticker}\n"
                               f"💰 *Precio:* ${price:,.0f}\n"
                               f"🎯 *Score Fase 4:* {phase4_score:.3f}\n"
                               f"🧭 *Régimen de mercado:* {market_regime.get('regime', 'sideways')}\n"
                               f"🏛️ *Señal AFP:* {afp_label} ({float(afp_info.get('pressure_score', 0.0)):+.2f})\n"
                               f"🗞️ *HE Analyzer:* {urgency_label} ({float(he_info.get('impact_score', 0.0)):+.2f})\n"
                               f"⚖️ *Motivo de bloqueo:* {risk_reason}\n"
                               f"💡 *Señal IA:* {analysis.reasoning}")
                    else:
                           msg = (f"{urgency_emoji} *ALERTA DE COMPRA* {urgency_emoji}\n"
                               f"📌 *Acción:* {ticker}\n"
                               f"💰 *Precio Recomendado:* ${price:,.0f}\n"
                               f"🎯 *Score Fase 4:* {phase4_score:.3f}\n"
                               f"🧭 *Régimen de mercado:* {market_regime.get('regime', 'sideways')}\n"
                               f"🏛️ *Señal AFP:* {afp_label} ({float(afp_info.get('pressure_score', 0.0)):+.2f})\n"
                               f"🗞️ *HE Analyzer:* {urgency_label} ({float(he_info.get('impact_score', 0.0)):+.2f})\n"
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

        if not self._acquire_process_lock():
            logger.warning("Autonomous bot process lock is already held; skipping duplicate process start.")
            return

        with AutonomousBot._START_GUARD_LOCK:
            if AutonomousBot._IS_RUNNING:
                logger.warning("Autonomous bot loop already running; skipping duplicate start request.")
                self._release_process_lock()
                return
            AutonomousBot._IS_RUNNING = True

        logger.info("=== Autonomous Trading Bot Started ===")
        try:
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
        finally:
            with AutonomousBot._START_GUARD_LOCK:
                AutonomousBot._IS_RUNNING = False
            self._release_process_lock()

if __name__ == "__main__":
    import sys
    bot = AutonomousBot()
    test_mode = "--test-run" in sys.argv
    bot.start(once=test_mode)
