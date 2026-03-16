import json
import os
import threading
import time
import importlib

import pandas as pd
import plotly.express as px
import streamlit as st
import datetime
import pytz
from dotenv import load_dotenv


if importlib.util.find_spec("streamlit_autorefresh") is not None:
    st_autorefresh = importlib.import_module("streamlit_autorefresh").st_autorefresh
else:
    def st_autorefresh(*args, **kwargs):
        return 0

from afp_tracker import AFPTracker
from database import TradingDB
from market_data import MarketData
from paper_trading import INITIAL_BALANCE_CLP, PaperPortfolio, PaperTradingDB
from paths import BACKTEST_RESULTS_FILE, BACKTEST_TRADES_FILE
from risk_engine import RiskEngine
from scenario_simulator import PRESET_SCENARIOS, ScenarioSimulator
from stats import StatsEngine
from trading_bot import AutonomousBot
from universe import get_trading_watchlist

load_dotenv()

st.set_page_config(page_title="Aureus Wealth", page_icon="💼", layout="wide")

# --- Horario Bolsa de Santiago ---
CHILE_TZ = pytz.timezone("America/Santiago")
MARKET_OPEN = datetime.time(9, 30)
MARKET_CLOSE = datetime.time(16, 0)


def chile_now() -> datetime.datetime:
    return datetime.datetime.now(CHILE_TZ)

def is_market_open():
    now = chile_now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


# --- Mostrar portafolio siempre, pero congelar operaciones y precios si el mercado está cerrado ---
if not is_market_open():
    st.warning("⏳ Mercado cerrado: la Bolsa de Santiago opera de 09:30 a 16:00. Se muestran los valores de cierre más recientes. No se pueden realizar operaciones ni actualizar precios en este momento.")
    market_open = False
else:
    market_open = True

_BOT_THREAD_LOCK = threading.Lock()
_BOT_THREAD = None
_CYCLE_THREAD = None
_LAST_CYCLE_LAUNCHED_AT = 0.0


def _is_hosted_runtime() -> bool:
    return any(
        os.environ.get(flag)
        for flag in ("RENDER", "RENDER_SERVICE_ID", "RAILWAY_ENVIRONMENT", "K_SERVICE")
    )



if "risk_profile" not in st.session_state:
    st.session_state.risk_profile = "agresivo"
if "paper_portfolio" not in st.session_state:
    st.session_state.paper_portfolio = PaperPortfolio()
if "bot_thread_started" not in st.session_state:
    st.session_state.bot_thread_started = False
if "show_reset_dialog" not in st.session_state:
    st.session_state.show_reset_dialog = False











def run_bot_in_background():
    AutonomousBot().start()


def run_bot_single_cycle() -> None:
    AutonomousBot().start(once=True)


def _ensure_single_bot_thread() -> None:
    global _BOT_THREAD
    with _BOT_THREAD_LOCK:
        if _BOT_THREAD is not None and _BOT_THREAD.is_alive():
            return
        _BOT_THREAD = threading.Thread(target=run_bot_in_background, daemon=True)
        _BOT_THREAD.start()


def _maybe_schedule_single_cycle(interval_seconds: int) -> None:
    """Streamlit-safe scheduler: launches at most one non-overlapping cycle per interval."""
    global _CYCLE_THREAD, _LAST_CYCLE_LAUNCHED_AT
    now = time.time()
    with _BOT_THREAD_LOCK:
        if _CYCLE_THREAD is not None and _CYCLE_THREAD.is_alive():
            return
        if (now - _LAST_CYCLE_LAUNCHED_AT) < max(15, int(interval_seconds)):
            return
        _LAST_CYCLE_LAUNCHED_AT = now
        _CYCLE_THREAD = threading.Thread(target=run_bot_single_cycle, daemon=True)
        _CYCLE_THREAD.start()



# Iniciar el bot solo una vez por sesión para evitar loops duplicados en cada refresh.
auto_start_bot_default = False if _is_hosted_runtime() else True
auto_start_bot = str(os.environ.get("APP_AUTO_START_BOT", str(auto_start_bot_default))).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

streamlit_single_cycle_default = True if _is_hosted_runtime() else False
streamlit_single_cycle_mode = str(
    os.environ.get("APP_STREAMLIT_SINGLE_CYCLE_MODE", str(streamlit_single_cycle_default))
).strip().lower() in {"1", "true", "yes", "on"}

bot_interval_seconds = int(os.environ.get("TRADING_INTERVAL_SECONDS", "120"))

if auto_start_bot and not st.session_state.bot_thread_started:
    _ensure_single_bot_thread()
    st.session_state.bot_thread_started = True
elif streamlit_single_cycle_mode:
    _maybe_schedule_single_cycle(bot_interval_seconds)

st_autorefresh(interval=20 * 1000, key="aureus_refresh")
# No reinicializar el portafolio en cada recarga, solo si no existe en session_state

paper = st.session_state.paper_portfolio
# Sincroniza estado desde DB en cada rerun para reflejar operaciones del bot en background.
paper.balance = PaperTradingDB.load_state()
paper.positions = PaperTradingDB.load_positions()
paper.position_costs = PaperTradingDB.load_position_costs()
paper_trades = PaperTradingDB.load_trade_log()
market_data = MarketData()
risk_engine = RiskEngine()
scenario_sim = ScenarioSimulator()
afp_tracker = AFPTracker()

initial = INITIAL_BALANCE_CLP
cash_value = paper.balance
active_positions = {ticker: qty for ticker, qty in paper.positions.items() if qty > 0}


# Si el mercado está abierto, usa precios actuales; si está cerrado, usa último precio almacenado o de cierre
total_value = paper.balance
portfolio_distribution = {"Cash": paper.balance}
latest_ticker_data = {}
for ticker, qty in active_positions.items():
    px_now = 0.0
    try:
        data = market_data.get_comprehensive_data(ticker)
        latest_ticker_data[ticker] = data
        if market_open:
            px_now = data.get("current_price", 0.0) or data.get("close_price", 0.0)
        else:
            # Mercado cerrado: prioriza cierre, pero usa current_price si es lo único disponible.
            px_now = data.get("close_price", 0.0) or data.get("current_price", 0.0)
    except Exception:
        pass
    if px_now and px_now > 0:
        total_value += qty * px_now
        portfolio_distribution[ticker] = qty * px_now
    else:
        # Si no hay precio, usar el costo promedio
        avg_cost = paper.position_costs.get(ticker, 0.0)
        total_value += qty * avg_cost
        portfolio_distribution[ticker] = qty * avg_cost

# Ahora que total_value está definido, calcula delta_value y delta_pct
delta_value = total_value - initial
delta_pct = (delta_value / initial) * 100 if initial else 0

# Store per-ticker P&L for display
ticker_pnl = {}
for ticker, qty in active_positions.items():
    data = latest_ticker_data.get(ticker, {})
    px_now = data.get("current_price", 0.0) or paper.position_costs.get(ticker, 0.0)
    cost = paper.position_costs.get(ticker, 0.0)
    if cost > 0:
        pnl = (px_now - cost) * qty
        pnl_pct = ((px_now - cost) / cost) * 100
        ticker_pnl[ticker] = {"pnl": pnl, "pct": pnl_pct, "current": px_now, "cost": cost}
    else:
        ticker_pnl[ticker] = {"pnl": 0, "pct": 0, "current": px_now, "cost": 0}


ticker_prices = {
    tk: (portfolio_distribution.get(tk, 0.0) / qty) if qty > 0 else 0.0
    for tk, qty in active_positions.items()
}
returns_vol = {}
returns_history = {}
for tk in active_positions.keys():
    try:
        tech = latest_ticker_data.get(tk, {}).get("technical_data", {})
        returns_vol[tk] = float(tech.get("Volatility_20d", 0.02) or 0.02)
        returns_history[tk] = tech.get("Recent_Returns_30d", []) or []
    except Exception:
        returns_vol[tk] = 0.02
        returns_history[tk] = []

corr_matrix = risk_engine.build_correlation_matrix(returns_history)
risk_snapshot = risk_engine.portfolio_risk_snapshot(
    cash_balance=paper.balance,
    positions=paper.positions,
    ticker_prices=ticker_prices,
)
var_snapshot = risk_engine.estimate_parametric_var(
    cash_balance=paper.balance,
    positions=paper.positions,
    ticker_prices=ticker_prices,
    returns_vol=returns_vol,
    correlation_matrix=corr_matrix,
    confidence=0.95,
)
stress_snapshot = risk_engine.run_stress_tests(
    cash_balance=paper.balance,
    positions=paper.positions,
    ticker_prices=ticker_prices,
)

st.sidebar.title("Aureus Wealth")
page = st.sidebar.radio(
    "Secciones",
    [
        "Portafolio",
        "Ofertas",
        "Prueba Ayer",
        "Predicciones",
        "Explainability",
        "Risk",
        "Escenarios",
        "Backtest",
        "AFP flows",
    ],
)



# Asegura que market_open esté definido antes de usarlo
if "market_open" not in locals():
    try:
        market_open = market_data.is_santiago_market_open()
    except Exception:
        market_open = False

st.session_state.risk_profile = "agresivo"
risk_profile = "agresivo"

st.title("Aureus Wealth — Plataforma de Decisión")
st.caption(
    f"Perfil: {risk_profile} | Mercado {'abierto' if market_open else 'cerrado'} | "
    f"Hora Chile {chile_now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Actualización automática 20s"
)

if page == "Portafolio":
    c1, c2, c3 = st.columns(3)
    c1.metric("Cash Libre", f"${cash_value:,.0f}")
    c2.metric(
        "Valor Total",
        f"${total_value:,.0f}",
        delta=f"${delta_value:,.0f} ({delta_pct:.2f}%)",
        delta_color="normal",
    )
    c3.metric("Operaciones", len(paper_trades))

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Distribución")
        dist_df = pd.DataFrame(list(portfolio_distribution.items()), columns=["Activo", "Valor"])
        if not dist_df.empty and float(dist_df["Valor"].sum()) > 0:
            pie = px.pie(dist_df, values="Valor", names="Activo", hole=0.5)
            st.plotly_chart(pie, width="stretch")
        else:
            st.info("Sin datos suficientes")

    with right:
        st.subheader("Posiciones")
        if active_positions:
            for tk, qty in active_positions.items():
                pnl_data = ticker_pnl.get(tk, {})
                pnl_val = pnl_data.get("pnl", 0)
                pnl_pct = pnl_data.get("pct", 0)
                px_now = pnl_data.get("current", 0)
                
                color = "green" if pnl_val >= 0 else "red"
                sign = "+" if pnl_val >= 0 else ""
                
                st.markdown(f"**{tk}** · {qty:.0f} acc")
                st.markdown(
                    f"<div style='font-size: 0.9em; color: gray;'>Precio: ${px_now:,.0f} | "
                    f"<span style='color: {color}; font-weight: bold;'>{sign}${pnl_val:,.0f} ({sign}{pnl_pct:.2f}%)</span></div>",
                    unsafe_allow_html=True
                )
                st.write("")
        else:
            st.caption("No hay posiciones abiertas")

    # --- Historial de Operaciones ---
    st.subheader("📊 Historial de Operaciones")
    if paper_trades:
        # Mostrar últimas 20 operaciones en orden descendente (más recientes primero)
        trades_df = pd.DataFrame(paper_trades[:20])
        
        # Asegurarse de que el columnas existan
        display_cols = []
        if "timestamp" in trades_df.columns:
            display_cols.append("timestamp")
        elif "Hora" in trades_df.columns:
            display_cols.append("Hora")
        elif "hora" in trades_df.columns:
            display_cols.append("hora")
            
        if "ticker" in trades_df.columns:
            display_cols.append("ticker")
        elif "Ticker" in trades_df.columns:
            display_cols.append("Ticker")
        elif "Activo" in trades_df.columns:
            display_cols.append("Activo")
            
        if "signal" in trades_df.columns:
            display_cols.append("signal")
        elif "Signal" in trades_df.columns:
            display_cols.append("Signal")
        elif "Acción" in trades_df.columns:
            display_cols.append("Acción")
        elif "accion" in trades_df.columns:
            display_cols.append("accion")
            
        if "price" in trades_df.columns:
            display_cols.append("price")
        elif "Precio" in trades_df.columns:
            display_cols.append("Precio")
            
        if "quantity" in trades_df.columns:
            display_cols.append("quantity")
        elif "Cantidad" in trades_df.columns:
            display_cols.append("Cantidad")
            
        if "reasoning" in trades_df.columns:
            display_cols.append("reasoning")
        elif "Razón" in trades_df.columns:
            display_cols.append("Razón")
        
        # Filtrar solo columnas que existen
        display_cols = [col for col in display_cols if col in trades_df.columns]
        
        if display_cols:
            trades_display = trades_df[display_cols].copy()
            # Renombrar columnas para mejor visualización
            rename_map = {
                "timestamp": "⏰ Hora",
                "Hora": "⏰ Hora",
                "hora": "⏰ Hora",
                "ticker": "📌 Activo",
                "Ticker": "📌 Activo",
                "Activo": "📌 Activo",
                "signal": "✅ Acción",
                "Signal": "✅ Acción",
                "Acción": "✅ Acción",
                "accion": "✅ Acción",
                "price": "💰 Precio",
                "Precio": "💰 Precio",
                "quantity": "📊 Cantidad",
                "Cantidad": "📊 Cantidad",
                "reasoning": "📝 Razón",
                "Razón": "📝 Razón",
                "confidence": "🎯 Confianza",
                "Confianza": "🎯 Confianza"
            }
            trades_display = trades_display.rename(columns=rename_map)
            if "✅ Acción" in trades_display.columns:
                trades_display["✅ Acción"] = (
                    trades_display["✅ Acción"].astype(str).str.upper().map({"BUY": "BUY", "SELL": "SELL"}).fillna(trades_display["✅ Acción"])
                )
            st.dataframe(trades_display, hide_index=True, use_container_width=True)
        else:
            st.info("No hay datos de operaciones disponibles")
    else:
        st.caption("Sin operaciones registradas todavía")

    # --- Reiniciar Demo ---
    st.markdown("---")
    if st.button("🔄 Reiniciar Demo", type="secondary"):
        st.session_state.show_reset_dialog = True

    if st.session_state.get("show_reset_dialog", False):
        with st.container(border=True):
            st.warning("⚠️ Esto borrará todas las posiciones e historial de operaciones.")
            reset_amount = st.number_input(
                "¿Con cuánto capital quieres empezar? (CLP)",
                min_value=1,
                max_value=1_000_000_000,
                value=int(INITIAL_BALANCE_CLP),
                step=1_000,
                format="%d",
            )
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("✅ Confirmar reinicio", type="primary"):
                    PaperTradingDB.reset(new_balance=float(reset_amount))
                    st.session_state.paper_portfolio = PaperPortfolio()
                    st.session_state.show_reset_dialog = False
                    st.success(f"Demo reiniciado con ${reset_amount:,} CLP 🎉")
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancelar"):
                    st.session_state.show_reset_dialog = False
                    st.rerun()


elif page == "Ofertas":
    st.subheader("📋 Ofertas — Order Book Trading")
    st.caption("Comisión: 0.14% por transacción | Precio dentro de ±10% del último precio")

    # Pending orders
    pending_orders = TradingDB.load_orders(limit=100, status_filter="PENDING")
    if pending_orders:
        st.markdown(f"### 🟡 Ofertas pendientes ({len(pending_orders)})")
        reserved_buy = sum(o["total_cost"] for o in pending_orders if o["side"] == "BUY")
        if reserved_buy > 0:
            st.info(f"💰 Cash reservado en ofertas BUY: ${reserved_buy:,.0f} CLP")
        pending_df = pd.DataFrame([{
            "ID": o["id"],
            "Ticker": o["ticker"],
            "Lado": o["side"],
            "Precio": f"${o['offer_price']:,.0f}",
            "Cantidad": int(o["quantity"]),
            "Comisión": f"${o['commission_clp']:,.0f}",
            "Total": f"${o['total_cost']:,.0f}",
            "Creada": o["created_at"][:19] if o["created_at"] else "",
        } for o in pending_orders])
        st.dataframe(pending_df, hide_index=True, use_container_width=True)
        st.caption("Confirma con `/confirmar ID` o cancela con `/cancelar ID` via Telegram.")
    else:
        st.info("No hay ofertas pendientes.")

    # Order history
    st.markdown("---")
    history_orders = TradingDB.load_orders(limit=100)
    non_pending = [o for o in history_orders if o["status"] != "PENDING"]
    if non_pending:
        st.markdown(f"### 📜 Historial de ofertas ({len(non_pending)})")
        hist_df = pd.DataFrame([{
            "ID": o["id"],
            "Ticker": o["ticker"],
            "Lado": o["side"],
            "Precio": f"${o['offer_price']:,.0f}",
            "Cantidad": int(o["quantity"]),
            "Total": f"${o['total_cost']:,.0f}",
            "Estado": o["status"],
            "Creada": o["created_at"][:19] if o["created_at"] else "",
            "Resuelta": o["resolved_at"][:19] if o.get("resolved_at") else "",
        } for o in non_pending])
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
    else:
        st.caption("Sin historial de ofertas todavía.")

elif page == "Prueba Ayer":
    st.title("Simulación de Portafolio - Día de Ayer (Trading Automático)")
    st.info("Esta vista simula el portafolio como si recién abriera el mercado ayer a las 9:30am. Pulsa ACTIVAR para iniciar el trading automático demo con 10 millones.")

    # --- Simulación progresiva de "Prueba Ayer" ---
    from datetime import datetime, timedelta, time as dtime
    fecha_ayer = (datetime.now() - timedelta(days=1)).date()
    hora_apertura = dtime(hour=9, minute=30)
    hora_cierre = dtime(hour=16, minute=0)
    dt_inicio = datetime.combine(fecha_ayer, hora_apertura)
    dt_fin = datetime.combine(fecha_ayer, hora_cierre)

    # Estado de la simulación
    if 'demo_ayer_activado' not in st.session_state:
        st.session_state.demo_ayer_activado = False
    if 'demo_ayer_datetime' not in st.session_state:
        st.session_state.demo_ayer_datetime = dt_inicio
    if 'demo_ayer_portfolio' not in st.session_state:
        st.session_state.demo_ayer_portfolio = PaperPortfolio()
        st.session_state.demo_ayer_portfolio.balance = 10_000_000.0
        st.session_state.demo_ayer_portfolio.positions = {}
        st.session_state.demo_ayer_portfolio.position_costs = {}
    if 'demo_ayer_trades' not in st.session_state:
        st.session_state.demo_ayer_trades = []

    if not st.session_state.demo_ayer_activado:
        if st.button("ACTIVAR TRADING AUTOMÁTICO DE AYER"):
            st.session_state.demo_ayer_activado = True
        else:
            st.stop()

    # Mostrar hora simulada
    hora_sim = st.session_state.demo_ayer_datetime
    st.markdown(f"### Hora simulada: {hora_sim.strftime('%Y-%m-%d %H:%M')}")

    # Ejecutar ciclo de trading solo si no hemos llegado al cierre
    if hora_sim <= dt_fin:
        demo_bot = AutonomousBot()
        demo_bot.paper_portfolio = st.session_state.demo_ayer_portfolio
        # Simula decisiones de trading con precios de ese minuto
        for ticker in get_trading_watchlist():
            try:
                data = demo_bot.market_data.get_comprehensive_data(ticker, date=hora_sim.date())
                price = data.get("current_price", 0.0)
                if price > 0:
                    demo_bot._execute_paper_signal(ticker, "BUY", price, f"Demo auto {hora_sim.strftime('%H:%M')}", confidence=0.5)
                    st.session_state.demo_ayer_trades.append({"hora": hora_sim.strftime('%H:%M'), "ticker": ticker, "accion": "BUY", "precio": price})
            except Exception:
                continue
        # Avanza la hora simulada 1 minuto simulado = 10 segundos reales
        st.session_state.demo_ayer_datetime = hora_sim + timedelta(minutes=1)
        st_autorefresh(interval=10 * 1000, key="aureus_demo_ayer")
    else:
        st.success("¡Fin de la simulación de ayer!")

    # Mostrar métricas y portafolio
    st.metric("Valor Total", f"${st.session_state.demo_ayer_portfolio.get_total_value(MarketData()):,.0f}")
    st.metric("Cash Libre", f"${st.session_state.demo_ayer_portfolio.balance:,.0f}")
    st.write("Distribución de portafolio:")
    dist_df = pd.DataFrame(list(st.session_state.demo_ayer_portfolio.positions.items()), columns=["Activo", "Cantidad"])
    if not dist_df.empty and float(dist_df["Cantidad"].sum()) > 0:
        st.dataframe(dist_df, hide_index=True)
    else:
        st.info("Sin posiciones abiertas.")

    if st.session_state.demo_ayer_trades:
        st.subheader("Historial de operaciones demo")
        st.dataframe(pd.DataFrame(st.session_state.demo_ayer_trades), hide_index=True)

elif page == "Predicciones":
    st.subheader("Predicciones registradas")
    preds = TradingDB.load_predictions(limit=500)
    if preds:
        df = pd.DataFrame(preds)
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("Sin predicciones registradas todavía")

elif page == "Explainability":
    st.subheader("Explainability de señales")
    preds = TradingDB.load_predictions(limit=200)
    if preds:
        df = pd.DataFrame(preds)
        if "metadata" in df.columns:
            df["reasoning"] = df["metadata"].apply(lambda m: (m or {}).get("reasoning", ""))
            df["signal"] = df["metadata"].apply(lambda m: (m or {}).get("signal", ""))
        view_cols = [c for c in ["timestamp", "ticker", "predicted_prob", "predicted_return", "signal", "reasoning"] if c in df.columns]
        st.dataframe(df[view_cols], width="stretch", hide_index=True)
    else:
        st.info("Sin datos de explainability")

elif page == "Risk":
    st.subheader("Riesgo agregado")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Exposición", f"{risk_snapshot['invested_pct']:.1%}")
    r2.metric("Max Activo", f"{risk_snapshot['max_single_position_pct']:.1%}")
    r3.metric("Max Sector", f"{risk_snapshot['max_sector_pct']:.1%}")
    r4.metric("VaR 1D 95%", f"{var_snapshot['var_1d_pct']:.1%}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Stress Mild", f"{stress_snapshot['shock_mild_pct']:.1%}")
    s2.metric("Stress Moderate", f"{stress_snapshot['shock_moderate_pct']:.1%}")
    s3.metric("Stress Severe", f"{stress_snapshot['shock_severe_pct']:.1%}")

    st.caption(f"Método VaR: {var_snapshot.get('method', 'diagonal')}")

    if corr_matrix:
        cm = pd.DataFrame(corr_matrix)
        st.subheader("Matriz de correlación")
        st.dataframe(cm, width="stretch")

    st.subheader("Métricas institucionales")
    metrics = StatsEngine.calculate_metrics(portfolio="paper")
    st.json(metrics)

elif page == "Escenarios":
    st.subheader("Simulador de escenarios")
    preset = st.selectbox("Escenario", list(PRESET_SCENARIOS.keys()))
    use_custom = st.checkbox("Usar shock personalizado")
    custom = None
    if use_custom:
        custom = st.slider("Shock mercado (%)", min_value=-30.0, max_value=30.0, value=-7.0, step=0.5) / 100.0

    if st.button("Simular"):
        sim = scenario_sim.run(
            positions=paper.positions,
            prices=ticker_prices,
            cash_balance=paper.balance,
            scenario_name=preset,
            custom_shock_pct=custom,
        )
        st.metric("Equity inicial", f"${sim['initial_equity']:,.0f}")
        st.metric("Equity simulado", f"${sim['shocked_equity']:,.0f}")
        st.metric("Impacto", f"${sim['pnl_clp']:,.0f} ({sim['pnl_pct']:.2%})")
        if sim.get("per_ticker"):
            st.dataframe(pd.DataFrame(sim["per_ticker"]).T, width="stretch")

elif page == "Backtest":
    st.subheader("Resultados de backtest")
    if os.path.exists(BACKTEST_RESULTS_FILE):
        bt_df = pd.read_csv(BACKTEST_RESULTS_FILE)
        if "Date" in bt_df.columns and "Equity" in bt_df.columns:
            fig = px.line(bt_df, x="Date", y="Equity", title="Equity Curve")
            st.plotly_chart(fig, width="stretch")
        st.dataframe(bt_df.tail(200), width="stretch", hide_index=True)
    else:
        st.info("No existe backtest_results.csv aún")

    if os.path.exists(BACKTEST_TRADES_FILE):
        tr_df = pd.read_csv(BACKTEST_TRADES_FILE)
        st.subheader("Trades backtest")
        st.dataframe(tr_df.tail(200), width="stretch", hide_index=True)

elif page == "AFP flows":
    st.subheader("Estimador de presión AFP")
    tickers = list(active_positions.keys())
    if not tickers:
        tickers = get_trading_watchlist(include_global=False)[:12]

    rows = []
    for tk in tickers:
        data = latest_ticker_data.get(tk)

