import json
import os
import threading
import time
import importlib

import pandas as pd
import plotly.express as px
import streamlit as st
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
paper_trades = PaperTradingDB.load_trade_log()
market_data = MarketData()
risk_engine = RiskEngine()
scenario_sim = ScenarioSimulator()
afp_tracker = AFPTracker()

initial = INITIAL_BALANCE_CLP
cash_value = paper.balance
active_positions = {ticker: qty for ticker, qty in paper.positions.items() if qty > 0}

# Siempre intenta usar el precio más actualizado posible para el valor total y la distribución
total_value = paper.balance
portfolio_distribution = {"Cash": paper.balance}
latest_ticker_data = {}
for ticker, qty in active_positions.items():
    px_now = 0.0
    try:
        data = market_data.get_comprehensive_data(ticker)
        latest_ticker_data[ticker] = data
        px_now = data.get("current_price", 0.0)
    except Exception:
        pass
    if px_now and px_now > 0:
        total_value += qty * px_now
        portfolio_distribution[ticker] = qty * px_now
    else:
        # Si no hay precio actual, usar el costo promedio
        avg_cost = paper.position_costs.get(ticker, 0.0)
        total_value += qty * avg_cost
        portfolio_distribution[ticker] = qty * avg_cost

# Ahora que total_value está definido, calcula delta_value y delta_pct
delta_value = total_value - initial
delta_pct = (delta_value / initial) * 100 if initial else 0


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
                st.write(f"**{tk}** · {qty:.0f}")
        else:
            st.caption("No hay posiciones abiertas")

    st.subheader("Historial de trading")
    if paper_trades:
        st.dataframe(pd.DataFrame(paper_trades), width="stretch", hide_index=True)
    else:
        st.info("No hay operaciones aún")

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
        if data is None:
            data = market_data.get_comprehensive_data(tk)
        tech = data.get("technical_data", {})
        afp = afp_tracker.estimate_pressure(tk, tech)
        rows.append(
            {
                "ticker": tk,
                "sector": afp.get("sector"),
                "pressure_type": afp.get("pressure_type"),
                "pressure_score": afp.get("pressure_score"),
            }
        )

    afp_df = pd.DataFrame(rows).sort_values("pressure_score", ascending=False)
    st.dataframe(afp_df, width="stretch", hide_index=True)

if st.sidebar.button("Reiniciar paper demo"):
    PaperTradingDB.reset()
    st.session_state.paper_portfolio = PaperPortfolio()
    st.rerun()
