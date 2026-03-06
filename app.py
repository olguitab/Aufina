import threading

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from market_data import MarketData
from paper_trading import INITIAL_BALANCE_CLP, PaperPortfolio, PaperTradingDB
from trading_bot import AutonomousBot

load_dotenv()

st.set_page_config(page_title="Aureus Demo", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
            display: none;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #1f3b53;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            color: #f8fafc;
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0;
        }
        .hero-subtitle {
            color: #cbd5e1;
            margin-top: .35rem;
            margin-bottom: 0;
        }
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #1f5132;
            color: #dcfce7;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: .6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "paper_portfolio" not in st.session_state:
    st.session_state.paper_portfolio = PaperPortfolio()

if "bot_thread_started" not in st.session_state:
    st.session_state.bot_thread_started = False


def run_bot_in_background():
    AutonomousBot().start()


if not st.session_state.bot_thread_started:
    thread = threading.Thread(target=run_bot_in_background, daemon=True)
    thread.start()
    st.session_state.bot_thread_started = True


refresh_count = st_autorefresh(interval=15 * 1000, key="demo_refresh")
if refresh_count >= 0:
    st.session_state.paper_portfolio = PaperPortfolio()


paper = st.session_state.paper_portfolio
paper_trades = PaperTradingDB.load_trade_log()
initial = INITIAL_BALANCE_CLP
market_data = MarketData()
market_open = market_data.is_santiago_market_open()

if market_open:
    with st.spinner("Actualizando portafolio demo..."):
        total_value = paper.get_total_value(market_data)
else:
    total_value = cash_value = paper.balance
    for ticker, qty in paper.positions.items():
        if qty > 0:
            total_value += qty * paper.position_costs.get(ticker, 0)

cash_value = paper.balance
trade_count = len(paper_trades)
delta_value = total_value - initial
delta_pct = (delta_value / initial) * 100 if initial else 0

st.markdown(
    """
    <div class="hero-card">
        <span class="status-pill">MODO DEMO</span>
        <h2 class="hero-title">Aureus Wealth — Demo de Inversión</h2>
        <p class="hero-subtitle">Dinero ficticio con precios reales de mercado. Vista diseñada para presentar resultados de inversión de forma clara.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
m1.metric("Cash Libre", f"${cash_value:,.0f}")
m2.metric(
    "Valor Total (Cash + Acciones)",
    f"${total_value:,.0f}",
    delta=f"${delta_value:,.0f} ({delta_pct:.2f}%)",
    delta_color="normal",
)
m3.metric("Total Operaciones", trade_count)

st.markdown("---")

active_positions = {ticker: qty for ticker, qty in paper.positions.items() if qty > 0}
portfolio_distribution = {"Cash": cash_value}

for ticker, qty in active_positions.items():
    if market_open:
        try:
            ticker_data = market_data.get_comprehensive_data(ticker)
            current_price = ticker_data.get("current_price", 0)
            if current_price > 0:
                portfolio_distribution[ticker] = qty * current_price
                continue
        except Exception:
            pass
    portfolio_distribution[ticker] = qty * paper.position_costs.get(ticker, 0)

chart_col, positions_col = st.columns([2, 1])
with chart_col:
    st.subheader("Distribución del Portafolio")
    if portfolio_distribution and sum(portfolio_distribution.values()) > 0:
        dist_df = pd.DataFrame(list(portfolio_distribution.items()), columns=["Activo", "Valor"])
        pie_chart = px.pie(
            dist_df,
            values="Valor",
            names="Activo",
            hole=0.55,
            color_discrete_sequence=px.colors.sequential.Emrld,
        )
        pie_chart.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(pie_chart, width="stretch")
    else:
        st.info("Aún no hay datos suficientes para graficar la distribución.")

with positions_col:
    st.subheader("Posiciones Abiertas")
    if active_positions:
        for ticker, qty in active_positions.items():
            st.write(f"**{ticker}** · {qty:.0f} acciones")
    else:
        st.caption("Sin posiciones abiertas por ahora.")

st.markdown("---")
st.subheader("Historial Completo de Trading")

if paper_trades:
    trades_df = pd.DataFrame(paper_trades)

    def style_signal(signal):
        if signal == "BUY":
            return "background-color:#134e4a;color:#a7f3d0;font-weight:600"
        if signal == "SELL":
            return "background-color:#7f1d1d;color:#fecaca;font-weight:600"
        return ""

    st.dataframe(
        trades_df.style.map(style_signal, subset=["Signal"]),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("Todavía no hay operaciones registradas. El bot demo generará trades automáticamente.")

left_btn, right_info = st.columns([1, 2])
with left_btn:
    if st.button("Reiniciar Demo", type="secondary"):
        PaperTradingDB.reset()
        st.session_state.paper_portfolio = PaperPortfolio()
        st.rerun()

with right_info:
    if market_open:
        st.caption("Actualización automática cada 15 segundos.")
    else:
        st.caption("Bolsa de Santiago cerrada: se muestran valores de referencia sin consultar mercado en vivo.")
