import os
import streamlit as st
import pandas as pd
import plotly.express as px
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Page Configuration
st.set_page_config(
    page_title="Quant Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize Credentials securely from Streamlit Secrets or Environment Variables
api_key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
secret_key = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))

# Title Header
st.title("📈 Quantitative Paper Trading Dashboard")

if not api_key or not secret_key:
    st.error(
        "Alpaca API credentials not found. Please add `ALPACA_API_KEY` and "
        "`ALPACA_SECRET_KEY` to your Streamlit secrets or environment variables."
    )
    st.stop()

@st.cache_resource
def get_client(key, secret):
    return TradingClient(key, secret, paper=True)

try:
    client = get_client(api_key, secret_key)
    account = client.get_account()
except Exception as e:
    st.error(f"Failed to connect to Alpaca API: {e}")
    st.stop()

# ==========================================
# 1. Metrics Overview Bar
# ==========================================
equity = float(account.equity)
cash = float(account.cash)
buying_power = float(account.buying_power)
last_equity = float(account.last_equity)
equity_change = equity - last_equity
equity_pct_change = (equity_change / last_equity) * 100 if last_equity else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", f"${equity:,.2f}", f"{equity_pct_change:+.2f}%")
col2.metric("Cash Balance", f"${cash:,.2f}")
col3.metric("Buying Power", f"${buying_power:,.2f}")
col4.metric("Account Status", account.status.value.upper())

st.markdown("---")

# ==========================================
# 2. Current Holdings & Portfolio Allocation
# ==========================================
st.subheader("Current Holdings & Portfolio Allocation")

try:
    positions = client.get_all_positions()
except Exception as e:
    st.error(f"Error fetching positions: {e}")
    positions = []

if positions:
    pos_data = []
    for p in positions:
        pos_data.append({
            "Symbol": p.symbol,
            "Market Value ($)": float(p.market_value),
            "Quantity": float(p.qty),
            "Avg Entry Price": float(p.avg_entry_price),
            "Current Price": float(p.current_price),
            "Unrealized PnL ($)": float(p.unrealized_pl),
            "Unrealized PnL (%)": float(p.unrealized_plpc) * 100
        })
    
    pos_df = pd.DataFrame(pos_data)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.dataframe(
            pos_df.style.format({
                "Market Value ($)": "${:,.2f}",
                "Avg Entry Price": "${:,.2f}",
                "Current Price": "${:,.2f}",
                "Unrealized PnL ($)": "${:+,.2f}",
                "Unrealized PnL (%)": "{:+.2f}%"
            }),
            use_container_width=True
        )

    with c2:
        fig = px.pie(
            pos_df, 
            values="Market Value ($)", 
            names="Symbol", 
            title="Portfolio Composition",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No active positions currently held in account.")

st.markdown("---")

# ==========================================
# 3. Order History & Execution Logs
# ==========================================
st.subheader("Recent Execution Logs & Order History")

try:
    request_params = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=20)
    orders = client.get_orders(filter=request_params)
except Exception as e:
    st.error(f"Error fetching orders: {e}")
    orders = []

if orders:
    order_data = []
    for o in orders:
        order_data.append({
            "Submitted At": o.submitted_at.strftime("%Y-%m-%d %H:%M:%S") if o.submitted_at else "N/A",
            "Symbol": o.symbol,
            "Side": o.side.value.upper(),
            "Notional ($)": f"${float(o.notional):,.2f}" if o.notional else "N/A",
            "Status": o.status.value.upper(),
            "Order ID": o.id
        })
    order_df = pd.DataFrame(order_data)
    st.dataframe(order_df, use_container_width=True)
else:
    st.info("No recent orders found.")