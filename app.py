import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from quant_model import generate_target_holdings
from backtest import run_walkforward_backtest

st.set_page_config(page_title="Quant Alpha Dashboard", layout="wide")

st.title("📈 Dynamic Quant Alpha Dashboard")
st.caption("All statistics are computed directly from live model and backtest executions.")

st.sidebar.header("Execution Controls")

# Live Rebalance Holdings Section
st.subheader("1. Live Target Holdings (Real-time Model Signal)")
if st.button("Generate Live Signals"):
    with st.spinner("Computing dynamic targets..."):
        targets = generate_target_holdings()
        
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            targets[['Ticker', 'Sector', 'Prob_Up', 'Weight', 'Close']].style.format({
                'Prob_Up': '{:.2%}',
                'Weight': '{:.2%}',
                'Close': '${:.2f}'
            }),
            use_container_width=True
        )
    with col2:
        fig_pie = px.pie(targets, names='Sector', values='Weight', title="Dynamic Sector Weights")
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# Dynamic Backtest Section
st.subheader("2. Walk-Forward Backtest Performance")
if st.button("Run Full Dynamic Backtest"):
    with st.spinner("Executing walk-forward backtest on historical data..."):
        metrics, equity_df, results_df = run_walkforward_backtest()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy Return", f"{metrics['strategy_total_return_pct']:+.2f}%", f"{metrics['strategy_total_return_pct'] - metrics['spy_total_return_pct']:+.2f}% vs SPY")
    m2.metric("Sharpe Ratio", f"{metrics['strategy_sharpe']:.2f}", f"{metrics['spy_sharpe']:.2f} SPY")
    m3.metric("Max Drawdown", f"{metrics['strategy_max_drawdown']:.2f}%", f"{metrics['spy_max_drawdown']:.2f}% SPY")
    m4.metric("Weekly Win Rate", f"{metrics['weekly_win_rate']:.1f}%", f"{metrics['total_test_weeks']} Test Weeks")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['Strategy_Equity'], name="Strategy Equity", line=dict(color='#00CC96', width=2.5)))
    fig.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['SPY_Equity'], name="S&P 500 (SPY)", line=dict(color='#636EFA', width=2, dash='dash')))
    fig.update_layout(title="Dynamically Calculated Walk-Forward Equity Curve", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Execution Log Snapshot")
    st.dataframe(results_df.tail(10), use_container_width=True)