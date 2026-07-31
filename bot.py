import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ==========================================
# 1. Configuration & Credentials
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY environment variable not set.")
    sys.exit(1)

# Initialize Alpaca Paper Trading Client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

TICKERS = [
    "NVDA", "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "AVGO", "META", "TSLA", "LLY",
    "MU", "BRK-B", "JPM", "WMT", "AMD", "V", "XOM", "JNJ", "MA", "INTC",
    "CSCO", "ABBV", "BAC", "COST", "AMAT", "CVX", "UNH", "CAT", "KO", "LRCX",
    "GE", "ORCL", "PG", "HD", "MS", "MRK", "GS", "NFLX", "PM", "PLTR",
    "RTX", "PANW", "DELL", "WFC", "TXN", "KLAC", "AXP", "ANET", "C",
    "TMO", "IBM", "AMGN", "APH", "VZ", "CRWD", "MCD", "PEP", "WDC", "ABT",
    "QCOM", "DIS", "ADBE", "ACN", "PFE", "T", "CMCSA", "NKE", "NEE", "LIN"
]

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"
PERIOD = "5y"
TOP_N = 10

# ==========================================
# 2. Data Processing & Model Inference
# ==========================================
def get_top_signals():
    print("Downloading market and macro data...")
    m_raw = yf.download(MARKET_TICKER, period=PERIOD, interval="1d", progress=False)
    v_raw = yf.download(VIX_TICKER, period=PERIOD, interval="1d", progress=False)

    macro_df = pd.DataFrame()
    macro_df['Market_Close'] = m_raw['Close'][MARKET_TICKER] if isinstance(m_raw.columns, pd.MultiIndex) else m_raw['Close']
    macro_df['VIX_Close'] = v_raw['Close'][VIX_TICKER] if isinstance(v_raw.columns, pd.MultiIndex) else v_raw['Close']
    
    macro_df['Market_Returns_5d'] = macro_df['Market_Close'].pct_change(periods=5)
    macro_df['VIX_Change_5d'] = macro_df['VIX_Close'].pct_change(periods=5)
    macro_df['VIX_Level'] = macro_df['VIX_Close']

    panel_list = []
    print("Processing universe feature panel...")

    for ticker in TICKERS:
        try:
            raw = yf.download(ticker, period=PERIOD, interval="1d", progress=False)
            if raw.empty:
                continue

            df = pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                df['Close'] = raw['Close'][ticker]
                df['High'] = raw['High'][ticker]
                df['Low'] = raw['Low'][ticker]
            else:
                df['Close'] = raw['Close']
                df['High'] = raw['High']
                df['Low'] = raw['Low']

            df['Ticker'] = ticker
            df = df.join(macro_df, how='inner')

            df['Ret_1d'] = df['Close'].pct_change()
            df['Ret_5d'] = df['Close'].pct_change(periods=5)
            df['Ret_10d'] = df['Close'].pct_change(periods=10)

            daily_range = (df['High'] - df['Low']) / df['Close']
            df['ATR_14'] = daily_range.rolling(window=14).mean()
            df['Norm_Ret_5d'] = df['Ret_5d'] / (df['ATR_14'] + 1e-8)
            df['Norm_Ret_10d'] = df['Ret_10d'] / (df['ATR_14'] + 1e-8)

            df['Price_to_SMA10'] = df['Close'] / df['Close'].rolling(window=10).mean()
            df['Price_to_SMA50'] = df['Close'] / df['Close'].rolling(window=50).mean()

            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -1 * delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI_14'] = 100 - (100 / (1 + rs))

            df['Rel_Strength_Market'] = df['Ret_5d'] - df['Market_Returns_5d']
            df['Future_5d_Return'] = df['Close'].pct_change(periods=5).shift(-5)
            df['Target_Class'] = (df['Future_5d_Return'] > 0).astype(int)

            panel_list.append(df.dropna())
        except Exception:
            continue

    pooled_df = pd.concat(panel_list, axis=0)
    
    feature_cols = [
        'Norm_Ret_5d', 'Norm_Ret_10d', 'Ret_1d',
        'Price_to_SMA10', 'Price_to_SMA50', 'ATR_14', 'RSI_14',
        'Market_Returns_5d', 'VIX_Level', 'VIX_Change_5d', 'Rel_Strength_Market'
    ]

    print("Training pooled XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        objective='binary:logistic', random_state=42, eval_metric='logloss'
    )
    model.fit(pooled_df[feature_cols], pooled_df['Target_Class'])

    # Get latest date snapshot for each ticker
    latest_snapshot = pooled_df.groupby('Ticker').last().reset_index()
    latest_snapshot['Prob_Up'] = model.predict_proba(latest_snapshot[feature_cols])[:, 1]
    
    # Rank Top N picks
    top_picks = latest_snapshot.sort_values(by='Prob_Up', ascending=False).head(TOP_N)
    return top_picks['Ticker'].tolist()

# ==========================================
# 3. Order Execution via Alpaca API
# ==========================================
def rebalance_portfolio():
    top_tickers = get_top_signals()
    print(f"\nTop {TOP_N} Selected Tickers for this week: {top_tickers}")

    # Step 1: Liquidate existing positions
    print("Liquidating existing positions...")
    try:
        trading_client.close_all_positions(cancel_orders=True)
        print("All existing positions liquidated.")
    except Exception as e:
        print(f"Position liquidation notice: {e}")

    # Step 2: Calculate purchasing budget
    account = trading_client.get_account()
    buying_power = float(account.buying_power)
    equity = float(account.equity)
    
    # Use 95% of liquid equity evenly across TOP_N
    allocation_per_stock = (equity * 0.95) / TOP_N
    print(f"Account Equity: ${equity:,.2f} | Capital per stock: ${allocation_per_stock:,.2f}")

    # Step 3: Place new fractional/notional market buy orders
    for ticker in top_tickers:
        formatted_ticker = ticker.replace('-', '.') # Handle symbols like BRK.B
        try:
            order_data = MarketOrderRequest(
                symbol=formatted_ticker,
                notional=round(allocation_per_stock, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = trading_client.submit_order(order_data=order_data)
            print(f"Successfully ordered ${allocation_per_stock:.2f} of {formatted_ticker} (Order ID: {order.id})")
        except Exception as e:
            print(f"Failed to place order for {formatted_ticker}: {e}")

if __name__ == "__main__":
    rebalance_portfolio()