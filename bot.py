import os
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from quant_model import generate_target_holdings

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

def execute_rebalance():
    if not API_KEY or not SECRET_KEY:
        print("ERROR: Missing ALPACA_API_KEY or ALPACA_SECRET_KEY environment variables.")
        sys.exit(1)

    print("Connecting to Alpaca Paper Trading...")
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    
    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    print(f"Current Portfolio Value: ${portfolio_value:,.2f}")

    print("Generating Live Target Allocation Signals...")
    targets_df = generate_target_holdings()
    print("\n--- Dynamically Sized Target Portfolio ---")
    print(targets_df.to_string(index=False))

    target_tickers = set(targets_df['Ticker'])
    
    # Sell non-target holdings
    current_positions = client.get_all_positions()
    for pos in current_positions:
        symbol = pos.symbol
        if symbol not in target_tickers:
            print(f"Closing position in {symbol}...")
            client.close_position(symbol)

    # Rebalance into Top 5 targets using dynamic Inverse Volatility weights
    for _, row in targets_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        price = row['Close']
        
        target_allocation_dollars = portfolio_value * weight
        target_qty = int(target_allocation_dollars // price)

        if target_qty <= 0:
            continue

        existing_qty = 0
        try:
            existing_pos = client.get_open_position(ticker)
            existing_qty = int(existing_pos.qty)
        except Exception:
            existing_qty = 0

        qty_diff = target_qty - existing_qty

        if qty_diff > 0:
            print(f"Submitting BUY order: {qty_diff} shares of {ticker}")
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty_diff,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order_request=req)
        elif qty_diff < 0:
            print(f"Submitting SELL order: {abs(qty_diff)} shares of {ticker}")
            req = MarketOrderRequest(
                symbol=ticker,
                qty=abs(qty_diff),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order_request=req)
        else:
            print(f"Position {ticker} is already balanced.")

    print("\nRebalance script finished successfully!")

if __name__ == "__main__":
    execute_rebalance()