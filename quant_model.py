import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Ticker Universe
# ==========================================
TICKERS = [
    "NVDA", "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "AVGO", "META", "TSLA", "LLY",
    "MU", "BRK-B", "JPM", "WMT", "AMD", "V", "XOM", "JNJ", "MA", "INTC",
    "CSCO", "ABBV", "BAC", "COST", "AMAT", "CVX", "UNH", "CAT", "KO", "LRCX",
    "GE", "ORCL", "PG", "HD", "MS", "MRK", "GS", "NFLX", "PM", "PLTR",
    "RTX", "PANW", "DELL", "WFC", "TXN", "KLAC", "AXP", "ANET", "C",
    "TMO", "IBM", "AMGN", "APH", "VZ", "CRWD", "MCD", "PEP", "WDC", "ABT",
    "QCOM", "DIS", "ADBE", "ACN", "PFE", "T", "CMCSA", "NKE", "NEE", "LIN",
    "UNP", "COP", "MDT", "TJX", "LOW", "DE", "INTU", "LMT", "BA", "SPGI",
    "BLK", "NOW", "SNPS", "CDNS", "EL", "CL", "MDLZ", "MO", "SYK", "BSX",
    "CI", "CVS", "HCA", "BDX", "ETN", "PH", "WM", "NSC", "CSX", "ITW", "EMR"
]

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"
PERIOD = "5y"
TRAIN_RATIO = 0.80
HORIZON_DAYS = 5       # 1-week non-overlapping steps
TOP_N = 10             # Number of top stocks to go long each week

# Transaction Friction
SLIPPAGE_PCT = 0.0005  # 0.05%
FEE_PCT = 0.0001       # 0.01%
TOTAL_FRICTION = SLIPPAGE_PCT + FEE_PCT

# ==========================================
# 2. Data Fetching & Pooled Feature Engineering
# ==========================================
def fetch_and_build_pooled_panel():
    print(f"Downloading Macro Data ({MARKET_TICKER} & {VIX_TICKER})...")
    m_raw = yf.download(MARKET_TICKER, period=PERIOD, interval="1d", progress=False)
    v_raw = yf.download(VIX_TICKER, period=PERIOD, interval="1d", progress=False)
    
    macro_df = pd.DataFrame()
    macro_df['Market_Close'] = m_raw['Close'][MARKET_TICKER] if isinstance(m_raw.columns, pd.MultiIndex) else m_raw['Close']
    macro_df['VIX_Close'] = v_raw['Close'][VIX_TICKER] if isinstance(v_raw.columns, pd.MultiIndex) else v_raw['Close']
    
    macro_df['Market_Returns_5d'] = macro_df['Market_Close'].pct_change(periods=5)
    macro_df['VIX_Change_5d'] = macro_df['VIX_Close'].pct_change(periods=5)
    macro_df['VIX_Level'] = macro_df['VIX_Close']

    panel_list = []
    print(f"Downloading and processing {len(TICKERS)} tickers...")

    for idx, ticker in enumerate(TICKERS):
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

            # 1. Raw Returns
            df['Ret_1d'] = df['Close'].pct_change()
            df['Ret_5d'] = df['Close'].pct_change(periods=5)
            df['Ret_10d'] = df['Close'].pct_change(periods=10)

            # 2. Volatility Normalization (ATR)
            daily_range = (df['High'] - df['Low']) / df['Close']
            df['ATR_14'] = daily_range.rolling(window=14).mean()
            
            # Volatility-Normalized Returns (Z-Score approximation per asset)
            df['Norm_Ret_5d'] = df['Ret_5d'] / (df['ATR_14'] + 1e-8)
            df['Norm_Ret_10d'] = df['Ret_10d'] / (df['ATR_14'] + 1e-8)

            # 3. Moving Averages & RSI
            df['Price_to_SMA10'] = df['Close'] / df['Close'].rolling(window=10).mean()
            df['Price_to_SMA50'] = df['Close'] / df['Close'].rolling(window=50).mean()

            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -1 * delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI_14'] = 100 - (100 / (1 + rs))

            df['Rel_Strength_Market'] = df['Ret_5d'] - df['Market_Returns_5d']

            # Target: Forward 5-day return
            df['Future_5d_Return'] = df['Close'].pct_change(periods=HORIZON_DAYS).shift(-HORIZON_DAYS)
            df['Target_Class'] = (df['Future_5d_Return'] > 0).astype(int)

            panel_list.append(df.dropna())

        except Exception as e:
            continue

    pooled_df = pd.concat(panel_list, axis=0)
    pooled_df.index = pd.to_datetime(pooled_df.index)
    return pooled_df.sort_index()

# ==========================================
# 3. Pipeline & Cross-Sectional Strategy
# ==========================================
def main():
    pooled_df = fetch_and_build_pooled_panel()
    
    feature_cols = [
        'Norm_Ret_5d', 'Norm_Ret_10d', 'Ret_1d',
        'Price_to_SMA10', 'Price_to_SMA50', 'ATR_14', 'RSI_14',
        'Market_Returns_5d', 'VIX_Level', 'VIX_Change_5d', 'Rel_Strength_Market'
    ]

    # Non-overlapping weekly timestamps
    unique_dates = np.unique(pooled_df.index)
    weekly_dates = unique_dates[::HORIZON_DAYS]

    split_idx = int(len(weekly_dates) * TRAIN_RATIO)
    train_dates = weekly_dates[:split_idx]
    test_dates = weekly_dates[split_idx:]

    train_data = pooled_df[pooled_df.index.isin(train_dates)]
    test_data = pooled_df[pooled_df.index.isin(test_dates)]

    print(f"\n=======================================================")
    print(f"Total Unique Tickers  : {pooled_df['Ticker'].nunique()}")
    print(f"Training Data Samples : {len(train_data)} rows across {len(train_dates)} weeks")
    print(f"Testing Data Samples  : {len(test_data)} rows across {len(test_dates)} weeks")
    print("=======================================================\n")

    X_train, y_train = train_data[feature_cols], train_data['Target_Class']
    X_test = test_data[feature_cols]

    # Universal Model across all 101 stocks
    print("Training Universal Pooled XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Attach predicted probabilities to test data
    test_data = test_data.copy()
    test_data['Prob_Up'] = model.predict_proba(X_test)[:, 1]

    # ==========================================
    # 4. Weekly Cross-Sectional Backtest Engine
    # ==========================================
    capital = 10000.0
    benchmark_capital = 10000.0
    
    portfolio_history = []
    benchmark_history = []

    for date in test_dates:
        weekly_slice = test_data[test_data.index == date]
        if len(weekly_slice) < TOP_N:
            continue

        # Cross-Sectional Ranking: Pick Top N assets with highest probability of going UP
        top_picks = weekly_slice.sort_values(by='Prob_Up', ascending=False).head(TOP_N)
        
        # Strategy Return: Equal-weighted allocation across Top N picks minus transaction friction
        top_returns = top_picks['Future_5d_Return'] - (2 * TOTAL_FRICTION)
        avg_strat_return = top_returns.mean()
        capital *= (1 + avg_strat_return)
        portfolio_history.append(capital)

        # Benchmark Return: Equal-weighted allocation across ALL universe stocks on that date
        avg_market_return = weekly_slice['Future_5d_Return'].mean()
        benchmark_capital *= (1 + avg_market_return)
        benchmark_history.append(benchmark_capital)

    # Performance Metrics
    total_strat_ret = ((capital - 10000.0) / 10000.0) * 100
    total_bm_ret = ((benchmark_capital - 10000.0) / 10000.0) * 100

    print("\n=======================================================")
    print("       POOLED CROSS-SECTIONAL RANKING BACKTEST         ")
    print("=======================================================\n")
    print(f"Initial Starting Capital   : $10,000.00")
    print(f"Strategy Selection Rule    : Equal-Weight Top {TOP_N} Stocks Each Week")
    print(f"Execution Friction         : {TOTAL_FRICTION*100:.2f}% per trade\n")
    print(f"Strategy Final Portfolio   : ${capital:.2f} ({total_strat_ret:+.2f}%)")
    print(f"Equal-Weight Universe B&H  : ${benchmark_capital:.2f} ({total_bm_ret:+.2f}%)\n")

    if total_strat_ret > total_bm_ret:
        print("Result: Universal Pooled Strategy OUTPERFORMED the Universe Benchmark!")
    else:
        print("Result: Benchmark Outperformed.")

    print("\nTop 5 Universal Predictive Features:")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.head(5).items():
        print(f"  {feat:<25}: {imp:.4f}")

if __name__ == "__main__":
    main()