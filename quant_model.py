import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Configuration & Universe
# ==========================================
TICKER_SECTORS = {
    "NVDA": "Tech", "AAPL": "Tech", "GOOGL": "Tech", "GOOG": "Tech", "MSFT": "Tech", "AMZN": "ConsDisc", "AVGO": "Tech", "META": "Tech", "TSLA": "ConsDisc", "LLY": "Health", "MU": "Tech", "BRK-B": "Financials", "JPM": "Financials", "WMT": "ConsStaples", "AMD": "Tech", "V": "Financials", "XOM": "Energy", "JNJ": "Health", "MA": "Financials", "INTC": "Tech", "CSCO": "Tech", "ABBV": "Health", "BAC": "Financials", "COST": "ConsStaples", "AMAT": "Tech", "CVX": "Energy", "UNH": "Health", "CAT": "Industrials", "KO": "ConsStaples", "LRCX": "Tech", "GE": "Industrials", "ORCL": "Tech", "PG": "ConsStaples", "HD": "ConsDisc", "MS": "Financials", "MRK": "Health", "GS": "Financials", "NFLX": "CommServices", "PM": "ConsStaples", "PLTR": "Tech", "RTX": "Industrials", "PANW": "Tech", "DELL": "Tech", "WFC": "Financials", "TXN": "Tech", "KLAC": "Tech", "AXP": "Financials", "ANET": "Tech", "C": "Financials", "TMO": "Health", "IBM": "Tech", "AMGN": "Health", "APH": "Tech", "VZ": "CommServices", "CRWD": "Tech", "MCD": "ConsDisc", "PEP": "ConsStaples", "WDC": "Tech", "ABT": "Health", "QCOM": "Tech", "DIS": "CommServices", "ADBE": "Tech", "ACN": "Tech", "PFE": "Health", "T": "CommServices", "CMCSA": "CommServices", "NKE": "ConsDisc", "NEE": "Utilities", "LIN": "Materials", "UNP": "Industrials", "COP": "Energy", "MDT": "Health", "TJX": "ConsDisc", "LOW": "ConsDisc", "DE": "Industrials", "INTU": "Tech", "LMT": "Industrials", "BA": "Industrials", "SPGI": "Financials", "BLK": "Financials", "NOW": "Tech", "SNPS": "Tech", "CDNS": "Tech", "EL": "ConsStaples", "CL": "ConsStaples", "MDLZ": "ConsStaples", "MO": "ConsStaples", "SYK": "Health", "BSX": "Health", "CI": "Health", "CVS": "Health", "HCA": "Health", "BDX": "Health", "ETN": "Industrials", "PH": "Industrials", "WM": "Industrials", "NSC": "Industrials", "CSX": "Industrials", "ITW": "Industrials", "EMR": "Industrials"
}

TICKERS = list(TICKER_SECTORS.keys())
TOP_N = 5
MAX_PER_SECTOR = 2

FEATURE_COLS = [
    'Norm_Ret_5d', 'Norm_Ret_10d', 'Ret_1d',
    'Price_to_SMA10', 'Price_to_SMA50', 'ATR_14', 'RSI_14',
    'SPY_Returns_5d', 'VIX_Level', 'VIX_Change_5d', 'Rel_Strength_SPY'
]

def build_feature_panel(period="5y"):
    """Downloads macro data and builds cross-sectional feature panel."""
    spy_raw = yf.download("SPY", period=period, interval="1d", progress=False)
    vix_raw = yf.download("^VIX", period=period, interval="1d", progress=False)

    macro_df = pd.DataFrame()
    macro_df['SPY_Close'] = spy_raw['Close']['SPY'] if isinstance(spy_raw.columns, pd.MultiIndex) else spy_raw['Close']
    macro_df['VIX_Close'] = vix_raw['Close']['^VIX'] if isinstance(vix_raw.columns, pd.MultiIndex) else vix_raw['Close']
    macro_df['SPY_Returns_5d'] = macro_df['SPY_Close'].pct_change(5)
    macro_df['VIX_Change_5d'] = macro_df['VIX_Close'].pct_change(5)
    macro_df['VIX_Level'] = macro_df['VIX_Close']

    panel = []
    for ticker in TICKERS:
        try:
            raw = yf.download(ticker, period=period, interval="1d", progress=False)
            if raw.empty:
                continue
            df = pd.DataFrame()
            df['Close'] = raw['Close'][ticker] if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
            df['High'] = raw['High'][ticker] if isinstance(raw.columns, pd.MultiIndex) else raw['High']
            df['Low'] = raw['Low'][ticker] if isinstance(raw.columns, pd.MultiIndex) else raw['Low']
            
            df['Ticker'] = ticker
            df['Sector'] = TICKER_SECTORS.get(ticker, "Unknown")
            df = df.join(macro_df, how='inner')

            df['Ret_1d'] = df['Close'].pct_change()
            df['Ret_5d'] = df['Close'].pct_change(5)
            df['Ret_10d'] = df['Close'].pct_change(10)

            daily_range = (df['High'] - df['Low']) / df['Close']
            df['ATR_14'] = daily_range.rolling(14).mean()
            df['Norm_Ret_5d'] = df['Ret_5d'] / (df['ATR_14'] + 1e-8)
            df['Norm_Ret_10d'] = df['Ret_10d'] / (df['ATR_14'] + 1e-8)

            df['Price_to_SMA10'] = df['Close'] / df['Close'].rolling(10).mean()
            df['Price_to_SMA50'] = df['Close'] / df['Close'].rolling(50).mean()

            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -1 * delta.clip(upper=0).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI_14'] = 100 - (100 / (1 + rs))
            df['Rel_Strength_SPY'] = df['Ret_5d'] - df['SPY_Returns_5d']

            df['Future_5d_Return'] = df['Close'].pct_change(5).shift(-5)
            df['Target_Class'] = (df['Future_5d_Return'] > 0).astype(int)

            panel.append(df.dropna())
        except Exception:
            continue

    pooled_df = pd.concat(panel, axis=0)
    pooled_df.index = pd.to_datetime(pooled_df.index)
    return pooled_df.sort_index(), macro_df

def generate_target_holdings():
    """Generates real-time long target holdings for live trading."""
    pooled_df, _ = build_feature_panel(period="2y")
    latest_date = pooled_df.index.max()
    
    train_slice = pooled_df[pooled_df.index < latest_date]
    latest_slice = pooled_df[pooled_df.index == latest_date].copy()

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        objective='binary:logistic', random_state=42, eval_metric='logloss'
    )
    model.fit(train_slice[FEATURE_COLS], train_slice['Target_Class'])

    latest_slice['Prob_Up'] = model.predict_proba(latest_slice[FEATURE_COLS])[:, 1]
    sorted_candidates = latest_slice.sort_values(by='Prob_Up', ascending=False)

    selected = []
    sector_counts = {}
    for _, row in sorted_candidates.iterrows():
        sec = row['Sector']
        if sector_counts.get(sec, 0) < MAX_PER_SECTOR:
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        if len(selected) == TOP_N:
            break

    selected_df = pd.DataFrame(selected)

    # Inverse Volatility Sizing (Risk Parity)
    inv_vol = 1.0 / (selected_df['ATR_14'] + 1e-8)
    selected_df['Weight'] = inv_vol / inv_vol.sum()

    return selected_df[['Ticker', 'Sector', 'Prob_Up', 'ATR_14', 'Weight', 'Close']]