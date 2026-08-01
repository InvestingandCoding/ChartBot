import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from quant_model import build_feature_panel, FEATURE_COLS, TOP_N, MAX_PER_SECTOR

warnings.filterwarnings('ignore')

TOTAL_FRICTION = 0.0006  # 0.06% slippage + fees
TRAIN_RATIO = 0.70
HORIZON_DAYS = 5

def run_walkforward_backtest():
    """Runs walk-forward backtest and returns computed equity curves and metrics."""
    pooled_df, macro_df = build_feature_panel(period="5y")
    unique_dates = np.unique(pooled_df.index)
    weekly_dates = unique_dates[::HORIZON_DAYS]

    split_idx = int(len(weekly_dates) * TRAIN_RATIO)
    test_dates = weekly_dates[split_idx:]

    strat_equity = [10000.0]
    spy_equity = [10000.0]
    dates_list = [test_dates[0]]
    weekly_logs = []

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        objective='binary:logistic', random_state=42, eval_metric='logloss'
    )

    for i in range(len(test_dates) - 1):
        current_date = test_dates[i]
        train_slice = pooled_df[pooled_df.index < current_date]
        test_slice = pooled_df[pooled_df.index == current_date]

        if len(test_slice) < TOP_N:
            continue

        model.fit(train_slice[FEATURE_COLS], train_slice['Target_Class'])
        test_slice = test_slice.copy()
        test_slice['Prob_Up'] = model.predict_proba(test_slice[FEATURE_COLS])[:, 1]

        sorted_candidates = test_slice.sort_values(by='Prob_Up', ascending=False)
        selected_picks = []
        sector_counts = {}

        for _, row in sorted_candidates.iterrows():
            sec = row['Sector']
            if sector_counts.get(sec, 0) < MAX_PER_SECTOR:
                selected_picks.append(row)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected_picks) == TOP_N:
                break

        top_df = pd.DataFrame(selected_picks)
        inv_vol = 1.0 / (top_df['ATR_14'] + 1e-8)
        weights = inv_vol / inv_vol.sum()
        
        returns_after_friction = top_df['Future_5d_Return'] - (2 * TOTAL_FRICTION)
        strat_ret = (returns_after_friction * weights).sum()

        next_date = test_dates[i+1]
        spy_start = macro_df.loc[macro_df.index <= current_date, 'SPY_Close'].iloc[-1]
        spy_end = macro_df.loc[macro_df.index <= next_date, 'SPY_Close'].iloc[-1]
        spy_ret = (spy_end - spy_start) / spy_start

        strat_equity.append(strat_equity[-1] * (1 + strat_ret))
        spy_equity.append(spy_equity[-1] * (1 + spy_ret))
        dates_list.append(next_date)

        weekly_logs.append({
            'Date': current_date,
            'Top_Picks': ", ".join(top_df['Ticker'].tolist()),
            'Strategy_Return_%': strat_ret * 100,
            'SP500_Return_%': spy_ret * 100
        })

    # Dynamically compute metrics from actual backtest output
    results_df = pd.DataFrame(weekly_logs)
    strat_returns = pd.Series(strat_equity).pct_change().dropna()
    spy_returns = pd.Series(spy_equity).pct_change().dropna()

    def calc_max_drawdown(equity_series):
        eq = pd.Series(equity_series)
        peak = eq.cummax()
        dd = (eq - peak) / peak
        return dd.min() * 100

    metrics = {
        'initial_capital': 10000.0,
        'strategy_final_equity': strat_equity[-1],
        'strategy_total_return_pct': ((strat_equity[-1] / 10000.0) - 1.0) * 100,
        'spy_final_equity': spy_equity[-1],
        'spy_total_return_pct': ((spy_equity[-1] / 10000.0) - 1.0) * 100,
        'strategy_sharpe': (strat_returns.mean() / (strat_returns.std() + 1e-8)) * np.sqrt(52),
        'spy_sharpe': (spy_returns.mean() / (spy_returns.std() + 1e-8)) * np.sqrt(52),
        'strategy_max_drawdown': calc_max_drawdown(strat_equity),
        'spy_max_drawdown': calc_max_drawdown(spy_equity),
        'weekly_win_rate': (strat_returns > 0).mean() * 100,
        'total_test_weeks': len(results_df)
    }

    equity_df = pd.DataFrame({
        'Date': dates_list,
        'Strategy_Equity': strat_equity,
        'SPY_Equity': spy_equity
    })

    return metrics, equity_df, results_df

if __name__ == "__main__":
    print("Running Walk-Forward Backtest...")
    metrics, _, results_df = run_walkforward_backtest()
    print("\n=======================================================")
    print("            DYNAMICALLY COMPUTED BACKTEST RESULTS       ")
    print("=======================================================")
    print(f"Strategy Final Equity : ${metrics['strategy_final_equity']:,.2f} ({metrics['strategy_total_return_pct']:+.2f}%)")
    print(f"SPY Final Equity      : ${metrics['spy_final_equity']:,.2f} ({metrics['spy_total_return_pct']:+.2f}%)")
    print(f"Strategy Sharpe Ratio : {metrics['strategy_sharpe']:.2f}")
    print(f"SPY Sharpe Ratio      : {metrics['spy_sharpe']:.2f}")
    print(f"Strategy Max Drawdown : {metrics['strategy_max_drawdown']:.2f}%")
    print(f"SPY Max Drawdown      : {metrics['spy_max_drawdown']:.2f}%")
    print(f"Weekly Win Rate       : {metrics['weekly_win_rate']:.1f}% across {metrics['total_test_weeks']} weeks")
    print("=======================================================")