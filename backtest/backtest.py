import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import main

''' 
The calculate_metrics function evaluates a trading strategy's performance on a given price dataset. 
It calculates percentage price changes, applies a model (from main.models) to generate trading signals 
based on a specified window, and determines long (+1) or short (-1) positions using thresholds (thres1, thres2). 
The function computes profits/losses (pnl), accounting for trading costs (0.06% per trades), 
and tracks cumulative profits (cumu_pnl) and drawdowns (dd). It returns a DataFrame with these metrics and a 
dictionary containing key performance indicators: Sharpe Ratio(SR), Maximum Drawdown(MDD), Annualized Return (AR), 
Calmar Ratio (AR/MDD), total trades, and Trades Per Instance (TPI).
'''
def calculate_metrics(df, window, thres1, thres2):
    df = pd.DataFrame(df.copy())
    df['close_chg'] = df['close'].pct_change()  
    df['model'] = main.models(df, window)
    df['pos'] = np.where(df['model'] > thres1, 1, np.where(df['model'] < thres2, -1, np.nan))
    df['pos'] = df['pos'].ffill()
    df['pos_t-1'] = df['pos'].shift(1).fillna(0)
    df['trade'] = abs(df['pos'].shift(1) - df['pos']) 
    df['pnl'] = df['pos_t-1'] * df['close_chg'] - df['trade'] * 0.0006
    df['cumu_pnl'] = df['pnl'].cumsum()
    df['dd'] = df['cumu_pnl'] - df['cumu_pnl'].cummax()

    sr = round(df['pnl'].mean() / df['pnl'].std() * np.sqrt(365 * 24), 4) if df['pnl'].std() != 0 else np.nan
    mdd = round(df['dd'].min(), 4) if not df['dd'].empty else np.nan
    ar = round(df['pnl'].mean() * (365 * 24), 4) if not df['pnl'].empty else np.nan
    cr = round(abs(ar / mdd), 4) if mdd != 0 else np.nan
    cumu_returns = round(df['cumu_pnl'].iloc[-1], 4) if not df['cumu_pnl'].empty else np.nan
    trades = int(df['trade'].sum())
    tpi = round(trades / (len(df) - window) * 100, 4)

    metrics = {
        'window': window,
        'thres1': thres1,
        'thres2': thres2,
        'Sharpe Ratio': sr,
        'Cumu Returns': cumu_returns,
        'MDD': mdd,
        'AR': ar,
        'CR': cr,
        'TRADES': trades,
        'TPI': tpi
    }

    return df, metrics

'''
The plot_equity_curve function help to show equity curve in a line chart from backtest result.
The backtest will include in sample and out sample price. (train-test split/forward test) 
When forward test == 'yes' and test_df (out sample) is not None, equity curve will be plotted.
'''
def plot_equity_curve(train_df, test_df=None, forward_test='no'):
    plt.figure(figsize=(10, 6))
    if 'cumu_pnl' not in train_df.columns:
        raise ValueError("No 'cumu_pnl' column found in the train DataFrame.")
    
    plt.plot(train_df.index, train_df['cumu_pnl'], label='Train Equity Curve', color='blue')
    
    if forward_test == 'yes' and test_df is not None:
        if 'cumu_pnl' not in test_df.columns:
            raise ValueError("No 'cumu_pnl' column found in the test DataFrame.")
        plt.plot(test_df.index, test_df['cumu_pnl'], label='Test Equity Curve', color='orange')
    
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.title('Equity Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png')
    plt.show()
