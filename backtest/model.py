import pandas as pd
import numpy as np


def zscore(df, window):
    df['ma'] = df['close_chg'].rolling(window=window, min_periods=window).mean()
    df['std'] = df['close_chg'].rolling(window=window, min_periods=window).std()
    df['zscore'] = (df['close_chg'] - df['ma']) / df['std']
    df['model'] = df['zscore']
    return df['model']

def minmax(df, window):
    df['max'] = df['close_chg'].rolling(window=window, min_periods=window).max()
    df['min'] = df['close_chg'].rolling(window=window, min_periods=window).min()
    df['minmax'] = 2* (df['close_chg'] - df['min']) / (df['max'] - df['min']) - 1
    df['model'] = df['minmax']
    return df['model']

def robust_scaling(df, window):
    df['median'] = df['close_chg'].rolling(window=window, min_periods=window).median()
    df['q1'] = df['close_chg'].rolling(window=window, min_periods=window).quantile(0.25)
    df['q3'] = df['close_chg'].rolling(window=window, min_periods=window).quantile(0.75)
    df['iqr'] = df['q3'] - df['q1']
    df['robust'] = (df['close_chg'] - df['median']) / df['iqr'].replace(0, np.nan)
    df['model'] = df['robust']
    return df['model']

def bollinger_width(df, window):
    df['ma'] = df['close_chg'].rolling(window=window, min_periods=window).mean()
    df['std'] = df['close_chg'].rolling(window=window, min_periods=window).std()
    df['upper_band'] = df['ma'] + 2 * df['std']
    df['lower_band'] = df['ma'] - 2 * df['std']
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['ma']
    df['model'] = df['bb_width']
    return df['model']

def rsi_scaled(df, window):
    delta = df['close_chg'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['model'] = df['rsi'] / 50 - 1  # Scale to [-1, 1]
    return df['model']

def log_return(df, window):
    df['log_return'] = np.log(df['close_chg']).rolling(window=window, min_periods=window).mean()
    df['model'] = df['log_return']
    return df['model']

def volatility_adjusted(df, window):
    df['vol'] = df['close_chg'].rolling(window=window, min_periods=window).std()
    df['model'] = df['close_chg'] / df['vol'].replace(0, np.nan)
    return df['model']
