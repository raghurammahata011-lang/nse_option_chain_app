# utils/technical_indicators.py
import pandas as pd
import numpy as np

def calculate_technical_indicators(df, window=14):
    """
    Calculate technical indicators for option chain analysis
    """
    df_tech = df.copy()
    
    # RSI-like indicator for OI momentum
    df_tech['CALL_OI_GAIN'] = np.where(df_tech['CALL_CHNG_IN_OI'] > 0, df_tech['CALL_CHNG_IN_OI'], 0)
    df_tech['CALL_OI_LOSS'] = np.where(df_tech['CALL_CHNG_IN_OI'] < 0, -df_tech['CALL_CHNG_IN_OI'], 0)
    df_tech['PUT_OI_GAIN'] = np.where(df_tech['PUT_CHNG_IN_OI'] > 0, df_tech['PUT_CHNG_IN_OI'], 0)
    df_tech['PUT_OI_LOSS'] = np.where(df_tech['PUT_CHNG_IN_OI'] < 0, -df_tech['PUT_CHNG_IN_OI'], 0)
    
    # Simple moving averages
    df_tech['SMA_CALL_OI'] = df_tech['CALL_OI'].rolling(window=window, min_periods=1).mean()
    df_tech['SMA_PUT_OI'] = df_tech['PUT_OI'].rolling(window=window, min_periods=1).mean()
    df_tech['SMA_TOTAL_OI'] = df_tech['TOTAL_OI'].rolling(window=window, min_periods=1).mean()
    
    # Exponential moving averages
    df_tech['EMA_CALL_OI'] = df_tech['CALL_OI'].ewm(span=window, min_periods=1).mean()
    df_tech['EMA_PUT_OI'] = df_tech['PUT_OI'].ewm(span=window, min_periods=1).mean()
    
    # Bollinger Bands for OI
    df_tech['CALL_OI_MA'] = df_tech['CALL_OI'].rolling(window=window, min_periods=1).mean()
    df_tech['CALL_OI_STD'] = df_tech['CALL_OI'].rolling(window=window, min_periods=1).std()
    df_tech['CALL_OI_UPPER'] = df_tech['CALL_OI_MA'] + (df_tech['CALL_OI_STD'] * 2)
    df_tech['CALL_OI_LOWER'] = df_tech['CALL_OI_MA'] - (df_tech['CALL_OI_STD'] * 2)
    
    df_tech['PUT_OI_MA'] = df_tech['PUT_OI'].rolling(window=window, min_periods=1).mean()
    df_tech['PUT_OI_STD'] = df_tech['PUT_OI'].rolling(window=window, min_periods=1).std()
    df_tech['PUT_OI_UPPER'] = df_tech['PUT_OI_MA'] + (df_tech['PUT_OI_STD'] * 2)
    df_tech['PUT_OI_LOWER'] = df_tech['PUT_OI_MA'] - (df_tech['PUT_OI_STD'] * 2)
    
    # MACD for OI momentum
    ema12_call = df_tech['CALL_OI'].ewm(span=12, min_periods=1).mean()
    ema26_call = df_tech['CALL_OI'].ewm(span=26, min_periods=1).mean()
    df_tech['MACD_CALL'] = ema12_call - ema26_call
    df_tech['MACD_SIGNAL_CALL'] = df_tech['MACD_CALL'].ewm(span=9, min_periods=1).mean()
    df_tech['MACD_HIST_CALL'] = df_tech['MACD_CALL'] - df_tech['MACD_SIGNAL_CALL']
    
    ema12_put = df_tech['PUT_OI'].ewm(span=12, min_periods=1).mean()
    ema26_put = df_tech['PUT_OI'].ewm(span=26, min_periods=1).mean()
    df_tech['MACD_PUT'] = ema12_put - ema26_put
    df_tech['MACD_SIGNAL_PUT'] = df_tech['MACD_PUT'].ewm(span=9, min_periods=1).mean()
    df_tech['MACD_HIST_PUT'] = df_tech['MACD_PUT'] - df_tech['MACD_SIGNAL_PUT']
    
    return df_tech