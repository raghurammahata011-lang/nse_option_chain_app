# utils/analytics.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def calculate_analytics(df, spot_price=None):
    """
    Enhanced analytics for option chain data
    """
    if df.empty:
        return {}
        
    df = df.copy()
    
    # estimate spot if not provided: strike with smallest difference between call & put OI
    if not spot_price or spot_price == 0:
        df['OI_DIFF'] = (df['CALL_OI'] - df['PUT_OI']).abs()
        spot_price = int(df.loc[df['OI_DIFF'].idxmin(), "STRIKE"])
        
    df['TOTAL_OI'] = df['CALL_OI'] + df['PUT_OI']
    df['OI_RATIO'] = df['PUT_OI'] / (df['CALL_OI'] + 1e-10)  # Avoid division by zero
    df['DELTA_CALL'] = df['CALL_OI'] / df['TOTAL_OI'].replace(0, 1)
    df['DELTA_PUT'] = df['PUT_OI'] / df['TOTAL_OI'].replace(0, 1)
    df['IV_DIFF'] = df['CALL_IV'] - df['PUT_IV']
    df['PRICE_RATIO'] = df['CALL_LTP'] / (df['PUT_LTP'] + 1e-10)
    
    total_call_oi = df['CALL_OI'].sum()
    total_put_oi = df['PUT_OI'].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

    # ATM window: pick nearest strike and +/- 3 strikes (adjust for strike step)
    strike_step = int(df['STRIKE'].diff().median() or 50)
    atm_strike = int(df.iloc[(df['STRIKE'] - spot_price).abs().argsort()[:1]]['STRIKE'].values[0])
    window = 3 * strike_step
    window_df = df[(df['STRIKE'] >= atm_strike - window) & (df['STRIKE'] <= atm_strike + window)]
    atm_pcr = window_df['PUT_OI'].sum() / window_df['CALL_OI'].sum() if window_df['CALL_OI'].sum() > 0 else 0

    # true max pain computation (min aggregated option seller loss)
    strikes = df['STRIKE'].values
    losses = []
    for k in strikes:
        call_loss = ((np.clip(k - strikes, 0, None)) * df['CALL_OI']).sum()
        put_loss = ((np.clip(strikes - k, 0, None)) * df['PUT_OI']).sum()
        losses.append(call_loss + put_loss)
    max_pain = int(strikes[np.argmin(losses)]) if len(losses) > 0 else 0

    # support/resistance by OI buildup and current OI
    df['PUT_SCORE'] = df['PUT_OI'] * (1 + df['PUT_CHNG_IN_OI'] / (df['PUT_OI'].replace(0, 1) + 1))
    df['CALL_SCORE'] = df['CALL_OI'] * (1 + df['CALL_CHNG_IN_OI'] / (df['CALL_OI'].replace(0, 1) + 1))
    strongest_support = int(df.loc[df['PUT_SCORE'].idxmax(), 'STRIKE'])
    strongest_resistance = int(df.loc[df['CALL_SCORE'].idxmax(), 'STRIKE'])

    # IV metrics
    avg_call_iv = df['CALL_IV'].mean()
    avg_put_iv = df['PUT_IV'].mean()
    iv_skew = round(avg_call_iv - avg_put_iv, 2)
    iv_atm = window_df[['CALL_IV', 'PUT_IV']].mean().mean() if not window_df.empty else (avg_call_iv + avg_put_iv) / 2
    expected_move_annual = iv_atm / 100.0
    # approximate 30d expected move (sqrt scaling for monthly ~ sqrt(1/12) of annual vol)
    expected_move_30d = round(spot_price * expected_move_annual / np.sqrt(12), 2)
    
    # Volatility surface analysis
    iv_slope_call = np.polyfit(df['STRIKE'], df['CALL_IV'], 1)[0] if len(df) > 1 else 0
    iv_slope_put = np.polyfit(df['STRIKE'], df['PUT_IV'], 1)[0] if len(df) > 1 else 0
    
    # Option flow analysis
    call_flow = (df['CALL_CHNG_IN_OI'] * df['CALL_LTP']).sum()
    put_flow = (df['PUT_CHNG_IN_OI'] * df['PUT_LTP']).sum()
    net_flow = call_flow - put_flow
    flow_ratio = abs(call_flow / put_flow) if put_flow != 0 else float('inf')

    # volume ratio and direction score
    call_vol = df['CALL_CHNG_IN_OI'].sum()
    put_vol = df['PUT_CHNG_IN_OI'].sum()
    vol_ratio = (call_vol + 1) / (put_vol + 1)
    dir_score = 0.0
    dir_score += (pcr - 1.0) * 2.0
    dir_score += (iv_skew / 5.0)
    dir_score += (vol_ratio - 1.0)
    dir_score += (net_flow / max(abs(net_flow), 1)) * 0.5  # Add flow impact
    
    if dir_score > 1.2:
        direction = "Bullish"
    elif dir_score < -1.2:
        direction = "Bearish"
    else:
        direction = "Neutral"

    top_calls = df.nlargest(5, 'CALL_OI')[['STRIKE', 'CALL_OI', 'CALL_IV', 'CALL_CHNG_IN_OI', 'CALL_LTP']]
    top_puts = df.nlargest(5, 'PUT_OI')[['STRIKE', 'PUT_OI', 'PUT_IV', 'PUT_CHNG_IN_OI', 'PUT_LTP']]
    
    # Option clustering for pattern recognition
    if len(df) > 10:
        try:
            # Prepare data for clustering
            cluster_data = df[['STRIKE', 'CALL_OI', 'PUT_OI', 'CALL_IV', 'PUT_IV']].copy()
            cluster_data = (cluster_data - cluster_data.mean()) / cluster_data.std()
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['CLUSTER'] = kmeans.fit_predict(cluster_data)
        except:
            df['CLUSTER'] = 0

    # Additional financial analysis metrics
    # VIX-like volatility indicator
    vix_like = (avg_call_iv + avg_put_iv) / 2
    
    # Put-Call parity deviation
    put_call_parity_dev = (df['CALL_LTP'] - df['PUT_LTP'] + df['STRIKE'] - spot_price).abs().mean()
    
    # Gamma exposure estimation
    gamma_exposure = (df['CALL_OI'] + df['PUT_OI']).sum() / 1e6  # In millions
    
    return {
        "df": df,
        "spot_price": spot_price,
        "pcr": round(pcr, 2),
        "pcr_atm": round(atm_pcr, 2),
        "max_pain": max_pain,
        "support": strongest_support,
        "resistance": strongest_resistance,
        "iv_skew": iv_skew,
        "iv_slope_call": round(iv_slope_call, 4),
        "iv_slope_put": round(iv_slope_put, 4),
        "expected_move_30d": expected_move_30d,
        "direction": direction,
        "dir_score": round(dir_score, 2),
        "call_flow": round(call_flow, 2),
        "put_flow": round(put_flow, 2),
        "net_flow": round(net_flow, 2),
        "flow_ratio": round(flow_ratio, 2),
        "top_calls": top_calls,
        "top_puts": top_puts,
        "vix_like": round(vix_like, 2),
        "put_call_parity_dev": round(put_call_parity_dev, 2),
        "gamma_exposure": round(gamma_exposure, 2)
    }