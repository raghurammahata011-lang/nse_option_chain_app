# strategies/options_strategies.py
import numpy as np
import pandas as pd
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price and Greeks
    """
    if T <= 0 or sigma <= 0:
        return {
            'price': 0,
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type.lower() == "call" else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'price': round(price, 2),
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4)
    }

def calculate_greeks_for_chain(df, spot_price, days_to_expiry=30, risk_free_rate=0.05):
    """
    Calculate Greeks for all options in the chain
    """
    df_greeks = df.copy()
    
    for idx, row in df_greeks.iterrows():
        # Call Greeks
        if row['CALL_IV'] > 0:
            call_greeks = black_scholes_greeks(
                spot_price, row['STRIKE'], days_to_expiry/365, 
                risk_free_rate, row['CALL_IV']/100, "call"
            )
            df_greeks.at[idx, 'CALL_DELTA'] = call_greeks['delta']
            df_greeks.at[idx, 'CALL_GAMMA'] = call_greeks['gamma']
            df_greeks.at[idx, 'CALL_THETA'] = call_greeks['theta']
            df_greeks.at[idx, 'CALL_VEGA'] = call_greeks['vega']
        
        # Put Greeks
        if row['PUT_IV'] > 0:
            put_greeks = black_scholes_greeks(
                spot_price, row['STRIKE'], days_to_expiry/365, 
                risk_free_rate, row['PUT_IV']/100, "put"
            )
            df_greeks.at[idx, 'PUT_DELTA'] = put_greeks['delta']
            df_greeks.at[idx, 'PUT_GAMMA'] = put_greeks['gamma']
            df_greeks.at[idx, 'PUT_THETA'] = put_greeks['theta']
            df_greeks.at[idx, 'PUT_VEGA'] = put_greeks['vega']
    
    return df_greeks

def suggest_strategies(analytics, df_greeks):
    """
    Suggest option strategies based on market conditions
    """
    strategies = []
    
    # Bullish strategies
    if analytics['direction'] == "Bullish":
        if analytics['iv_like'] > 20:  # High volatility
            strategies.append({
                'name': 'Bull Call Spread',
                'description': 'Buy lower strike call, sell higher strike call',
                'risk': 'Limited',
                'reward': 'Limited',
                'breakeven': 'Lower strike + net debit'
            })
        else:  # Low volatility
            strategies.append({
                'name': 'Long Call',
                'description': 'Buy call option',
                'risk': 'Limited to premium paid',
                'reward': 'Unlimited',
                'breakeven': 'Strike price + premium paid'
            })
    
    # Bearish strategies
    elif analytics['direction'] == "Bearish":
        if analytics['iv_like'] > 20:  # High volatility
            strategies.append({
                'name': 'Bear Put Spread',
                'description': 'Buy higher strike put, sell lower strike put',
                'risk': 'Limited',
                'reward': 'Limited',
                'breakeven': 'Higher strike - net debit'
            })
        else:  # Low volatility
            strategies.append({
                'name': 'Long Put',
                'description': 'Buy put option',
                'risk': 'Limited to premium paid',
                'reward': 'Substantial (but limited)',
                'breakeven': 'Strike price - premium paid'
            })
    
    # Neutral strategies
    else:
        if analytics['iv_like'] > 20:  # High volatility
            strategies.append({
                'name': 'Iron Condor',
                'description': 'Sell OTM call spread and OTM put spread',
                'risk': 'Limited',
                'reward': 'Limited to premium received',
                'breakeven': 'Multiple breakeven points'
            })
        else:  # Low volatility
            strategies.append({
                'name': 'Short Straddle',
                'description': 'Sell ATM call and put',
                'risk': 'Unlimited',
                'reward': 'Limited to premium received',
                'breakeven': 'Strike ± total premium received'
            })
    
    return strategies