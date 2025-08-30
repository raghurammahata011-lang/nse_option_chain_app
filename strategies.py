import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

# -----------------------------
# Black-Scholes Greeks
# -----------------------------
def option_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes Greeks.
    S = Spot price
    K = Strike price
    T = Time to expiry (in years)
    r = Risk-free rate (decimal, e.g. 0.06 for 6%)
    sigma = Implied Volatility (decimal, e.g. 0.20 for 20%)
    option_type = "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T))
                 - r*K*math.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        delta = -norm.cdf(-d1)
        theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T))
                 + r*K*math.exp(-r*T)*norm.cdf(-d2)) / 365

    gamma = norm.pdf(d1)/(S*sigma*math.sqrt(T))
    vega = S*norm.pdf(d1)*math.sqrt(T)/100  # per 1% IV change

    return {
        "delta": round(delta,4),
        "gamma": round(gamma,4),
        "theta": round(theta,4),
        "vega": round(vega,4)
    }

# -----------------------------
# Payoff Simulator
# -----------------------------
def payoff_option(S, K, premium, option_type="call", position="long"):
    """Payoff for a single option leg"""
    if option_type == "call":
        payoff = np.maximum(S-K,0) - premium
    else:
        payoff = np.maximum(K-S,0) - premium
    return payoff if position=="long" else -payoff

def strategy_payoff(strategy, spot, price_range):
    """Aggregate payoff for multi-leg strategy"""
    S = np.linspace(price_range[0], price_range[1], 200)
    total_payoff = np.zeros_like(S)
    for leg in strategy:
        total_payoff += payoff_option(S, leg["K"], leg["premium"], leg["type"], leg["pos"])
    return S, total_payoff

def plot_strategy(strategy_name, strategy, spot, price_range):
    """Plot strategy payoff curve"""
    S, payoff = strategy_payoff(strategy, spot, price_range)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(S, payoff, label=strategy_name, color="blue")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(spot, color="red", linestyle="--", label="Spot")
    ax.set_title(f"{strategy_name} Payoff")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Profit / Loss")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Strategy PnL Table
# -----------------------------
def strategy_summary(strategy, spot, price_range):
    """Return max profit, max loss, breakevens"""
    S, payoff = strategy_payoff(strategy, spot, price_range)
    max_profit = np.max(payoff)
    max_loss = np.min(payoff)
    
    # Breakeven points = where payoff crosses 0
    breakevens = []
    for i in range(1, len(S)):
        if payoff[i-1] <= 0 < payoff[i] or payoff[i-1] >= 0 > payoff[i]:
            breakevens.append(round(S[i],2))

    return {
        "Max Profit": round(max_profit,2) if max_profit < 1e6 else "Unlimited",
        "Max Loss": round(max_loss,2) if max_loss > -1e6 else "Unlimited",
        "Breakevens": breakevens
    }
