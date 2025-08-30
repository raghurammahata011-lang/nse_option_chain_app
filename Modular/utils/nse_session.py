# utils/nse_session.py
import requests
import streamlit as st
import time
import random

@st.cache_data(ttl=300)
def get_nse_session():
    """
    Create a requests session with common NSE headers to fetch option chain.
    Using a session helps maintain cookies and reduces repeated costs.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain"
    })
    # Try a warm-up GET to set cookies (may fail inside some sandboxed envs)
    try:
        s.get("https://www.nseindia.com/option-chain", timeout=8)
    except Exception:
        pass
    return s