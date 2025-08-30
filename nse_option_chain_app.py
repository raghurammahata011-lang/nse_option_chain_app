# nse_option_chain_app.py
# Professional, full-featured Streamlit app for NSE option chain analytics,
# watchlist, strategy signals, and backtesting (underlying via yfinance).
#
# Requirements:
# streamlit, pandas, numpy, plotly, scikit-learn, openpyxl, requests, kaleido, yfinance

import time
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.drawing.image import Image as XLImage
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import os
import concurrent.futures
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
SAVE_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "NSE_STOCK")
os.makedirs(SAVE_FOLDER, exist_ok=True)
INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
# tune this if necessary
THREAD_WORKERS = 6

# ---------------- PROFESSIONAL CSS ----------------
PRO_CSS = """
<style>
:root{--card-bg:#fff;--accent:#0b69ff;--muted:#6b7280}
body {background: linear-gradient(180deg,#f6f8fb 0%, #ffffff 100%);}
.header {display:flex;align-items:center;gap:16px;padding:12px;border-radius:10px;margin-bottom:8px;}
.app-title{font-size:26px;font-weight:700;color:var(--accent)}
.app-sub{color:var(--muted);font-size:13px}
.kpi{background:var(--card-bg);border-radius:10px;padding:12px;box-shadow:0 6px 18px rgba(11,105,255,0.06);}
.kpi .value{font-size:18px;font-weight:700}
.kpi .label{font-size:12px;color:var(--muted)}
.section{background:var(--card-bg);padding:14px;border-radius:10px;margin-bottom:12px}
.small{font-size:12px;color:var(--muted)}
.code{background:#f3f4f6;padding:8px;border-radius:6px;font-family:monospace}
.card{border-radius:10px;padding:12px;background:#fff;box-shadow:0 4px 12px rgba(0,0,0,0.06);margin-bottom:12px}
</style>
"""

# ---------------- UTILITY: NSE Session ----------------
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

# ---------------- FETCH / PARSE ----------------
def fetch_option_chain(symbol, session):
    """
    Fetch option-chain JSON for a symbol. Retries 3 times with backoff.
    Returns JSON or None.
    """
    url = (f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
           if symbol in INDICES else f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}")
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # backoff with jitter
            time.sleep(random.uniform(1.0, 2.5) * (attempt + 1))
    return None

def parse_data(symbol, data):
    """
    Convert NSE option-chain JSON to a DataFrame for the nearest expiry.
    Columns: STRIKE, CALL_OI, CALL_CHNG_IN_OI, CALL_IV, CALL_LTP, PUT_OI, PUT_CHNG_IN_OI, PUT_IV, PUT_LTP
    """
    if not data:
        return pd.DataFrame()
    records = data.get("records", {}) or {}
    expiry_dates = records.get("expiryDates", []) or []
    if not expiry_dates:
        return pd.DataFrame()
    expiry = expiry_dates[0]
    rows = []
    for item in records.get("data", []) or []:
        if item.get("expiryDate") != expiry:
            continue
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        rows.append({
            "STRIKE": item.get("strikePrice"),
            "CALL_OI": ce.get("openInterest", 0),
            "CALL_CHNG_IN_OI": ce.get("changeinOpenInterest", 0),
            "CALL_IV": ce.get("impliedVolatility", 0),
            "CALL_LTP": ce.get("lastPrice", 0),
            "PUT_OI": pe.get("openInterest", 0),
            "PUT_CHNG_IN_OI": pe.get("changeinOpenInterest", 0),
            "PUT_IV": pe.get("impliedVolatility", 0),
            "PUT_LTP": pe.get("lastPrice", 0)
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("STRIKE").reset_index(drop=True)

# ---------------- ENHANCED ANALYTICS ----------------
def calculate_analytics(df, spot_price=None):
    """
    Enhanced analytics:
    - total/ATM PCR
    - true max pain (loss-based)
    - support/resistance by OI buildup
    - IV skew and approx expected 30-day move
    - directional score
    Returns a dict with rich metrics and modified df (with extras).
    """
    if df.empty:
        return {}
    df = df.copy()
    # estimate spot if not provided: strike with smallest difference between call & put OI
    if not spot_price or spot_price == 0:
        df['OI_DIFF'] = (df['CALL_OI'] - df['PUT_OI']).abs()
        spot_price = int(df.loc[df['OI_DIFF'].idxmin(), "STRIKE"])
    df['TOTAL_OI'] = df['CALL_OI'] + df['PUT_OI']
    df['DELTA_CALL'] = df['CALL_OI'] / df['TOTAL_OI'].replace(0, 1)
    df['DELTA_PUT'] = df['PUT_OI'] / df['TOTAL_OI'].replace(0, 1)
    df['IV_DIFF'] = df['CALL_IV'] - df['PUT_IV']
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

    # volume ratio and direction score
    call_vol = df['CALL_CHNG_IN_OI'].sum()
    put_vol = df['PUT_CHNG_IN_OI'].sum()
    vol_ratio = (call_vol + 1) / (put_vol + 1)
    dir_score = 0.0
    dir_score += (pcr - 1.0) * 2.0
    dir_score += (iv_skew / 5.0)
    dir_score += (vol_ratio - 1.0)
    if dir_score > 1.2:
        direction = "Bullish"
    elif dir_score < -1.2:
        direction = "Bearish"
    else:
        direction = "Neutral"

    top_calls = df.nlargest(5, 'CALL_OI')[['STRIKE', 'CALL_OI', 'CALL_IV', 'CALL_CHNG_IN_OI']]
    top_puts = df.nlargest(5, 'PUT_OI')[['STRIKE', 'PUT_OI', 'PUT_IV', 'PUT_CHNG_IN_OI']]

    return {
        "df": df,
        "spot_price": spot_price,
        "pcr": round(pcr, 2),
        "pcr_atm": round(atm_pcr, 2),
        "max_pain": max_pain,
        "support": strongest_support,
        "resistance": strongest_resistance,
        "iv_skew": iv_skew,
        "expected_move_30d": expected_move_30d,
        "direction": direction,
        "dir_score": round(dir_score, 2),
        "top_calls": top_calls,
        "top_puts": top_puts
    }

# ---------------- ML: Regression & Classification ----------------
def train_ml_models_regression(df):
    """
    Regression models to predict 'attractive' strikes (explainable).
    Returns models' results dict and recommended top calls/puts by prediction.
    """
    if df.empty or len(df) < 10:
        return {}, [], []
    X = df[['CALL_OI', 'PUT_OI', 'CALL_CHNG_IN_OI', 'PUT_CHNG_IN_OI', 'CALL_IV', 'PUT_IV']].fillna(0)
    y = df['STRIKE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "RF": RandomForestRegressor(n_estimators=150, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=150, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"model": model, "mae": mae, "r2": r2, "scaler": scaler}

    best_name = min(results, key=lambda k: results[k]['mae'])
    best_model = results[best_name]['model']
    best_scaler = results[best_name]['scaler']

    X_full = scaler.transform(X)
    df['ML_PREDICTED_STRIKE'] = best_model.predict(X_full)
    top_calls = df.nlargest(3, 'ML_PREDICTED_STRIKE')['STRIKE'].tolist()
    top_puts = df.nsmallest(3, 'ML_PREDICTED_STRIKE')['STRIKE'].tolist()

    return results, top_calls, top_puts

def train_ml_models_classification(df):
    """
    Classification to detect bias from options features.
    Simple labeled by PCR thresholds.
    """
    if df.empty or len(df) < 12:
        return {"RF": 0.0, "LR": 0.0}, "Neutral"
    d = df.copy()
    d['PCR'] = d['PUT_OI'] / (d['CALL_OI'] + 1)
    d['IVS'] = d['CALL_IV'] - d['PUT_IV']
    features = ['CALL_OI', 'PUT_OI', 'PCR', 'IVS', 'CALL_CHNG_IN_OI', 'PUT_CHNG_IN_OI']
    d['LABEL'] = np.where(d['PCR'] > 1.2, 'Bullish', np.where(d['PCR'] < 0.8, 'Bearish', 'Neutral'))
    X = d[features].fillna(0)
    y = d['LABEL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    # consensus label from RF predictions on the set (mode)
    preds = rf.predict(X)
    consensus = pd.Series(preds).mode().iloc[0] if len(preds) > 0 else "Neutral"
    return {"RF": round(acc_rf, 3), "LR": round(acc_lr, 3)}, consensus

# ---------------- CHARTS ----------------
def create_oi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['CALL_OI'], mode='lines', name='Call OI'))
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['PUT_OI'], mode='lines', name='Put OI'))
    fig.update_layout(title="Open Interest Distribution", xaxis_title="Strike", yaxis_title="OI",
                      height=320, margin=dict(t=40, b=30, l=40, r=20))
    return fig

def create_sentiment_chart(df):
    df_local = df.copy()
    df_local['SENT'] = df_local['CALL_OI'] - df_local['PUT_OI']
    fig = go.Figure([go.Bar(x=df_local['STRIKE'], y=df_local['SENT'])])
    fig.update_layout(title="Sentiment (Call OI - Put OI)", xaxis_title="Strike", yaxis_title="Call-Put OI",
                      height=320, margin=dict(t=40, b=40))
    return fig

def create_iv_comparison_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['CALL_IV'], name='Call IV'))
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['PUT_IV'], name='Put IV'))
    fig.update_layout(title="Implied Volatility (Call vs Put)", xaxis_title="Strike", yaxis_title="IV",
                      height=320, margin=dict(t=40, b=40))
    return fig

def create_ml_prediction_chart(df, analytics, top_calls, top_puts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['TOTAL_OI'], mode='lines+markers', name='Total OI'))
    if 'ML_PREDICTED_STRIKE' in df.columns:
        fig.add_trace(go.Scatter(x=df['ML_PREDICTED_STRIKE'], y=df['TOTAL_OI'], mode='markers', name='ML Predicted', marker=dict(size=8)))
    fig.add_vline(x=analytics['max_pain'] if 'max_pain' in analytics else analytics.get('max_pain', None), line_dash='dash', annotation_text='Max Pain')
    for s in top_calls:
        fig.add_vline(x=s, line_dash='dot', line_color='green')
    for s in top_puts:
        fig.add_vline(x=s, line_dash='dot', line_color='red')
    fig.update_layout(title="ML Predicted Strikes & OI", xaxis_title="Strike", yaxis_title="Total OI", height=360)
    return fig

def create_model_performance_chart(ml_results):
    if not ml_results:
        return go.Figure()
    models = list(ml_results.keys())
    mae = [ml_results[m]['mae'] for m in models]
    r2 = [ml_results[m]['r2'] for m in models]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=mae, name='MAE'))
    fig.add_trace(go.Line(x=models, y=r2, name='R2'))
    fig.update_layout(title="Model Performance", height=320)
    return fig

# ---------------- EXCEL EXPORT ----------------
def save_to_excel(df, analytics, symbol, ml_results, top_calls, top_puts):
    """
    Writes:
     - OptionChain sheet: full dataframe with formatting
     - Analytics sheet: key metrics
     - ML_Results sheet: model metrics
     - Chart sheets: exported PNGs embedded in their own sheets (requires kaleido)
    """
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "OptionChain"

    header_fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
    bold_font = Font(bold=True)
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))

    # Write option chain
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws1.cell(row=r_idx, column=c_idx, value=value)
            cell.alignment = center_align
            cell.border = thin_border
            if r_idx == 1:
                cell.fill = header_fill
                cell.font = bold_font

    # Analytics sheet
    ws2 = wb.create_sheet("Analytics")
    analytics_rows = [
        ("Symbol", symbol),
        ("Spot Price", analytics.get("spot_price")),
        ("Direction", analytics.get("direction")),
        ("Directional Score", analytics.get("dir_score")),
        ("PCR (overall)", analytics.get("pcr")),
        ("PCR (ATM)", analytics.get("pcr_atm")),
        ("Max Pain", analytics.get("max_pain")),
        ("Support", analytics.get("support")),
        ("Resistance", analytics.get("resistance")),
        ("IV Skew", analytics.get("iv_skew")),
        ("Expected 30d Move", analytics.get("expected_move_30d"))
    ]
    for i, (k, v) in enumerate(analytics_rows, 1):
        ws2.cell(row=i, column=1, value=k).font = bold_font
        ws2.cell(row=i, column=2, value=v)

    # ML results
    ws3 = wb.create_sheet("ML_Results")
    ws3.append(["Model", "MAE", "R2"])
    for m, res in (ml_results or {}).items():
        ws3.append([m, res.get("mae", ""), res.get("r2", "")])

    # Chart sheets
    chart_map = {
        "OI_Chart": create_oi_chart(df),
        "Sentiment_Chart": create_sentiment_chart(df),
        "IV_Chart": create_iv_comparison_chart(df)
    }
    if "ML_PREDICTED_STRIKE" in df.columns:
        chart_map["ML_Predictions"] = create_ml_prediction_chart(df, analytics, top_calls, top_puts)
        chart_map["ML_Performance"] = create_model_performance_chart(ml_results)

    for sheet_name, fig in chart_map.items():
        ws = wb.create_sheet(sheet_name)
        try:
            img_data = BytesIO(pio.to_image(fig, format="png"))
            img = XLImage(img_data)
            img.anchor = "A1"
            ws.add_image(img)
        except Exception as e:
            # If to_image fails (kaleido missing), write the fig as JSON summary
            ws.cell(row=1, column=1, value=f"Chart export failed: {e}")

    # Save
    file_path = os.path.join(SAVE_FOLDER, f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    wb.save(file_path)
    return file_path

# ---------------- WATCHLIST (parallel) ----------------
def fetch_for_watchlist(symbol):
    """Helper to fetch and compute analytics for a single symbol (used in parallel)."""
    session = get_nse_session()
    data = fetch_option_chain(symbol, session)
    if not data:
        return symbol, None
    df = parse_data(symbol, data)
    if df.empty:
        return symbol, None
    analytics = calculate_analytics(df)
    return symbol, analytics

def watchlist_snapshot(symbols):
    """
    Fetch analytics for all symbols in watchlist in parallel and return a snapshot DataFrame.
    """
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(THREAD_WORKERS, max(1, len(symbols)))) as ex:
        futures = {ex.submit(fetch_for_watchlist, sym): sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures):
            sym = futures[fut]
            try:
                s, analytics = fut.result()
                results[s] = analytics
            except Exception:
                results[sym] = None
    # Build DataFrame
    rows = []
    for sym in symbols:
        a = results.get(sym)
        if not a:
            rows.append({"Symbol": sym, "Spot": None, "Direction": "N/A", "PCR": None, "MaxPain": None, "Support": None, "Resistance": None})
        else:
            rows.append({"Symbol": sym, "Spot": a.get("spot_price"), "Direction": a.get("direction"),
                         "PCR": a.get("pcr"), "MaxPain": a.get("max_pain"), "Support": a.get("support"), "Resistance": a.get("resistance")})
    return pd.DataFrame(rows)

# ---------------- STRATEGY SIGNALS ----------------
def generate_strategy_signals(analytics):
    """
    From single-symbol analytics, produce human-friendly strategy suggestions
    with rationale.
    Returns list of dicts {strategy, rationale, risk_note}.
    """
    signals = []
    if not analytics:
        return signals
    pcr = analytics.get("pcr", 0)
    iv_skew = analytics.get("iv_skew", 0)
    direction = analytics.get("direction", "Neutral")
    support = analytics.get("support")
    resistance = analytics.get("resistance")
    max_pain = analytics.get("max_pain")

    # Directional ideas
    if pcr > 1.2 and direction == "Bullish":
        signals.append({
            "strategy": "Bull Put Spread",
            "rationale": f"High PCR ({pcr}) + Bullish direction → Put selling bias. Use a bull-put vertical near support ({support}).",
            "risk": "Requires margin; limited risk if proper strike selection."
        })
    elif pcr < 0.8 and direction == "Bearish":
        signals.append({
            "strategy": "Bear Call Spread",
            "rationale": f"Low PCR ({pcr}) + Bearish direction → Call selling bias. Use a bear-call vertical near resistance ({resistance}).",
            "risk": "Limited profit, limited loss. Monitor IV changes."
        })
    else:
        signals.append({
            "strategy": "Neutral / Income (Iron Condor)",
            "rationale": f"PCR neutral ({pcr}) and range between {support} and {resistance}. Consider income strategies (iron condor).",
            "risk": "Complex multi-leg; requires margin and careful width selection."
        })

    # Volatility ideas
    if iv_skew > 2:
        signals.append({
            "strategy": "Long Call Calendar / Long Straddle (vol buy)",
            "rationale": f"Positive IV skew ({iv_skew}) indicates upside priced; consider long volatility strategies.",
            "risk": "Premium may erode if no move; theta can hurt."
        })
    elif iv_skew < -2:
        signals.append({
            "strategy": "Short Strangle or Iron Condor (vol sell)",
            "rationale": f"Negative IV skew ({iv_skew}) indicates downside priced; if neutral, sell premium.",
            "risk": "High tail risk; use defined-risk structures or hedges."
        })

    # Max Pain note
    signals.append({
        "strategy": "Max Pain Watch",
        "rationale": f"Max pain at {max_pain} — price often gravitates toward this near expiry. Use as reference for expiry-targeted trades.",
        "risk": "Not a guarantee — price may diverge if underlying fundamentals change."
    })
    return signals

# ---------------- BACKTEST (underlying via yfinance) ----------------
def generate_signals_time_series(df_analytics_history):
    """
    Input: DataFrame indexed by date with analytics columns (pcr, direction).
    Simple rule:
      - Enter LONG when direction turns Bullish
      - Exit LONG when direction turns Neutral/Bearish
    Similar for SHORT when direction turns Bearish.
    Returns list of trades and performance metrics.
    """
    trades = []
    position = None
    entry_price = None
    entry_date = None

    for date, row in df_analytics_history.iterrows():
        dir_label = row.get("direction")
        price = row.get("close")  # underlying close price

        if position is None:
            if dir_label == "Bullish":
                position = "LONG"
                entry_price = price
                entry_date = date
            elif dir_label == "Bearish":
                position = "SHORT"
                entry_price = price
                entry_date = date
        else:
            # close LONG on bearish or neutral
            if position == "LONG" and dir_label in ("Bearish", "Neutral"):
                pnl = (price - entry_price)
                trades.append({"entry_date": entry_date, "exit_date": date, "position": position, "entry": entry_price, "exit": price, "pnl": pnl})
                position = None
            elif position == "SHORT" and dir_label in ("Bullish", "Neutral"):
                pnl = (entry_price - price)
                trades.append({"entry_date": entry_date, "exit_date": date, "position": position, "entry": entry_price, "exit": price, "pnl": pnl})
                position = None
    # if position still open at end, close at last price
    if position is not None and entry_price is not None:
        last_row = df_analytics_history.iloc[-1]
        last_price = last_row.get("close")
        date = df_analytics_history.index[-1]
        if position == "LONG":
            pnl = (last_price - entry_price)
        else:
            pnl = (entry_price - last_price)
        trades.append({"entry_date": entry_date, "exit_date": date, "position": position, "entry": entry_price, "exit": last_price, "pnl": pnl})
    return trades

def backtest_underlying(symbol, analytics_series, start_date, end_date):
    """
    Run backtest using underlying OHLC via yfinance and analytics_series (per day).
    analytics_series: DataFrame indexed by date containing 'direction' produced by option snapshots.
    We'll align times by business date; this is a simple overlay backtest to evaluate signals vs underlying moves.
    """
    # Fetch underlying history
    # For indices like NIFTY/BANKNIFTY, yfinance tickers may vary (use '^NSEI' for NIFTY?); user may pass symbols that exist on yfinance.
    yf_symbol = symbol if symbol not in ("NIFTY", "BANKNIFTY") else None
    # Try common index mappings:
    idx_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK"
    }
    if symbol in idx_map:
        yf_symbol = idx_map[symbol]
    if yf_symbol is None:
        return {"error": "No yfinance mapping for symbol."}
    hist = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
    if hist.empty:
        return {"error": "yfinance returned no data for symbol."}
    hist = hist[['Close']].rename(columns={'Close': 'close'})
    hist.index = pd.to_datetime(hist.index.date)
    # Combine analytics_series (assumed daily) with hist by date
    df = hist.join(analytics_series[['direction']], how='left')
    df['direction'] = df['direction'].ffill().fillna('Neutral')
    # Generate trades
    trades = generate_signals_time_series(df)
    # Performance metrics
    pnl_list = [t['pnl'] for t in trades] if trades else []
    total_pnl = sum(pnl_list)
    win_count = sum(1 for p in pnl_list if p > 0)
    loss_count = sum(1 for p in pnl_list if p <= 0)
    win_rate = win_count / len(pnl_list) if pnl_list else 0
    avg_pnl = (np.mean(pnl_list) if pnl_list else 0)
    # Max drawdown of equity curve
    equity = np.cumsum(pnl_list) if pnl_list else np.array([0.0])
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity)
    max_dd = np.max(dd) if len(dd) > 0 else 0.0
    return {
        "trades": trades,
        "total_pnl": total_pnl,
        "win_rate": round(win_rate, 4),
        "avg_pnl": round(avg_pnl, 4),
        "max_drawdown": round(max_dd, 4),
        "equity_curve": equity.tolist()
    }

# ---------------- STREAMLIT APP ----------------
def run_streamlit_app():
    st.set_page_config(page_title="NSE Option Chain Professional", layout="wide")
    st.markdown(PRO_CSS, unsafe_allow_html=True)
    # HEADER
    left, right = st.columns([8, 2])
    with left:
        st.markdown('<div class="header"><div class="app-title">NSE Option Chain — Professional</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Realtime option chain analytics, watchlist, strategy signals and backtesting (yfinance underlying)</div>', unsafe_allow_html=True)
    with right:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/24/Chart_up_trend_icon.svg", width=60)

    # SIDEBAR configuration
    with st.sidebar:
        st.header("Configuration")
        symbol = st.text_input("Primary symbol (for single-symbol analysis)", "NIFTY").upper()
        spot_price = st.number_input("Spot price (optional, numeric)", value=0.0)
        enable_ml = st.checkbox("Enable ML Regression", value=True)
        enable_class = st.checkbox("Enable ML Classification", value=True)
        watchlist_raw = st.text_area("Watchlist (comma separated)", value="NIFTY,BANKNIFTY")
        enable_export = st.checkbox("Enable Excel export (charts embedded)", value=True)
        backtest_period_days = st.number_input("Backtest period (days)", min_value=30, max_value=3650, value=180, step=30)
        st.write("Disclaimer: This analysis is for informational purposes only. It is not financial advice. Use at your own risk—creators are not responsible for any losses or market volatility")

    # BUTTONS
    fetch_clicked = st.button("Fetch & Analyze")
    if not fetch_clicked:
        st.info("Enter symbol and click 'Fetch & Analyze' to start.")
        # still show watchlist snapshot optionally
        if st.sidebar.button("Refresh Watchlist Snapshot"):
            watchlist_syms = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]
            if watchlist_syms:
                with st.spinner("Fetching watchlist..."):
                    snap = watchlist_snapshot(watchlist_syms)
                    st.dataframe(snap)
        return

    # MAIN: fetch and analyze primary symbol
    session = get_nse_session()
    data_json = fetch_option_chain(symbol, session)
    if not data_json:
        st.error("Failed to fetch option chain from NSE. Check symbol or try again later.")
        return
    df = parse_data(symbol, data_json)
    if df.empty:
        st.error("No option chain data available (perhaps wrong symbol or no current expiry).")
        return

    # Estimate spot if not provided
    if not spot_price or spot_price == 0:
        df['OI_DIFF'] = (df['CALL_OI'] - df['PUT_OI']).abs()
        spot_price = int(df.loc[df['OI_DIFF'].idxmin(), 'STRIKE']) if not df.empty else 0
        st.info(f"Estimated spot price: {spot_price}")

    analytics = calculate_analytics(df, spot_price)

    # ML
    ml_results, ml_top_calls, ml_top_puts = {}, [], []
    if enable_ml:
        with st.spinner("Training regression models..."):
            ml_results, ml_top_calls, ml_top_puts = train_ml_models_regression(analytics['df'])

    class_results, consensus_label = {}, "Neutral"
    if enable_class:
        with st.spinner("Training classification models..."):
            class_results, consensus_label = train_ml_models_classification(analytics['df'])

    # KPI cards row
    kpis = [
        ("Direction", analytics['direction']),
        ("PCR", analytics['pcr']),
        ("Max Pain", analytics['max_pain']),
        ("Exp. 30d Move", analytics['expected_move_30d'])
    ]
    cols = st.columns(len(kpis))
    for c, (lbl, val) in zip(cols, kpis):
        c.markdown(f"<div class='kpi'><div class='label'>{lbl}</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)

    # Insights / signals
    st.markdown("---")
    st.subheader("Professional Insights & Strategy Signals")
    insights = [
        f"Market direction: **{analytics['direction']}** (score {analytics['dir_score']})",
        f"PCR (overall): **{analytics['pcr']}** | PCR (ATM window): **{analytics['pcr_atm']}**",
        f"Expected 30-day move (approx): **{analytics['expected_move_30d']}** (absolute price move)",
        f"Max Pain: **{analytics['max_pain']}**; Support: **{analytics['support']}**, Resistance: **{analytics['resistance']}**"
    ]
    for itm in insights:
        st.info(itm)

    # Strategy suggestions
    signals = generate_strategy_signals(analytics)
    st.markdown("#### Suggested Strategies")
    for s in signals:
        st.markdown(f"**{s['strategy']}** — {s['rationale']}")
        st.markdown(f"<span class='small'>Risk note: {s['risk']}</span>", unsafe_allow_html=True)

    # TABS: Option chain, Charts, ML, Watchlist, Backtest, Export
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Option Chain Table", "Charts", "ML Results", "Watchlist", "Backtest"])
    with tab1:
        st.dataframe(analytics['df'], use_container_width=True)
        st.markdown("**Top Calls**")
        st.dataframe(analytics['top_calls'])
        st.markdown("**Top Puts**")
        st.dataframe(analytics['top_puts'])

    with tab2:
        st.plotly_chart(create_oi_chart(analytics['df']), use_container_width=True)
        st.plotly_chart(create_sentiment_chart(analytics['df']), use_container_width=True)
        st.plotly_chart(create_iv_comparison_chart(analytics['df']), use_container_width=True)
        if 'ML_PREDICTED_STRIKE' in analytics['df'].columns:
            st.plotly_chart(create_ml_prediction_chart(analytics['df'], analytics, ml_top_calls, ml_top_puts), use_container_width=True)

    with tab3:
        st.subheader("Regression Model Performance")
        if ml_results:
            perf = {k: {"MAE": v['mae'], "R2": v['r2']} for k, v in ml_results.items()}
            st.table(pd.DataFrame(perf).T)
            st.plotly_chart(create_model_performance_chart(ml_results), use_container_width=True)
            st.markdown("**ML recommended calls**: " + (", ".join(map(str, ml_top_calls)) if ml_top_calls else "N/A"))
            st.markdown("**ML recommended puts**: " + (", ".join(map(str, ml_top_puts)) if ml_top_puts else "N/A"))
        else:
            st.info("ML disabled or not enough data.")

        st.subheader("Classification Model")
        st.write("Models accuracy:", class_results)
        st.info(f"Consensus label from options features: **{consensus_label}**")

    with tab4:
        st.subheader("Watchlist Snapshot")
        watchlist_syms = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]
        if watchlist_syms:
            with st.spinner("Fetching watchlist data..."):
                snap = watchlist_snapshot(watchlist_syms)
            st.dataframe(snap)
        else:
            st.info("Add symbols to the watchlist in the sidebar.")

    with tab5:
        st.subheader("Backtest (Underlying via yfinance)")
        # Build a simple daily analytics series: ideally you would capture analytics over time,
        # but here we simulate by using the single analytics snapshot repeated, or optionally allow rolling snapshots.
        # Better approach: user should schedule daily exports to create a real analytics time series.
        backtest_symbol = symbol
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=int(backtest_period_days))
        st.write(f"Backtest underlying: {backtest_symbol} from {start_date} to {end_date}")
        # Build mock analytics_series: for demo we will use the option snapshot's direction as repeated daily signal
        # In real deployment you should collect daily analytics snapshots to a time-series DB and feed that here.
        date_index = pd.date_range(start=start_date, end=end_date, freq='B')  # business days
        analytics_series = pd.DataFrame(index=date_index)
        analytics_series['direction'] = analytics['direction']  # repeated
        try:
            bt = backtest_underlying(backtest_symbol, analytics_series, start_date.strftime("%Y-%m-%d"), (end_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if 'error' in bt:
                st.error(bt['error'])
            else:
                st.metric("Total PnL", bt['total_pnl'])
                st.metric("Win Rate", f"{bt['win_rate']*100:.2f}%")
                st.metric("Avg PnL per trade", bt['avg_pnl'])
                st.metric("Max Drawdown", bt['max_drawdown'])
                st.write("Trades:")
                st.dataframe(pd.DataFrame(bt['trades']))
                # equity curve plot
                if bt['equity_curve']:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(y=bt['equity_curve'], mode='lines+markers', name='Equity'))
                    fig_eq.update_layout(title="Backtest Equity Curve", height=300)
                    st.plotly_chart(fig_eq, use_container_width=True)
        except Exception as e:
            st.error(f"Backtest error: {e}")

    # Export button
    st.markdown("---")
    if enable_export:
        if st.button("Export Analysis to Excel (with Charts)"):
            with st.spinner("Exporting to Excel..."):
                try:
                    path = save_to_excel(analytics['df'], analytics, symbol, ml_results, ml_top_calls, ml_top_puts)
                    st.success(f"Exported to: {path}")
                    with open(path, "rb") as f:
                        st.download_button("Download Excel", data=f, file_name=os.path.basename(path))
                except Exception as e:
                    st.error(f"Export failed: {e}")

if __name__ == "__main__":
    run_streamlit_app()
