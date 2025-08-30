import time, random, requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
SAVE_FOLDER = r"C:\Users\RAGHURAM MAHATA\Desktop\NSE_STOCK"
INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ---------------- UTILITY FUNCTIONS ----------------
@st.cache_data(ttl=300)
def get_nse_session():
    """
    Create a requests session with NSE headers to fetch option chain data.
    No Selenium required. Compatible with Streamlit Cloud.
    """
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.5845.97 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain"
    })

    # Make initial request to set cookies
    try:
        response = session.get("https://www.nseindia.com/option-chain", timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Failed to initialize NSE session: {e}")
    
    return session


def fetch_option_chain(symbol, session):
    """
    Fetch NSE option chain data for the given symbol using a requests session.
    Compatible with Streamlit Cloud (no Selenium required).
    """
    url = (
        f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        if symbol in INDICES
        else f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    )

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            st.error(f"Attempt {attempt+1} failed for {symbol}: {e}")
            time.sleep(random.randint(3, 7))
    return None


def parse_data(symbol, data):
    if not data: return pd.DataFrame()
    expiry_dates = data.get("records", {}).get("expiryDates", [])
    if not expiry_dates: return pd.DataFrame()
    expiry = expiry_dates[0]
    
    records = []
    for item in data.get("records", {}).get("data", []):
        if item.get("expiryDate") != expiry: continue
        ce, pe = item.get("CE", {}), item.get("PE", {})
        records.append({
            "STRIKE": item["strikePrice"],
            "CALL_OI": ce.get("openInterest", 0),
            "CALL_CHNG_IN_OI": ce.get("changeinOpenInterest", 0),
            "CALL_IV": ce.get("impliedVolatility", 0),
            "CALL_LTP": ce.get("lastPrice", 0),
            "PUT_OI": pe.get("openInterest", 0),
            "PUT_CHNG_IN_OI": pe.get("changeinOpenInterest", 0),
            "PUT_IV": pe.get("impliedVolatility", 0),
            "PUT_LTP": pe.get("lastPrice", 0)
        })
    
    df = pd.DataFrame(records)
    return df.sort_values("STRIKE").reset_index(drop=True) if not df.empty else df

# ---------------- ENHANCED ANALYTICS FUNCTIONS ----------------
def calculate_analytics(df, spot_price=None):
    """
    Enhanced analytics: PCR, max pain (true calc), support/resistance (OI + buildup), IV skew
    """
    if df.empty: return {}
    
    # If spot_price is not provided, estimate it from the data
    if spot_price is None:
        # Estimate spot price as the strike with minimum difference between call and put OI
        df['OI_DIFF'] = abs(df['CALL_OI'] - df['PUT_OI'])
        spot_price = df.loc[df['OI_DIFF'].idxmin(), 'STRIKE'] if not df.empty else 0
    
    # Calculate total OI and deltas
    df['TOTAL_OI'] = df['CALL_OI'] + df['PUT_OI']
    df['OI_RATIO'] = df['PUT_OI'] / df['CALL_OI'].replace(0, 1)
    df['DELTA_CALL'] = df['CALL_OI'] / df['TOTAL_OI'].replace(0, 1)
    df['DELTA_PUT'] = df['PUT_OI'] / df['TOTAL_OI'].replace(0, 1)
    df['IV_DIFF'] = df['CALL_IV'] - df['PUT_IV']
    
    # Basic metrics
    total_put_oi = df['PUT_OI'].sum()
    total_call_oi = df['CALL_OI'].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    
    # --- PCR ---
    overall_pcr = df['PUT_OI'].sum() / df['CALL_OI'].sum() if df['CALL_OI'].sum() > 0 else 0

    # PCR near ATM (±3 strikes, adjust multiplier for stocks vs index)
    atm_strike = df.iloc[(df['STRIKE'] - spot_price).abs().argsort()[:1]]['STRIKE'].values[0] if not df.empty else spot_price
    range_df = df[(df['STRIKE'] >= atm_strike - 3 * 100) & (df['STRIKE'] <= atm_strike + 3 * 100)]
    atm_pcr = range_df['PUT_OI'].sum() / range_df['CALL_OI'].sum() if range_df['CALL_OI'].sum() > 0 else 0

    # --- Max Pain (true loss-based) ---
    strikes = df['STRIKE'].values
    losses = []
    for k in strikes:
        call_loss = ((k - strikes).clip(lower=0) * df['CALL_OI']).sum()
        put_loss = ((strikes - k).clip(lower=0) * df['PUT_OI']).sum()
        losses.append(call_loss + put_loss)
    max_pain = strikes[np.argmin(losses)] if len(losses) > 0 else 0

    # --- Support (PUT OI buildup) ---
    df['PUT_SCORE'] = df['PUT_OI'] * (1 + df['PUT_CHNG_IN_OI'] / (df['PUT_OI'].replace(0, 1) + 1))
    strongest_support = df.loc[df['PUT_SCORE'].idxmax(), 'STRIKE'] if not df.empty else 0

    # --- Resistance (CALL OI buildup) ---
    df['CALL_SCORE'] = df['CALL_OI'] * (1 + df['CALL_CHNG_IN_OI'] / (df['CALL_OI'].replace(0, 1) + 1))
    strongest_resistance = df.loc[df['CALL_SCORE'].idxmax(), 'STRIKE'] if not df.empty else 0

    # --- IV Skew ---
    avg_call_iv = df['CALL_IV'].mean() if not df.empty else 0
    avg_put_iv = df['PUT_IV'].mean() if not df.empty else 0
    iv_skew = avg_call_iv - avg_put_iv

    # --- Volume Ratio ---
    volume_ratio = df['CALL_CHNG_IN_OI'].sum() / df['PUT_CHNG_IN_OI'].sum() if df['PUT_CHNG_IN_OI'].sum() > 0 else 0

    # Sentiment classification
    if pcr > 1.5:
        sentiment = "Strongly Bullish"
        sentiment_score = 2
    elif pcr > 1.2:
        sentiment = "Bullish"
        sentiment_score = 1
    elif pcr < 0.5:
        sentiment = "Strongly Bearish"
        sentiment_score = -2
    elif pcr < 0.8:
        sentiment = "Bearish"
        sentiment_score = -1
    else:
        sentiment = "Neutral"
        sentiment_score = 0
    
    # Top values
    top_3_call = df.nlargest(3, 'CALL_OI')[['STRIKE', 'CALL_OI']] if not df.empty else pd.DataFrame()
    top_3_put = df.nlargest(3, 'PUT_OI')[['STRIKE', 'PUT_OI']] if not df.empty else pd.DataFrame()
    top_3_call_change = df.nlargest(3, 'CALL_CHNG_IN_OI')[['STRIKE', 'CALL_CHNG_IN_OI']] if not df.empty else pd.DataFrame()
    top_3_put_change = df.nlargest(3, 'PUT_CHNG_IN_OI')[['STRIKE', 'PUT_CHNG_IN_OI']] if not df.empty else pd.DataFrame()
    top3_call_iv = df.nlargest(3, 'CALL_IV')[['STRIKE', 'CALL_IV']] if not df.empty else pd.DataFrame()
    top3_put_iv = df.nlargest(3, 'PUT_IV')[['STRIKE', 'PUT_IV']] if not df.empty else pd.DataFrame()
    
    # Predicted values
    predicted_max_pain = np.average(df['STRIKE'], weights=df['TOTAL_OI']) if not df.empty else 0
    predicted_range = (predicted_max_pain - df['STRIKE'].std(), predicted_max_pain + df['STRIKE'].std()) if not df.empty else (0, 0)
    
    return {
        "pcr": pcr,
        "pcr_overall": round(overall_pcr, 2),
        "pcr_atm": round(atm_pcr, 2),
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "strongest_support": strongest_support,
        "strongest_resistance": strongest_resistance,
        "max_pain": max_pain,
        "predicted_max_pain": predicted_max_pain,
        "predicted_range": predicted_range,
        "top_3_call": top_3_call,
        "top_3_put": top_3_put,
        "top_3_call_change": top_3_call_change,
        "top_3_put_change": top_3_put_change,
        "top3_call_iv": top3_call_iv,
        "top3_put_iv": top3_put_iv,
        "volume_ratio": volume_ratio,
        "iv_skew": round(iv_skew, 2),
        "delta_table": df[['STRIKE', 'DELTA_CALL', 'DELTA_PUT']],
        "df": df
    }


def generate_insights(analytics, spot_price):
    """Generate professional market insights based on option chain analytics."""

    insights = []

    # --- PCR Analysis ---
    if analytics['pcr_overall'] > 1.2:
        insights.append(f"Overall PCR = {analytics['pcr_overall']} → Strong PUT writing, bias is **Bullish**.")
    elif analytics['pcr_overall'] < 0.8:
        insights.append(f"Overall PCR = {analytics['pcr_overall']} → Strong CALL writing, bias is **Bearish**.")
    else:
        insights.append(f"Overall PCR = {analytics['pcr_overall']} → Neutral positioning.")

    # ATM PCR for sharper signal
    if analytics['pcr_atm'] > 1.2:
        insights.append(f"ATM PCR = {analytics['pcr_atm']} → Immediate sentiment favors **upside**.")
    elif analytics['pcr_atm'] < 0.8:
        insights.append(f"ATM PCR = {analytics['pcr_atm']} → Immediate sentiment favors **downside**.")

    # --- Support & Resistance ---
    if spot_price > analytics['strongest_resistance']:
        insights.append(f"Spot is **above strongest resistance** ({analytics['strongest_resistance']}) → Possible breakout.")
    elif spot_price < analytics['strongest_support']:
        insights.append(f"Spot is **below strongest support** ({analytics['strongest_support']}) → Possible breakdown.")
    else:
        insights.append(f"Trading between {analytics['strongest_support']} (support) and {analytics['strongest_resistance']} (resistance).")

    # --- Max Pain ---
    if abs(spot_price - analytics['max_pain']) < 150:  # index example, adjust for stocks
        insights.append(f"Spot is near **Max Pain ({analytics['max_pain']})** → Expect consolidation.")
    else:
        insights.append(f"Max Pain at {analytics['max_pain']} is far from spot → Expect volatility.")

    # --- IV Skew ---
    if analytics['iv_skew'] > 2:
        insights.append(f"CALL IV ({analytics['iv_skew']:+}) higher than PUT IV → Traders pricing in **upside risk**.")
    elif analytics['iv_skew'] < -2:
        insights.append(f"PUT IV ({analytics['iv_skew']:+}) higher than CALL IV → Traders pricing in **downside risk**.")
    else:
        insights.append("IV skew is balanced → No major directional bias from volatility.")

    # --- Volume Bias ---
    if analytics['volume_ratio'] > 1.2:
        insights.append(f"CALL volumes dominate (Ratio {analytics['volume_ratio']}) → Short-term bullish sentiment.")
    elif analytics['volume_ratio'] < 0.8:
        insights.append(f"PUT volumes dominate (Ratio {analytics['volume_ratio']}) → Short-term bearish sentiment.")
    else:
        insights.append("Volume distribution is balanced → No short-term bias.")

    return insights


def train_ml_models_classification(df):
    """Train ML models to predict market bias (Bullish/Bearish/Neutral)"""

    if df.empty or len(df) < 10:
        return {"Random Forest": 0, "Logistic Regression": 0}

    # --- Feature Engineering ---
    df['PCR'] = df['PUT_OI'] / (df['CALL_OI'] + 1)
    df['OI_Buildup'] = (df['CALL_CHNG_IN_OI'] + df['PUT_CHNG_IN_OI']) / (df['CALL_OI'] + df['PUT_OI'] + 1)
    df['IV_Skew'] = df['CALL_IV'] - df['PUT_IV']

    features = ['CALL_OI', 'PUT_OI', 'CALL_IV', 'PUT_IV', 'CALL_CHNG_IN_OI', 'PUT_CHNG_IN_OI', 'PCR', 'OI_Buildup', 'IV_Skew']

    # --- Target: Market Bias ---
    conditions = [
        (df['PCR'] > 1.2) & (df['PUT_OI'] > df['CALL_OI']),   # Bullish
        (df['PCR'] < 0.8) & (df['CALL_OI'] > df['PUT_OI']),   # Bearish
    ]
    choices = ['Bullish', 'Bearish']
    df['Bias'] = np.select(conditions, choices, default='Neutral')

    X = df[features].fillna(0)
    y = df['Bias']

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Models ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    return {
        "Random Forest": round(acc_rf, 3),
        "Logistic Regression": round(acc_lr, 3)
    }


# ---------------- MACHINE LEARNING MODELS (REGRESSION) ----------------
def train_ml_models_regression(df):
    """Train multiple ML models for prediction"""
    if df.empty or len(df) < 10:
        return {}, [], []
    
    # Prepare features and target
    X = df[['CALL_OI', 'PUT_OI', 'CALL_CHNG_IN_OI', 'PUT_CHNG_IN_OI', 
            'CALL_IV', 'PUT_IV', 'CALL_LTP', 'PUT_LTP']].fillna(0)
    y = df['STRIKE']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'mae': mae, 'r2': r2, 'scaler': scaler}
    
    # Get best model
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    best_model = results[best_model_name]['model']
    best_scaler = results[best_model_name]['scaler']
    
    # Make predictions on full dataset
    X_full_scaled = best_scaler.transform(X)
    df['ML_PREDICTED_STRIKE'] = best_model.predict(X_full_scaled)
    
    # Find top calls and puts based on ML predictions
    top_calls = df.nlargest(3, 'ML_PREDICTED_STRIKE')['STRIKE'].tolist()
    top_puts = df.nsmallest(3, 'ML_PREDICTED_STRIKE')['STRIKE'].tolist()
    
    return results, top_calls, top_puts

# ---------------- ADVANCED CHARTS ----------------
def create_oi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['CALL_OI'], mode='lines', name='Call OI',
                             line=dict(shape='spline', smoothing=1.3, color='red')))
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['PUT_OI'], mode='lines', name='Put OI',
                             line=dict(shape='spline', smoothing=1.3, color='green')))
    fig.update_layout(
        title="Open Interest Distribution", 
        xaxis_title="Strike Price",
        yaxis_title="Open Interest", 
        height=250, 
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_sentiment_chart(df):
    df['SENTIMENT'] = df['CALL_OI'] - df['PUT_OI']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['STRIKE'], 
        y=df['SENTIMENT'],
        marker_color=['green' if val > 0 else 'red' for val in df['SENTIMENT']]
    ))
    fig.update_layout(
        title="Sentiment (Call OI - Put OI)", 
        xaxis_title="Strike", 
        yaxis_title="Call-Put OI",
        height=250, 
        margin=dict(t=30, b=10, l=10, r=10)
    )
    return fig

def create_iv_comparison_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['STRIKE'], 
        y=df['CALL_IV'], 
        mode='lines', 
        name='Call IV',
        line=dict(color='blue', shape='spline', smoothing=1.3)
    ))
    fig.add_trace(go.Scatter(
        x=df['STRIKE'], 
        y=df['PUT_IV'], 
        mode='lines', 
        name='Put IV',
        line=dict(color='red', shape='spline', smoothing=1.3)
    ))
    fig.update_layout(
        title="Implied Volatility Comparison", 
        xaxis_title="Strike Price",
        yaxis_title="IV (%)", 
        height=250, 
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_ml_prediction_chart(df, analytics, top_calls, top_puts):
    fig = go.Figure()
    
    # Add actual strikes
    fig.add_trace(go.Scatter(
        x=df['STRIKE'], 
        y=df['TOTAL_OI'], 
        mode='lines+markers', 
        name='Total OI',
        line=dict(color='blue', shape='spline', smoothing=1.3)
    ))
    
    # Add ML predicted strikes
    fig.add_trace(go.Scatter(
        x=df['ML_PREDICTED_STRIKE'], 
        y=df['TOTAL_OI'], 
        mode='markers', 
        name='ML Predicted',
        marker=dict(color='orange', size=8, symbol='diamond')
    ))
    
    # Add max pain
    fig.add_vline(
        x=analytics['max_pain'], 
        line_dash="dash", 
        line_color="purple", 
        annotation_text="Max Pain"
    )
    
    # Add top calls
    for strike in top_calls:
        fig.add_vline(
            x=strike, 
            line_dash="dot", 
            line_color="green", 
            annotation_text=f"C {strike}"
        )
    
    # Add top puts
    for strike in top_puts:
        fig.add_vline(
            x=strike, 
            line_dash="dot", 
            line_color="red", 
            annotation_text=f"P {strike}"
        )
    
    fig.update_layout(
        title="ML Predictions & Key Levels", 
        xaxis_title="Strike Price",
        yaxis_title="Total OI", 
        height=300, 
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_model_performance_chart(ml_results):
    models = list(ml_results.keys())
    mae_scores = [ml_results[model]['mae'] for model in models]
    r2_scores = [ml_results[model]['r2'] for model in models]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, 
        y=mae_scores, 
        name='MAE',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=models, 
        y=r2_scores, 
        name='R²',
        marker_color='orange',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="ML Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="MAE (Lower is better)",
        yaxis2=dict(title="R² (Higher is better)", overlaying='y', side='right'),
        barmode='group',
        height=300,
        margin=dict(t=30, b=10, l=10, r=10)
    )
    return fig

# ---------------- EXCEL EXPORT ----------------
def save_to_excel(df, analytics, symbol, ml_results, top_calls, top_puts):
    wb = Workbook()
    
    # Option Chain sheet
    ws1 = wb.active
    ws1.title = "OptionChain"
    
    # Formatting styles
    header_fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
    bold_font = Font(bold=True)
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style='thin'), 
        right=Side(style='thin'),
        top=Side(style='thin'), 
        bottom=Side(style='thin')
    )
    
    # Write option chain data
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
    analytics_data = [
        ["PCR", analytics["pcr"]],
        ["Sentiment", analytics["sentiment"]],
        ["Strongest Support", analytics["strongest_support"]],
        ["Strongest Resistance", analytics["strongest_resistance"]],
        ["Max Pain", analytics["max_pain"]],
        ["Predicted Max Pain", analytics["predicted_max_pain"]],
        ["Volume Ratio (P/C)", analytics["volume_ratio"]],
        ["ML Top Calls", ", ".join(map(str, top_calls))],
        ["ML Top Puts", ", ".join(map(str, top_puts))]
    ]
    
    for i, (label, value) in enumerate(analytics_data, 1):
        ws2.cell(row=i, column=1, value=label).font = bold_font
        ws2.cell(row=i, column=2, value=value)
    
    # ML Results sheet
    ws3 = wb.create_sheet("ML_Results")
    ws3.append(["Model", "MAE", "R²"])
    for model, results in ml_results.items():
        ws3.append([model, results['mae'], results['r2']])
    
    # Save file
    file_path = os.path.join(SAVE_FOLDER, f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    wb.save(file_path)
    return file_path

# ---------------- Styled Option Chain Table ----------------
def styled_option_chain(df):
    max_call = df['CALL_OI'].max()
    max_put = df['PUT_OI'].max()

    def color_rows(row):
        if row['CALL_OI'] > row['PUT_OI']:
            return ['background-color: #4CAF50; color: white;' if col not in ['STRIKE'] else 'font-weight: bold;' for col in row.index]
        elif row['PUT_OI'] > row['CALL_OI']:
            return ['background-color: #f44336; color: white;' if col not in ['STRIKE'] else 'font-weight: bold;' for col in row.index]
        else:
            return ['background-color: #FFEB3B;' if col not in ['STRIKE'] else 'font-weight: bold;' for col in row.index]

    def bar_call(val):
        width = int((val / max_call) * 100) if max_call > 0 else 0
        return f'background: linear-gradient(to right, #4CAF50 {width}%, transparent 0%);'

    def bar_put(val):
        width = int((val / max_put) * 100) if max_put > 0 else 0
        return f'background: linear-gradient(to right, #f44336 {width}%, transparent 0%);'

    styler = df.style.apply(color_rows, axis=1)\
                     .set_properties(subset=['CALL_OI'], **{'background': df['CALL_OI'].apply(bar_call)})\
                     .set_properties(subset=['PUT_OI'], **{'background': df['PUT_OI'].apply(bar_put)})\
                     .set_properties(subset=['STRIKE'], **{'font-weight': 'bold'})\
                     .set_table_styles([{'selector':'th','props':[('font-weight','bold'), ('background-color','#1976D2'),('color','white')]}])\
                     .format({col: "{:,.0f}" for col in df.select_dtypes(include=np.number).columns})
    return styler

# ================= STREAMLIT APP =================
def run_streamlit_app():
    st.set_page_config(page_title="NSE Option Chain Dashboard", layout="wide", initial_sidebar_state="expanded")

    # ---------------- Professional CSS ----------------
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 15px;
        border-left: 4px solid #1f77b4;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        color: #212529;
        font-weight: 700;
    }
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .section-header {
        font-size: 1.4rem;
        color: #1f77b4;
        margin: 20px 0 15px 0;
        font-weight: 600;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- Header ----------------
    st.markdown('<div class="main-header">📊 Advanced NSE Option Chain Analyzer</div>', unsafe_allow_html=True)
    
    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("🔧 Configuration")
        symbol = st.text_input("Symbol", "NIFTY", help="Enter NIFTY, BANKNIFTY, or any stock symbol").upper()
        spot_price = st.number_input("Spot Price", value=0.0, help="Enter the current spot price for accurate analysis")
        
        st.subheader("ML Settings")
        ml_enabled = st.checkbox("Enable ML Predictions", value=True)
        ml_classification = st.checkbox("Enable ML Classification", value=True)
        auto_refresh = st.checkbox("Auto-Refresh (30s)", value=False)
        
        st.subheader("Chart Options")
        show_oi_chart = st.checkbox("Show OI Chart", value=True)
        show_sentiment_chart = st.checkbox("Show Sentiment Chart", value=True)
        show_iv_chart = st.checkbox("Show IV Chart", value=True)
        
        st.info("ℹ️ Data is fetched in real-time from NSE India")
    
    # ---------------- Main Content ----------------
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    fetch_clicked = st.button("🚀 Fetch & Analyze Data", type="primary", use_container_width=True)
    
    if fetch_clicked:
        with st.spinner("🔄 Fetching real-time data from NSE..."):
            session = get_nse_session()
            data = fetch_option_chain(symbol, session)

        if data:
            df = parse_data(symbol, data)
            if not df.empty:
                # Estimate spot price if not provided
                if spot_price == 0:
                    df['OI_DIFF'] = abs(df['CALL_OI'] - df['PUT_OI'])
                    spot_price = df.loc[df['OI_DIFF'].idxmin(), 'STRIKE'] if not df.empty else 0
                    st.info(f"Estimated spot price: {spot_price}")
                
                analytics = calculate_analytics(df, spot_price)
                
                # Machine Learning
                if ml_enabled:
                    with st.spinner("🤖 Training ML models..."):
                        ml_results, top_calls, top_puts = train_ml_models_regression(df)
                else:
                    ml_results, top_calls, top_puts = {}, [], []
                
                # ML Classification
                ml_classification_results = {}
                if ml_classification:
                    with st.spinner("🤖 Training ML classification models..."):
                        ml_classification_results = train_ml_models_classification(df)
                
                # ---------------- Display Results ----------------
                st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)
                
                # Metrics in columns
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">PCR</div>
                        <div class="metric-value">{analytics['pcr']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Sentiment</div>
                        <div class="metric-value">{analytics['sentiment']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Support</div>
                        <div class="metric-value">{analytics['strongest_support']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Resistance</div>
                        <div class="metric-value">{analytics['strongest_resistance']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Max Pain</div>
                        <div class="metric-value">{analytics['max_pain']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ---------------- Market Insights ----------------
                st.markdown('<div class="section-header">💡 Market Insights</div>', unsafe_allow_html=True)
                
                insights = generate_insights(analytics, spot_price)
                for insight in insights:
                    st.info(insight)
                
                # ---------------- Charts ----------------
                st.markdown('<div class="section-header">📊 Visual Analytics</div>', unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["Option Chain", "Analytics Charts", "ML Results", "Export"])
                
                with tab1:
                    st.subheader("Styled Option Chain")
                    st.dataframe(styled_option_chain(df), use_container_width=True)
                
                with tab2:
                    if show_oi_chart:
                        st.plotly_chart(create_oi_chart(df), use_container_width=True)
                    if show_sentiment_chart:
                        st.plotly_chart(create_sentiment_chart(df), use_container_width=True)
                    if show_iv_chart:
                        st.plotly_chart(create_iv_comparison_chart(df), use_container_width=True)
                
                with tab3:
                    if ml_enabled and ml_results:
                        st.subheader("Machine Learning Predictions")
                        st.plotly_chart(create_ml_prediction_chart(df, analytics, top_calls, top_puts), use_container_width=True)
                        st.plotly_chart(create_model_performance_chart(ml_results), use_container_width=True)
                        
                        st.subheader("Top ML Predictions")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Top Calls**")
                            for strike in top_calls:
                                st.write(f"- {strike}")
                        with col2:
                            st.markdown("**Top Puts**")
                            for strike in top_puts:
                                st.write(f"- {strike}")
                    
                    if ml_classification and ml_classification_results:
                        st.subheader("ML Classification Results")
                        st.write("Model Accuracy Scores:")
                        for model, score in ml_classification_results.items():
                            st.write(f"- {model}: {score:.3f}")
                
                with tab4:
                    st.subheader("Export Data")
                    if st.button("💾 Export to Excel", use_container_width=True):
                        file_path = save_to_excel(df, analytics, symbol, ml_results, top_calls, top_puts)
                        st.success(f"✅ Data exported to: {file_path}")
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Excel File",
                                data=f,
                                file_name=os.path.basename(file_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
            else:
                st.error("❌ No data available for the selected symbol. Please try again.")
        else:
            st.error("❌ Failed to fetch data. Please check the symbol and try again.")
    else:
        st.info("👈 Enter a symbol and click 'Fetch & Analyze Data' to begin")

# ================= MAIN =================
if __name__ == "__main__":
    run_streamlit_app()