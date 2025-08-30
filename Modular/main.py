# main.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import base64
from io import BytesIO

# Import modular components
from config import SAVE_FOLDER, INDICES
from utils.nse_session import get_nse_session
from utils.data_fetcher import fetch_all_option_chains
from utils.data_parser import parse_data
from utils.analytics import calculate_analytics
from utils.ml_models import train_ml_models_regression, train_ml_models_classification
from utils.charting import (
    create_oi_chart, create_sentiment_chart, create_iv_comparison_chart,
    create_volatility_surface_chart, create_ml_prediction_chart, create_model_performance_chart
)
from utils.excel_export import create_excel_export
from strategies.options_strategies import calculate_greeks_for_chain, suggest_strategies
from utils.technical_indicators import calculate_technical_indicators

# Page configuration
st.set_page_config(
    page_title="NSE Option Chain Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    with open("templates/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main application
def main():
    load_css()
    
    st.title("📊 NSE Option Chain Analyzer")
    st.markdown("Advanced analytics and machine learning for NSE option chain data")
    
    # Sidebar
    st.sidebar.header("Configuration")
    symbol = st.sidebar.selectbox("Select Symbol", INDICES, index=0)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {}
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data") or (auto_refresh and time.time() - st.session_state.last_refresh > refresh_interval):
        with st.spinner("Fetching option chain data..."):
            session = get_nse_session()
            data = fetch_all_option_chains([symbol], session)
            st.session_state.data = data
            
            if data and symbol in data and data[symbol]:
                df = parse_data(symbol, data[symbol])
                if not df.empty:
                    analytics = calculate_analytics(df)
                    st.session_state.analytics = analytics
                    
                    # ML Analysis
                    ml_results, top_calls, top_puts, feature_importance = train_ml_models_regression(analytics['df'])
                    st.session_state.ml_results = ml_results
                    st.session_state.top_calls = top_calls
                    st.session_state.top_puts = top_puts
                    st.session_state.feature_importance = feature_importance
                    
                    # Classification analysis
                    cls_results, consensus, cls_details = train_ml_models_classification(analytics['df'])
                    st.session_state.cls_results = cls_results
                    st.session_state.consensus = consensus
                    st.session_state.cls_details = cls_details
                    
                    # Technical indicators
                    tech_df = calculate_technical_indicators(analytics['df'])
                    st.session_state.tech_df = tech_df
                    
                    # Greeks calculation
                    greeks_df = calculate_greeks_for_chain(analytics['df'], analytics.get('spot_price', 0))
                    st.session_state.greeks_df = greeks_df
                    
                    # Strategy suggestions
                    strategies = suggest_strategies(analytics, greeks_df)
                    st.session_state.strategies = strategies
                
                st.session_state.last_refresh = time.time()
                st.sidebar.success("Data refreshed successfully!")
            else:
                st.sidebar.error("Failed to fetch data. Please try again.")
    
    # Display data if available
    if st.session_state.data and symbol in st.session_state.data and st.session_state.analytics:
        analytics = st.session_state.analytics
        df = analytics['df']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"₹{analytics.get('spot_price', 'N/A')}")
            st.metric("PCR", analytics.get('pcr', 'N/A'))
        with col2:
            st.metric("Max Pain", f"₹{analytics.get('max_pain', 'N/A')}")
            st.metric("PCR (ATM)", analytics.get('pcr_atm', 'N/A'))
        with col3:
            st.metric("Support", f"₹{analytics.get('support', 'N/A')}")
            st.metric("Resistance", f"₹{analytics.get('resistance', 'N/A')}")
        with col4:
            st.metric("Direction", analytics.get('direction', 'N/A'))
            st.metric("30D Expected Move", f"₹{analytics.get('expected_move_30d', 'N/A')}")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Charts", "🤖 ML Analysis", "📊 Data", "🔄 Technical", "🧮 Greeks", "💡 Strategies"
        ])
        
        with tab1:
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_oi_chart(df), use_container_width=True)
            with col2:
                st.plotly_chart(create_sentiment_chart(df), use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(create_iv_comparison_chart(df), use_container_width=True)
            with col4:
                st.plotly_chart(create_volatility_surface_chart(df), use_container_width=True)
        
        with tab2:
            # ML Analysis
            if st.session_state.ml_results:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_ml_prediction_chart(
                        df, analytics, 
                        st.session_state.top_calls, 
                        st.session_state.top_puts
                    ), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_model_performance_chart(
                        st.session_state.ml_results
                    ), use_container_width=True)
                
                st.subheader("ML Recommendations")
                st.write(f"**Top Call Strikes:** {', '.join(map(str, st.session_state.top_calls))}")
                st.write(f"**Top Put Strikes:** {', '.join(map(str, st.session_state.top_puts))}")
                
                st.subheader("Feature Importance")
                feature_df = pd.DataFrame.from_dict(
                    st.session_state.feature_importance, 
                    orient='index', 
                    columns=['Importance']
                ).sort_values('Importance', ascending=False)
                st.dataframe(feature_df.style.format({'Importance': '{:.4f}'}))
            
            # Classification results
            if hasattr(st.session_state, 'cls_results'):
                st.subheader("Classification Analysis")
                st.write(f"**Market Consensus:** {st.session_state.consensus}")
                st.write(f"**Random Forest Accuracy:** {st.session_state.cls_results.get('RF', 0)}")
                st.write(f"**Logistic Regression Accuracy:** {st.session_state.cls_results.get('LR', 0)}")
        
        with tab3:
            # Data table
            st.dataframe(df.style.format({
                'CALL_OI': '{:,.0f}',
                'PUT_OI': '{:,.0f}',
                'CALL_IV': '{:.2f}',
                'PUT_IV': '{:.2f}',
                'CALL_LTP': '{:.2f}',
                'PUT_LTP': '{:.2f}'
            }))
            
            # Export to Excel
            if st.button("📥 Export to Excel"):
                buffer = create_excel_export(
                    df, analytics, symbol, 
                    st.session_state.ml_results,
                    st.session_state.top_calls,
                    st.session_state.top_puts,
                    get_nse_session()
                )
                
                st.download_button(
                    label="Download Excel File",
                    data=buffer,
                    file_name=f"option_chain_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with tab4:
            # Technical indicators
            if hasattr(st.session_state, 'tech_df'):
                st.dataframe(st.session_state.tech_df.style.format({
                    'SMA_CALL_OI': '{:,.0f}',
                    'SMA_PUT_OI': '{:,.0f}',
                    'EMA_CALL_OI': '{:,.0f}',
                    'EMA_PUT_OI': '{:,.0f}'
                }))
        
        with tab5:
            # Greeks
            if hasattr(st.session_state, 'greeks_df'):
                st.dataframe(st.session_state.greeks_df.style.format({
                    'CALL_DELTA': '{:.4f}',
                    'PUT_DELTA': '{:.4f}',
                    'CALL_GAMMA': '{:.6f}',
                    'PUT_GAMMA': '{:.6f}',
                    'CALL_THETA': '{:.4f}',
                    'PUT_THETA': '{:.4f}',
                    'CALL_VEGA': '{:.4f}',
                    'PUT_VEGA': '{:.4f}'
                }))
        
        with tab6:
            # Strategy suggestions
            if hasattr(st.session_state, 'strategies'):
                for strategy in st.session_state.strategies:
                    with st.expander(f"📋 {strategy['name']}"):
                        st.write(f"**Description:** {strategy['description']}")
                        st.write(f"**Risk:** {strategy['risk']}")
                        st.write(f"**Reward:** {strategy['reward']}")
                        st.write(f"**Breakeven:** {strategy['breakeven']}")
    
    else:
        st.info("Click 'Refresh Data' to load option chain information")

if __name__ == "__main__":
    main()