# utils/charting.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def create_oi_chart(df, line_shape='spline'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['CALL_OI'], mode='lines', name='Call OI', 
                             line=dict(shape=line_shape, width=3, color='#FF6B6B')))
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['PUT_OI'], mode='lines', name='Put OI', 
                             line=dict(shape=line_shape, width=3, color='#4ECDC4')))
    fig.update_layout(
        title="Open Interest Distribution",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig

def create_sentiment_chart(df, line_shape='spline'):
    df_local = df.copy()
    df_local['SENT'] = df_local['CALL_OI'] - df_local['PUT_OI']
    fig = go.Figure([go.Bar(x=df_local['STRIKE'], y=df_local['SENT'], 
                           marker_color=np.where(df_local['SENT'] > 0, '#FF6B6B', '#4ECDC4'))])
    fig.update_layout(
        title="Sentiment (Call OI - Put OI)",
        xaxis_title="Strike Price",
        yaxis_title="Call-Put OI Difference",
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig

def create_iv_comparison_chart(df, line_shape='spline'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['CALL_IV'], name='Call IV', 
                             line=dict(shape=line_shape, width=3, color='#FF6B6B')))
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['PUT_IV'], name='Put IV', 
                             line=dict(shape=line_shape, width=3, color='#4ECDC4')))
    fig.update_layout(
        title="Implied Volatility (Call vs Put)",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%)",
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig

def create_volatility_surface_chart(df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Call IV Surface", "Put IV Surface"))
    
    # Call IV surface
    fig.add_trace(
        go.Scatter(x=df['STRIKE'], y=df['CALL_IV'], mode='lines+markers', 
                  name='Call IV', line=dict(width=3, color='#FF6B6B')),
        row=1, col=1
    )
    
    # Put IV surface
    fig.add_trace(
        go.Scatter(x=df['STRIKE'], y=df['PUT_IV'], mode='lines+markers', 
                  name='Put IV', line=dict(width=3, color='#4ECDC4')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Strike Price", row=1, col=1)
    fig.update_xaxes(title_text="Strike Price", row=1, col=2)
    fig.update_yaxes(title_text="Implied Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="Implied Volatility (%)", row=1, col=2)
    
    fig.update_layout(
        title="Volatility Surface Analysis",
        height=400,
        margin=dict(t=80, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_ml_prediction_chart(df, analytics, top_calls, top_puts, line_shape='spline'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['TOTAL_OI'], mode='lines+markers', 
                             name='Total OI', line=dict(shape=line_shape, width=3, color='#45B7D1')))
    
    if 'ML_PREDICTED_VALUE' in df.columns:
        fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['ML_PREDICTED_VALUE'], mode='markers', 
                                name='ML Predicted', marker=dict(size=8, color='#FBE555')))
    
    fig.add_vline(x=analytics.get('max_pain', None), line_dash='dash', line_color='#964B00', 
                  annotation_text='Max Pain', annotation_position="top right")
    
    for s in top_calls:
        fig.add_vline(x=s, line_dash='dot', line_color='green', annotation_text=f'Call {s}')
    
    for s in top_puts:
        fig.add_vline(x=s, line_dash='dot', line_color='red', annotation_text=f'Put {s}')
    
    fig.update_layout(
        title="ML Predicted Strikes & OI",
        xaxis_title="Strike Price",
        yaxis_title="Total OI / ML Prediction",
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    return fig

def create_model_performance_chart(ml_results):
    if not ml_results:
        return go.Figure()
    
    models = list(ml_results.keys())
    mae = [ml_results[m]['mae'] for m in models]
    r2 = [ml_results[m]['r2'] for m in models]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar for MAE
    fig.add_trace(
        go.Bar(
            x=models,
            y=mae,
            name='MAE',
            marker_color='orange'
        ),
        secondary_y=False
    )
    
    # Line for R2
    fig.add_trace(
        go.Scatter(
            x=models,
            y=r2,
            name='R²',
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    # Layout with dual y-axis
    fig.update_layout(
        title="Regression Model Performance",
        xaxis_title="Model",
        yaxis=dict(title="MAE", side='left', showgrid=False),
        yaxis2=dict(title="R²", side='right', showgrid=False, range=[-1, 1]),
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig