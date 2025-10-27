import streamlit as st
import duckdb, pandas as pd, numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import stock analysis components
try:
    from stock_summarizer import get_stock_analysis
    from stock_analysis_models import StockAnalysisResponse
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Stock analysis not available: {e}")
    ANALYSIS_AVAILABLE = False

import os
from pathlib import Path

# Determine the correct database path
if os.path.exists("data/duckdb/investo.duckdb"):
    DB_PATH = "data/duckdb/investo.duckdb"
elif os.path.exists(".venv/data/duckdb/investo.duckdb"):
    DB_PATH = ".venv/data/duckdb/investo.duckdb"
else:
    # Fallback to relative path from script location
    script_dir = Path(__file__).parent
    DB_PATH = str(script_dir / "data" / "duckdb" / "investo.duckdb")

st.set_page_config(page_title="💰 Investo", page_icon="💰", layout="wide")

# Get available tickers (needed for both tabs)
@st.cache_data(ttl=60)
def list_tickers():
    con = duckdb.connect(DB_PATH, read_only=True)
    # fall back to prices if predictions empty
    dfp = con.execute("SELECT DISTINCT ticker FROM predictions").df()
    if dfp.empty:
        dfp = con.execute("SELECT DISTINCT ticker FROM prices").df()
    con.close()
    return sorted(dfp["ticker"].tolist())

@st.cache_data(ttl=60)
def current_price_for(ticker: str) -> float:
    con = duckdb.connect(DB_PATH, read_only=True)
    q = """
    WITH last AS (
      SELECT ticker, MAX(date) AS d FROM prices GROUP BY ticker
    )
    SELECT p.adj_close AS current_price
    FROM prices p
    JOIN last l ON p.ticker = l.ticker AND p.date = l.d
    WHERE p.ticker = ?
    """
    res = con.execute(q, [ticker]).df()
    con.close()
    return float(res["current_price"].iloc[0]) if not res.empty else np.nan

@st.cache_data(ttl=60)
def predicted_price_for(ticker: str, years: int, months: int = 0) -> float:
    """
    Pick the latest as_of_ts run for this ticker, then choose the pred_date
    closest to (today + years + months). If we overshoot/undershoot, take nearest.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    # target future date (use business days in predictions, but here pick nearest calendar date)
    total_months = years * 12 + months
    future_target = pd.to_datetime(date.today()) + pd.DateOffset(months=total_months)

    q = """
    WITH latest AS (
      SELECT ticker, MAX(as_of_ts) AS asof
      FROM predictions
      WHERE ticker = ?
      GROUP BY ticker
    ), preds AS (
      SELECT p.pred_date, p.pred_price
      FROM predictions p
      JOIN latest l ON p.ticker = ? AND p.ticker = l.ticker AND p.as_of_ts = l.asof
      ORDER BY p.pred_date
    )
    SELECT pred_date, pred_price
    FROM preds
    """
    df = con.execute(q, [ticker, ticker]).df()
    con.close()
    if df.empty:
        return np.nan

    df["pred_date"] = pd.to_datetime(df["pred_date"])
    # nearest date on/after target; if none, take last available
    aft = df[df["pred_date"] >= future_target]
    if not aft.empty:
        return float(aft.iloc[0]["pred_price"])
    return float(df.iloc[-1]["pred_price"])

@st.cache_data(ttl=60)
def get_historical_data(ticker: str, years_back: int = 2) -> pd.DataFrame:
    """Get historical price data for plotting"""
    con = duckdb.connect(DB_PATH, read_only=True)
    cutoff_date = date.today() - timedelta(days=years_back * 365)
    
    q = """
    SELECT date, adj_close
    FROM prices
    WHERE ticker = ? AND date >= ?
    ORDER BY date
    """
    df = con.execute(q, [ticker, cutoff_date]).df()
    con.close()
    return df

@st.cache_data(ttl=60)
def get_predicted_data(ticker: str) -> pd.DataFrame:
    """Get predicted price data for plotting"""
    con = duckdb.connect(DB_PATH, read_only=True)
    
    q = """
    WITH latest AS (
      SELECT ticker, MAX(as_of_ts) AS asof
      FROM predictions
      WHERE ticker = ?
      GROUP BY ticker
    )
    SELECT p.pred_date AS date, p.pred_price AS adj_close
    FROM predictions p
    JOIN latest l ON p.ticker = ? AND p.ticker = l.ticker AND p.as_of_ts = l.asof
    ORDER BY p.pred_date
    """
    df = con.execute(q, [ticker, ticker]).df()
    con.close()
    return df

def compute_return_row(ticker: str, amount: float, years: int, months: int = 0):
    cp = current_price_for(ticker)
    pp = predicted_price_for(ticker, years, months)
    if np.isnan(cp) or np.isnan(pp) or cp <= 0:
        return {
            "ticker": ticker, "current_price": np.nan, "pred_price": np.nan,
            "shares": np.nan, "future_value": np.nan, "abs_return": np.nan, "return_pct": np.nan
        }
    shares = amount / cp
    fv = shares * pp
    absret = fv - amount
    pct = absret / amount
    return {
        "ticker": ticker,
        "current_price": cp,
        "pred_price": pp,
        "shares": shares,
        "future_value": fv,
        "abs_return": absret,
        "return_pct": pct,
    }

def compute_for_many(tickers, amount, years, months=0, exclude=None):
    rows = []
    for t in tickers:
        if exclude and t == exclude:
            continue
        rows.append(compute_return_row(t, amount, years, months))
    df = pd.DataFrame(rows)
    # nicer formatting/calcs
    if not df.empty:
        df["return_pct"] = (df["return_pct"] * 100.0).round(2)
        for col in ["current_price","pred_price","future_value","abs_return"]:
            df[col] = df[col].round(2)
        df["shares"] = df["shares"].round(4)
        df = df.sort_values("return_pct", ascending=False, na_position="last").reset_index(drop=True)
    return df

def create_price_chart(ticker: str, amount: float, years: int, months: int = 0):
    """Create a line chart showing historical and predicted prices"""
    # Get historical data
    hist_df = get_historical_data(ticker, years_back=2)
    pred_df = get_predicted_data(ticker)
    
    if hist_df.empty or pred_df.empty:
        return None
    
    # Convert dates
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    # Create the plot
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['adj_close'],
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Calculate if prediction shows profit or loss
    current_price = current_price_for(ticker)
    predicted_price = predicted_price_for(ticker, years, months)
    is_profitable = predicted_price > current_price
    
    # Add predicted data with color based on profit/loss
    pred_color = 'green' if is_profitable else 'red'
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df['adj_close'],
        mode='lines',
        name='Predicted',
        line=dict(color=pred_color, width=2, dash='dash')
    ))
    
    # Add investment value line
    shares = amount / current_price if current_price > 0 else 0
    investment_value = shares * pred_df['adj_close']
    
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=investment_value,
        mode='lines',
        name=f'Investment Value (${amount:,.0f})',
        line=dict(color='purple', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Price History & Predictions',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        yaxis2=dict(
            title='Investment Value ($)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500
    )
    
    return fig

# Get available tickers dynamically from database
try:
    from ticker_dag import get_available_tickers
    avail = get_available_tickers()
except ImportError:
    # Fallback to original method
    avail = list_tickers()

if not avail:
    st.warning("No tickers found in predictions/prices. Run ingestion → training → prediction first.")
    st.stop()

# Create tabs for different pages
tab1, tab2, tab3 = st.tabs(["💰 Investment Calculator", "🤖 AI Analysis", "➕ Add Ticker"])

with tab1:
    st.title("💰 Investo - Investment Calculator")
    
    # Input section
    st.subheader("Investment Calculator")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sel = st.selectbox("Select Stock", options=avail, index=0, key="investment_ticker")
        # Store selected ticker in session state for AI analysis tab
        st.session_state.selected_ticker = sel
        
    with col2:
        amt = st.number_input("Investment Amount ($)", min_value=100.0, value=10000.0, step=100.0)
        
    with col3:
        yrs = st.slider("Years", min_value=0, max_value=10, value=5)
        
    with col4:
        months = st.slider("Months", min_value=0, max_value=11, value=0)

    # Calculate button
    calculate_clicked = st.button("Calculate Investment Returns", type="primary")

    # Main layout: 70% chart, 30% calculation results
    if calculate_clicked or 'last_result' in st.session_state:
        # Create the main layout with 70/30 split
        chart_col, results_col = st.columns([0.7, 0.3])
        
        with chart_col:
            st.subheader("Price History & Predictions")
            chart = create_price_chart(sel, amt, yrs, months)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("Unable to generate chart. Please check if historical and prediction data are available.")
        
        with results_col:
            st.subheader("Investment Returns")
            # Calculate and display results
            if calculate_clicked:
                result = compute_return_row(sel, amt, yrs, months)
                st.session_state.last_result = result
            elif 'last_result' in st.session_state:
                result = st.session_state.last_result
            else:
                result = None
            
            if result and not np.isnan(result["current_price"]):
                # Display results in a card format
                st.metric(
                    label="Current Price",
                    value=f"${result['current_price']:.2f}"
                )
                
                st.metric(
                    label="Predicted Price",
                    value=f"${result['pred_price']:.2f}",
                    delta=f"{((result['pred_price'] - result['current_price']) / result['current_price'] * 100):.1f}%"
                )
                
                st.metric(
                    label="Future Value",
                    value=f"${result['future_value']:,.2f}"
                )
                
                st.metric(
                    label="Total Return",
                    value=f"${result['abs_return']:,.2f}",
                    delta=f"{result['return_pct']:.2f}%"
                )
                
                # Additional details in expander
                with st.expander("Investment Details"):
                    total_period = f"{yrs} years" + (f" {months} months" if months > 0 else "")
                    total_months = yrs * 12 + months
                    
                    st.write(f"**Shares Purchased:** {result['shares']:.4f}")
                    st.write(f"**Investment Amount:** ${amt:,.2f}")
                    st.write(f"**Investment Period:** {total_period}")
                    if total_months > 0:
                        st.write(f"**Annualized Return:** {(result['return_pct'] / (total_months / 12)):.2f}%")
                
                # Show profit/loss indicator
                if result['abs_return'] > 0:
                    st.success(f"🎉 Expected gain: ${result['abs_return']:,.2f}")
                else:
                    st.error(f"⚠️ Expected loss: ${abs(result['abs_return']):,.2f}")
            else:
                st.info("Click 'Calculate Investment Returns' to see results here.")
    else:
        # Show chart only when no calculation has been done
        st.subheader("Price History & Predictions")
        chart = create_price_chart(sel, amt, yrs, months)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("Unable to generate chart. Please check if historical and prediction data are available.")

    # Other Stocks Comparison Table
    st.markdown("---")
    st.subheader("📊 Expected Returns for Other Stocks")
    
    other_pool = [t for t in avail if t != sel]
    if other_pool:
        base_df = compute_for_many(other_pool, amt, yrs, months)
        if not base_df.empty:
            # Format the dataframe for better display
            display_df = base_df[["ticker", "current_price", "pred_price", "future_value", "abs_return", "return_pct"]].copy()
            display_df.columns = ["Stock", "Current Price", "Predicted Price", "Future Value", "Total Return ($)", "Return (%)"]
            
            # Create a custom formatter for the Return (%) column that includes styling
            def format_return_with_color(val):
                # Handle formatted strings by extracting the numeric value
                if isinstance(val, str):
                    if val == "N/A":
                        return 'background-color: #6c757d; color: white; font-weight: bold'
                    # Extract numeric value from formatted string like "5.25%"
                    try:
                        numeric_val = float(val.replace('%', ''))
                    except:
                        return 'background-color: #6c757d; color: white; font-weight: bold'
                else:
                    numeric_val = val
                
                if pd.isna(numeric_val):
                    return 'background-color: #6c757d; color: white; font-weight: bold'
                elif numeric_val > 0:
                    return 'background-color: #28a745; color: white; font-weight: bold'
                elif numeric_val < 0:
                    return 'background-color: #dc3545; color: white; font-weight: bold'
                else:
                    return 'background-color: #6c757d; color: white; font-weight: bold'
            
            # Format the dataframe for better presentation
            display_df_formatted = display_df.copy()
            
            # Format currency columns
            display_df_formatted['Current Price'] = display_df_formatted['Current Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            display_df_formatted['Predicted Price'] = display_df_formatted['Predicted Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            display_df_formatted['Future Value'] = display_df_formatted['Future Value'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
            display_df_formatted['Total Return ($)'] = display_df_formatted['Total Return ($)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
            
            # Format percentage column
            display_df_formatted['Return (%)'] = display_df_formatted['Return (%)'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            
            # Apply styling with better formatting
            styled_df = (display_df_formatted.style
                        .map(format_return_with_color, subset=['Return (%)'])
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([
                            {'selector': 'thead th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold')]},
                            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8f9fa')]},
                            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
                            {'selector': 'td', 'props': [('padding', '8px'), ('border', '1px solid #dee2e6')]}
                        ]))
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No comparison data available.")
    else:
        st.info("No other stocks available for comparison.")

    st.caption("Returns use the most recent prediction run per ticker and the prediction closest to today + years + months.")

# AI Analysis Tab
with tab2:
    st.title("💰 Investo - AI Stock Analysis")
    
    if not ANALYSIS_AVAILABLE:
        st.warning("Stock analysis is not available. Please check your OpenAI API key configuration.")
        st.stop()
    
    # Check if we have a ticker from the Investment Calculator tab
    if 'selected_ticker' in st.session_state and st.session_state.selected_ticker:
        st.info(f"📈 Analyzing the same stock from Investment Calculator: **{st.session_state.selected_ticker}**")
        default_ticker = st.session_state.selected_ticker
        default_index = avail.index(default_ticker) if default_ticker in avail else 0
    else:
        st.info("💡 Select a stock below for AI analysis, or go to the Investment Calculator tab first to analyze the same stock.")
        default_ticker = avail[0] if avail else 'AAPL'
        default_index = 0
    
    # Always show dropdown for flexibility
    selected_ticker = st.selectbox(
        "Select Stock for AI Analysis", 
        options=avail, 
        index=default_index,
        key="ai_analysis_ticker",
        help="Choose any stock for AI-powered analysis"
    )
    
    # Update session state
    st.session_state.ai_selected_ticker = selected_ticker
    
    analyze_clicked = st.button("🤖 Analyze Stock with AI", type="primary")
    
    if analyze_clicked:
        with st.spinner("Analyzing stock with AI..."):
            try:
                analysis = get_stock_analysis(selected_ticker)
                
                # Header with stock info
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h2 style="color: white; margin: 0; text-align: center;">🤖 AI Analysis: {selected_ticker}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Key Metrics Dashboard
                st.markdown("### 📊 Investment Dashboard")
                
                # Recommendation Card
                rec_color = "🟢" if "buy" in analysis.recommendations.primary_recommendation.lower() else "🔴" if "sell" in analysis.recommendations.primary_recommendation.lower() else "🟡"
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; text-align: center;">
                        <h4 style="margin: 0; color: #28a745;">{rec_color} Recommendation</h4>
                        <h3 style="margin: 5px 0; color: #333;">{analysis.recommendations.primary_recommendation}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    risk_color = "#dc3545" if analysis.risk_assessment.risk_score > 7 else "#ffc107" if analysis.risk_assessment.risk_score > 4 else "#28a745"
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {risk_color}; text-align: center;">
                        <h4 style="margin: 0; color: {risk_color};">⚠️ Risk Level</h4>
                        <h3 style="margin: 5px 0; color: #333;">{analysis.risk_assessment.risk_score}/10</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    sentiment_color = "#28a745" if "bullish" in analysis.market_outlook.market_sentiment.lower() else "#dc3545" if "bearish" in analysis.market_outlook.market_sentiment.lower() else "#ffc107"
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {sentiment_color}; text-align: center;">
                        <h4 style="margin: 0; color: {sentiment_color};">📈 Market Sentiment</h4>
                        <h3 style="margin: 5px 0; color: #333;">{analysis.market_outlook.market_sentiment}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; text-align: center;">
                        <h4 style="margin: 0; color: #17a2b8;">🎯 Confidence</h4>
                        <h3 style="margin: 5px 0; color: #333;">{analysis.confidence_score}/10</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Executive Summary Card
                st.markdown("### 💡 Executive Summary")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;">
                    <p style="margin: 0; font-size: 16px; line-height: 1.5;">{analysis.summary}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Main Analysis Cards
                st.markdown("### 📋 Detailed Analysis")
                
                # Investment Recommendation Card
                st.markdown("#### 🎯 Investment Recommendation")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #e8f5e8; padding: 20px; border-radius: 8px; border: 1px solid #28a745;">
                        <h4 style="color: #28a745; margin-top: 0;">Primary Action</h4>
                        <h3 style="color: #333; margin: 10px 0;">{analysis.recommendations.primary_recommendation}</h3>
                        <p><strong>Portfolio Allocation:</strong> {analysis.recommendations.portfolio_allocation}</p>
                        {f'<p><strong>Target Price:</strong> {analysis.recommendations.target_price}</p>' if analysis.recommendations.target_price else ''}
                        {f'<p><strong>Stop Loss:</strong> {analysis.recommendations.stop_loss_suggestion}</p>' if analysis.recommendations.stop_loss_suggestion else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; padding: 20px; border-radius: 8px; border: 1px solid #ffc107;">
                        <h4 style="color: #856404; margin-top: 0;">Risk Assessment</h4>
                        <h3 style="color: #333; margin: 10px 0;">{analysis.risk_assessment.risk_score}/10</h3>
                        <p><strong>Level:</strong> {analysis.risk_assessment.risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Technical & Fundamental Analysis
                st.markdown("#### 📈 Technical & Fundamental Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #d1ecf1; padding: 20px; border-radius: 8px; border: 1px solid #17a2b8;">
                        <h4 style="color: #0c5460; margin-top: 0;">📊 Technical Analysis</h4>
                        <p><strong>Trend:</strong> {analysis.performance.current_trend}</p>
                        <p><strong>Momentum:</strong> {analysis.performance.momentum}</p>
                        <p><strong>Volatility:</strong> {analysis.performance.volatility_level}</p>
                        <p><strong>Key Indicators:</strong></p>
                        <ul>
                            {''.join([f'<li>{indicator}</li>' for indicator in analysis.technical_analysis.key_indicators[:2]])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; padding: 20px; border-radius: 8px; border: 1px solid #dc3545;">
                        <h4 style="color: #721c24; margin-top: 0;">🏢 Fundamental Analysis</h4>
                        <p><strong>Company Health:</strong> {analysis.fundamental_analysis.company_health}</p>
                        <p><strong>Growth Prospects:</strong> {analysis.fundamental_analysis.growth_prospects}</p>
                        <p><strong>Competitive Position:</strong> {analysis.fundamental_analysis.competitive_position}</p>
                        <p><strong>Valuation:</strong> {analysis.fundamental_analysis.valuation_assessment}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Timing & Market Outlook
                st.markdown("#### ⏰ Investment Timing & Market Outlook")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 20px; border-radius: 8px; border: 1px solid #28a745;">
                        <h4 style="color: #155724; margin-top: 0;">⏰ Investment Timing</h4>
                        <p><strong>Entry Time:</strong> {analysis.investment_timing.optimal_entry_time}</p>
                        <p><strong>Exit Time:</strong> {analysis.investment_timing.optimal_exit_time}</p>
                        <p><strong>Holding Period:</strong> {analysis.investment_timing.holding_period}</p>
                        <p><strong>Market Conditions:</strong> {analysis.investment_timing.market_conditions}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color: #cce5ff; padding: 20px; border-radius: 8px; border: 1px solid #007bff;">
                        <h4 style="color: #004085; margin-top: 0;">🌍 Market Outlook</h4>
                        <p><strong>Sector Outlook:</strong> {analysis.market_outlook.sector_outlook}</p>
                        <p><strong>Sentiment:</strong> {analysis.market_outlook.market_sentiment}</p>
                        <p><strong>External Factors:</strong></p>
                        <ul>
                            {''.join([f'<li>{factor}</li>' for factor in analysis.market_outlook.external_factors[:2]])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Alternative Strategies
                if analysis.recommendations.alternative_strategies:
                    st.markdown("#### 🔄 Alternative Strategies")
                    strategies_html = ""
                    for i, strategy in enumerate(analysis.recommendations.alternative_strategies[:3], 1):
                        strategies_html += f'<div style="background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #6c757d;"><strong>{i}.</strong> {strategy}</div>'
                    
                    st.markdown(strategies_html, unsafe_allow_html=True)
                
                # Key Insights
                if analysis.key_insights:
                    st.markdown("#### 💡 Key Insights")
                    insights_html = ""
                    for insight in analysis.key_insights[:3]:
                        insights_html += f'<div style="background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #ffc107;">💡 {insight}</div>'
                    
                    st.markdown(insights_html, unsafe_allow_html=True)
                
                # Warnings
                if analysis.warnings:
                    st.markdown("#### ⚠️ Important Warnings")
                    warnings_html = ""
                    for warning in analysis.warnings[:2]:
                        warnings_html += f'<div style="background-color: #f8d7da; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #dc3545;">⚠️ {warning}</div>'
                    
                    st.markdown(warnings_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")
                st.info("Make sure you have set your OpenAI API key in the .env file.")
    else:
        st.info("Click 'Analyze Stock with AI' to get comprehensive AI-powered analysis.")

# Add New Ticker Tab
with tab3:
    st.title("💰 Investo - Add New Ticker")
    
    st.markdown("""
    ### Add a New Stock to the System
    
    This will automatically:
    1. **Download** historical data for the new ticker
    2. **Compute** technical indicators and features
    3. **Retrain** the global LSTM model with the new data
    4. **Generate** predictions for the new ticker
    5. **Update** the system to include the new ticker in all analyses
    
    ⚠️ **Note**: This process may take several minutes as it retrains the entire model.
    """)
    
    # Import the DAG functions
    try:
        from ticker_dag import add_new_ticker, get_pipeline_status, get_available_tickers
        DAG_AVAILABLE = True
    except ImportError as e:
        st.error(f"DAG functionality not available: {e}")
        DAG_AVAILABLE = False
        st.stop()
    
    # Input section for new ticker
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_ticker = st.text_input(
            "Enter New Ticker Symbol", 
            placeholder="e.g., TSLA, META, NFLX",
            help="Enter a valid stock ticker symbol (1-10 characters)"
        ).upper()
    
    with col2:
        add_ticker_clicked = st.button("Add Ticker", type="primary", disabled=not new_ticker)
    
    # Validation
    if new_ticker and len(new_ticker) > 10:
        st.error("Ticker symbol must be 10 characters or less")
        add_ticker_clicked = False
    
    if new_ticker and new_ticker in avail:
        st.warning(f"Ticker {new_ticker} already exists in the system")
        add_ticker_clicked = False
    
    # Process new ticker addition
    if add_ticker_clicked and new_ticker:
        st.markdown("---")
        st.subheader(f"Adding Ticker: {new_ticker}")
        
        # Initialize progress tracking
        if 'ticker_addition_progress' not in st.session_state:
            st.session_state.ticker_addition_progress = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a container for task status
        task_container = st.container()
        
        try:
            with st.spinner("Processing new ticker..."):
                # Run the DAG pipeline
                results = add_new_ticker(new_ticker)
                
                # Display results
                st.markdown("### Pipeline Results")
                
                total_tasks = len(results)
                completed_tasks = 0
                
                for i, (task_name, result) in enumerate(results.items()):
                    completed_tasks += 1
                    progress = completed_tasks / total_tasks
                    progress_bar.progress(progress)
                    
                    # Display task status
                    with task_container:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"**{task_name.replace('_', ' ').title()}**")
                        
                        with col2:
                            if result.status.value == "completed":
                                st.success("✅ Completed")
                            elif result.status.value == "failed":
                                st.error("❌ Failed")
                            elif result.status.value == "skipped":
                                st.warning("⏭️ Skipped")
                            else:
                                st.info(f"🔄 {result.status.value.title()}")
                        
                        with col3:
                            if result.start_time and result.end_time:
                                duration = (result.end_time - result.start_time).total_seconds()
                                st.write(f"{duration:.1f}s")
                        
                        # Show error message if failed
                        if result.error_message:
                            st.error(f"Error: {result.error_message}")
                        
                        st.markdown("---")
                
                # Check if pipeline was successful
                failed_tasks = [name for name, result in results.items() 
                              if result.status.value == "failed"]
                
                if not failed_tasks:
                    st.success(f"🎉 Successfully added {new_ticker} to the system!")
                    st.balloons()
                    
                    # Refresh the available tickers
                    st.rerun()
                else:
                    st.error(f"❌ Pipeline failed. Failed tasks: {', '.join(failed_tasks)}")
                    
        except Exception as e:
            st.error(f"Error adding ticker: {str(e)}")
    
    # Show current system status
    st.markdown("---")
    st.subheader("System Status")
    
    # Get current tickers from database
    try:
        current_tickers = get_available_tickers()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Tickers in System:** {len(current_tickers)}")
            st.write("**Available Tickers:**")
            for ticker in sorted(current_tickers):
                st.write(f"• {ticker}")
        
        with col2:
            st.write("**System Components:**")
            
            # Check database status
            con = duckdb.connect(DB_PATH, read_only=True)
            try:
                price_count = con.execute("SELECT COUNT(*) FROM prices").df().iloc[0, 0]
                feature_count = con.execute("SELECT COUNT(*) FROM features").df().iloc[0, 0]
                pred_count = con.execute("SELECT COUNT(*) FROM predictions").df().iloc[0, 0]
                
                st.write(f"• Price Records: {price_count:,}")
                st.write(f"• Feature Records: {feature_count:,}")
                st.write(f"• Prediction Records: {pred_count:,}")
                
            finally:
                con.close()
    
    except Exception as e:
        st.error(f"Error checking system status: {str(e)}")
    
    # Pipeline information
    st.markdown("---")
    st.subheader("Pipeline Information")
    
    with st.expander("What happens when you add a new ticker?"):
        st.markdown("""
        **1. Validation** 🔍
        - Checks if ticker is valid format
        - Verifies ticker doesn't already exist
        
        **2. Data Ingestion** 📥
        - Downloads historical price data from Yahoo Finance
        - Computes technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - Stores data in DuckDB database
        
        **3. Model Retraining** 🤖
        - Retrains the global LSTM model with all tickers (including new one)
        - Updates model weights and embeddings
        - Saves updated model files
        
        **4. Prediction Generation** 🔮
        - Generates future price predictions for all tickers
        - Creates prediction data for the new ticker
        - Updates prediction tables
        
        **5. Integration Verification** ✅
        - Verifies all components are working
        - Ensures new ticker appears in all analyses
        """)
    
    with st.expander("Technical Details"):
        st.markdown("""
        **Database Tables Updated:**
        - `prices`: Historical OHLCV data
        - `features`: Technical indicators and features
        - `predictions`: Future price predictions
        
        **Model Files Updated:**
        - `global_lstm.pt`: PyTorch model weights
        - `global_scaler.pkl`: Feature scaler
        - `global_meta.json`: Model metadata
        - `ticker_index.json`: Ticker mappings
        
        **Processing Time:**
        - Small datasets: 2-5 minutes
        - Large datasets: 5-15 minutes
        - Depends on data availability and model complexity
        """)

    # Initialize session state for results if not exists
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
