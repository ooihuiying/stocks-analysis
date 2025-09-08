import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from tab1 import show_big_picture_trend
from tab2 import show_oscillators
from tab3 import show_volume_confirmation_charts
from tab4 import show_reversal_continuation_patterns

def show_news(ticker):
    """Fetches and displays recent news for the given ticker."""
    st.header("4. Recent News")
    news = yf.Ticker(ticker).news
    if news:
        st.info(news)
        
    else:
        st.info("No recent news found for this ticker.")

# --- Main App Logic ---

# User input for the stock ticker and time period
st.title("Trading Strategy Visualizer")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", "AAPL")
period_options = {
    "6 months": "6mo",
    "1 month": "1mo",
    "3 weeks": "3wk",
    "1 year": "1y",
    "10 years": "10y",
    "all time": "max",
    "1 day": "1d",
    "4 days": "4d",
    "1 week": "1wk",
}

selected_period_name = st.selectbox("Select a time period:", list(period_options.keys()))
selected_period = period_options[selected_period_name]

# Fetch data and handle potential errors
if ticker:
    try:
        data = yf.download(ticker, period=selected_period)
        
        # Fix: Flatten the column headers to avoid multi-level index issues
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Check if data is empty before proceeding
        if not data.empty and len(data) > 1:   
            
            # Create tabs and call the functions to render their content
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Big Picture Trend", # check overall trend
                "Oscillators", # check momentum
                "Volume to confirm Price Action",
                "Reversals & Continuations", 
                "News & Sentiment"
            ])

            with tab1:
                show_big_picture_trend(data)
            
            with tab2:
                show_oscillators(data)
                
            with tab3:
                show_volume_confirmation_charts(data)
            with tab4:
                show_reversal_continuation_patterns(data)
            with tab5:
                show_news(ticker)

        else:
            st.warning("No data found for the selected ticker and period. Please try a different ticker or time period.")

    except Exception as e:
        print(f"An error occurred: {e}")
