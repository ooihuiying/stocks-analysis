import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from tab1 import show_big_picture_trend
from tab2 import show_oscillators
from tab3 import show_volume_confirmation_charts
from tab4 import show_reversal_continuation_patterns
from tab5 import show_news_with_sentiment

# Display latest prices for selected tickers
st.title("Trading Strategy Visualizer")
tickers_list = ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA", "META"]
latest_prices = {}

# Fetch latest price for each ticker
for t in tickers_list:
    try:
        stock = yf.Ticker(t)
        latest_prices[t] = stock.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        latest_prices[t] = None
        print(f"Error fetching {t}: {e}")

# Show the latest prices in a nice layout
st.subheader("Latest Stock Prices")
cols = st.columns(len(tickers_list))
for col, t in zip(cols, tickers_list):
    price = latest_prices[t]
    col.metric(label=t, value=f"{price:.2f}")

# User input for the stock ticker and time period
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
        
        # Flatten the column headers if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        if not data.empty and len(data) > 1:
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Big Picture Trend", 
                "Oscillators", 
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
                show_news_with_sentiment(ticker)
        else:
            st.warning("No data found for the selected ticker and period. Please try a different ticker or time period.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
