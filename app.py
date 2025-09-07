import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from tab1 import show_big_picture_trend

def show_oscillators(data):
    """Displays common oscillator charts like RSI and MACD."""
    st.header("2. Oscillators")

    # Relative Strength Index (RSI)
    st.subheader("Relative Strength Index (RSI)")
    data['RSI'] = ta.rsi(data['Close'], length=14)
    st.line_chart(data['RSI'])
    
    # Moving Average Convergence Divergence (MACD)
    st.subheader("Moving Average Convergence Divergence (MACD)")
    macd = ta.macd(data['Close'])
    st.area_chart(macd[['MACD_12_26_9', 'MACDs_12_26_9']])

def show_price_action_and_confirmation(data):
    # TODO: Add pattern detection like Head and Shoulders, Double Tops/Bottoms
    # Reversal vs Continuation patterns
    # https://github.com/neurotrader888/TechnicalAnalysisAutomation
    """Displays charts for price action confirmation like Volume and Bollinger Bands."""
    st.header("3. Price Action & Confirmation")
    
    # Volume Data
    st.subheader("Volume Data for Confirmation")
    st.bar_chart(data['Volume'])
    
    # Bollinger Bands (with check for enough data)
    st.subheader("Bollinger Bands")
    if len(data) >= 20:
        bbands = ta.bbands(data['Close'])
        data = pd.concat([data, bbands], axis=1)
        chart_data = data[['Close', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']].copy()
        st.line_chart(chart_data)
    else:
        st.warning("Not enough data points to plot Bollinger Bands (at least 20 required).")
    
    # Average Directional Index (ADX)
    st.subheader("Average Directional Index (ADX)")
    adx = ta.adx(data['High'], data['Low'], data['Close'])
    st.line_chart(adx)

def show_news(ticker):
    """Fetches and displays recent news for the given ticker."""
    st.header("4. Recent News")
    news = yf.Ticker(ticker).news
    if news:
        for item in news:
            st.subheader(item['title'])
            st.write(f"Source: {item['publisher']}")
            st.write(f"Published: {pd.to_datetime(item['providerPublishTime'], unit='s')}")
            st.write(f"[Read more]({item['link']})")
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
            tab1, tab2, tab3, tab4 = st.tabs([
                "Big Picture Trend", 
                "Oscillators", 
                "Price Action & Confirmation", 
                "News"
            ])

            with tab1:
                show_big_picture_trend(data)
            
            with tab2:
                show_oscillators(data)
                
            with tab3:
                show_price_action_and_confirmation(data)
                
            with tab4:
                show_news(ticker)

        else:
            st.warning("No data found for the selected ticker and period. Please try a different ticker or time period.")

    except Exception as e:
        print(f"An error occurred: {e}")
