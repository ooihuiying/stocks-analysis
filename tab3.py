import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

def simple_volume_analysis(data):
    # Volume Data
    st.subheader("Volume Data for Confirmation")
    st.bar_chart(data['Volume'])
    # Get the latest volume compare with the latest average volume
    data['Volume_SMA'] = ta.sma(data['Volume'], length=20) # 20-day period SMA
    latest_volume = data['Volume'].iloc[-1]
    latest_volume_sma = data['Volume_SMA'].iloc[-1]
    if latest_volume > 2 * latest_volume_sma:
        st.success("📈 **High Volume Detected**: The current volume is more than double the 20-day average. This indicates strong conviction.")
    elif latest_volume > latest_volume_sma:
        st.success("📊 **Above Average Volume**: The current volume is higher than the 20-day average. This suggests healthy participation.")
    else:
        st.error("📉 **Below Average Volume**: The current volume is below the 20-day average. This indicates a lack of significant interest.")

def on_balance_volume(data):
    st.subheader("On-Balance Volume (OBV)")
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    st.line_chart(data['OBV'])
    obv_current = data['OBV'].iloc[-1]

    obv_10_days_ago = data['OBV'].iloc[-10]

    if obv_current > obv_10_days_ago:
        st.success("📈 **Rising OBV**: The OBV is trending up. This indicates buying pressure and confirms the current uptrend.")
    elif obv_current:
        st.error("📉 **Falling OBV**: The OBV is trending down. This indicates selling pressure and confirms the current downtrend.")
    else:
        st.info("⚖️ **Flat OBV**: The OBV is relatively flat. This indicates a lack of conviction in either direction.")

    # Check for Bullish and Bearish Divergence
    # Look for divergence over the last 30 trading days
    st.info("Checking for Bullish/Bearish Divergence between Price and OBV. Used for confirmation of potential reversals. 30 days lookback.")
    lookback_period = 30
    recent_data = data.tail(lookback_period)
    
    # Divide the data into two halves
    half_period = lookback_period // 2
    first_half = recent_data.head(half_period)
    second_half = recent_data.tail(half_period)
    
    # Find peaks and troughs in each half
    first_half_low_idx = first_half['Close'].idxmin()
    first_half_high_idx = first_half['Close'].idxmax()
    second_half_low_idx = second_half['Close'].idxmin()
    second_half_high_idx = second_half['Close'].idxmax()

    # Bullish Divergence check (Price lower low, OBV higher low)
    if (second_half.loc[second_half_low_idx]['Close'] < first_half.loc[first_half_low_idx]['Close']) and \
       (second_half.loc[second_half_low_idx]['OBV'] > first_half.loc[first_half_low_idx]['OBV']):
        st.success("📊 **Bullish Divergence Detected**: The price is making a lower low, but the OBV is making a higher low (more buying than selling). This suggests a potential reversal to the upside.")
    # Bearish Divergence check (Price higher high, OBV lower high)
    elif (second_half.loc[second_half_high_idx]['Close'] > first_half.loc[first_half_high_idx]['Close']) and \
         (second_half.loc[second_half_high_idx]['OBV'] < first_half.loc[first_half_high_idx]['OBV']):
        st.error("📉 **Bearish Divergence Detected**: The price is making a higher high, but the OBV is making a lower high. This suggests a potential reversal to the downside.")
    else:
        st.info("No bullish or bearish divergence detected.")
    
def show_volume_confirmation_charts(data):
    """Displays charts for volume to confirm price action."""
    st.header("3. Volume Confirmation")
    simple_volume_analysis(data)
    on_balance_volume(data)