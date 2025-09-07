import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

def simple_volume_analysis(data):
    # 1. Volume Data
    st.subheader("Volume Data for Confirmation")
    st.bar_chart(data['Volume'])
    # Get the latest volume compare with the latest average volume
    data['Volume_SMA'] = ta.sma(data['Volume'], length=20) #20 period SMA
    latest_volume = data['Volume'].iloc[-1]
    latest_volume_sma = data['Volume_SMA'].iloc[-1]
    if latest_volume > 2 * latest_volume_sma:
        st.success("📈 **High Volume Detected**: The current volume is more than double the 20-day average. This indicates strong conviction.")
    elif latest_volume > latest_volume_sma:
        st.success("📊 **Above Average Volume**: The current volume is higher than the 20-day average. This suggests healthy participation.")
    else:
        st.error("📉 **Below Average Volume**: The current volume is below the 20-day average. This indicates a lack of significant interest.")

def on_balance_volume(data):
    # 2. Volume in detail
    st.subheader("On-Balance Volume (OBV)")
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    st.line_chart(data['OBV'])
    obv_current = data['OBV'].iloc[-1]

    obv_10_days_ago = data['OBV'].iloc[-10]

    if obv_current > obv_10_days_ago:
        st.info("📈 **Rising OBV**: The OBV is trending up. This indicates buying pressure and confirms the current uptrend.")
    else:
        st.info("📉 **Falling OBV**: The OBV is trending down. This indicates selling pressure and confirms the current downtrend.")

    # b. Check for Bullish and Bearish Divergence
    # Look for divergence over the last 30 trading days
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
        st.success("📊 **Bullish Divergence Detected**: The price is making a lower low, but the OBV is making a higher low. This suggests a potential reversal to the upside.")

    # Bearish Divergence check (Price higher high, OBV lower high)
    elif (second_half.loc[second_half_high_idx]['Close'] > first_half.loc[first_half_high_idx]['Close']) and \
         (second_half.loc[second_half_high_idx]['OBV'] < first_half.loc[first_half_high_idx]['OBV']):
        st.error("📉 **Bearish Divergence Detected**: The price is making a higher high, but the OBV is making a lower high. This suggests a potential reversal to the downside.")
    else:
        st.info("No bullish or bearish divergence detected.")
    
def show_price_action_and_confirmation(data):
    # TODO: Add pattern detection like Head and Shoulders, Double Tops/Bottoms
    # Reversal vs Continuation patterns
    # https://github.com/neurotrader888/TechnicalAnalysisAutomation
    """Displays charts for price action confirmation like Volume and Bollinger Bands."""
    st.header("3. Price Action & Confirmation")
    simple_volume_analysis(data)
    on_balance_volume(data)
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
