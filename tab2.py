import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

def show_oscillators(data):
    """Displays common oscillator charts with improved visuals."""
    st.header("2. Oscillators")
    st.info("Strong Buy Signal: Occurs when the RSI is moving out of the oversold region (e.g., crossing above 30) and the MACD has a bullish crossover (MACD line crosses above the signal line) with green histogram bars. This provides strong confirmation of a potential upward trend.")
    st.info("Strong Sell Signal: Occurs when the RSI is moving out of the overbought region (e.g., dropping below 70) and the MACD has a bearish crossover (MACD line crosses below the signal line) with red histogram bars. This suggests a likely downward trend.")
    
    # 1. Relative Strength Index (RSI)
    st.subheader("Relative Strength Index (RSI)")
    st.write("RSI values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.")
    st.write("Ideal RSI range for trading is between 30 and 70.")
    st.info("In an uptrend, buy dips when RSI is 30-40; in a downtrend, sell rallies when RSI is 60-70.")
    data['RSI'] = ta.rsi(data['Close'], length=14)
    st.line_chart(data['RSI'])
    
    # 2. Moving Average Convergence Divergence (MACD)
    st.subheader("Moving Average Convergence Divergence (MACD)")
    # Ensure the MACD calculation is correct
    macd_df = ta.macd(data['Close'])
    
    if macd_df.empty:
        st.error("MACD data could not be generated. This may be due to insufficient data points for the calculation.")
        return

    # Dynamically find MACD, Signal Line, and Histogram column names
    macd_col = [col for col in macd_df.columns if 'MACD_' in col][0]
    signal_col = [col for col in macd_df.columns if 'MACDs_' in col][0]
    hist_col = [col for col in macd_df.columns if 'MACDh_' in col][0]
    
    # Create a clean DataFrame for charting
    macd_data = pd.DataFrame({
        'MACD': macd_df[macd_col],
        'Signal Line': macd_df[signal_col],
        'Histogram': macd_df[hist_col]
    })
    
    # Create new columns for positive and negative values to allow for custom coloring
    macd_data['Positive'] = macd_data['Histogram'].apply(lambda x: x if x >= 0 else None)
    macd_data['Negative'] = macd_data['Histogram'].apply(lambda x: x if x < 0 else None)
    
    # 2a. MACD Histogram Bar Chart
    st.info("MACD Histogram: The histogram bars turn green (or positive), which visually confirms that the MACD line is now above the signal line.")
    st.bar_chart(macd_data[['Positive', 'Negative']], color=['#33CC00', '#FF3300'])

    # 2b. MACD Line and Signal Line Chart
    st.info("MACD Line Crossover: The MACD line crosses above the signal line. This is the primary signal for an increase in upward momentum.")
    st.line_chart(macd_data[['MACD', 'Signal Line']])

    # 3. Check for and display signals
    current_rsi = data['RSI'].iloc[-1]
    last_two_rsi = data['RSI'].iloc[-2:]
    
    macd_crossover = macd_data['MACD'].iloc[-1] > macd_data['Signal Line'].iloc[-1] and macd_data['MACD'].iloc[-2] <= macd_data['Signal Line'].iloc[-2]
    
    # Check for strong buy signal
    if current_rsi > 30 and last_two_rsi.iloc[0] <= 30 and macd_crossover and macd_data['Histogram'].iloc[-1] > 0:
        st.success("✅ **Strong Buy Signal**: RSI is moving out of the oversold region and MACD has a bullish crossover with a green histogram. This suggests a potential upward trend.")

    # Check for strong sell signal
    elif current_rsi < 70 and last_two_rsi.iloc[0] >= 70 and not macd_crossover and macd_data['Histogram'].iloc[-1] < 0:
        st.error("❌ **Strong Sell Signal**: RSI is moving out of the overbought region and MACD has a bearish crossover with a red histogram. This suggests a potential downward trend.")

    else:
        st.info("No strong buy or sell signal detected based on the current data.")