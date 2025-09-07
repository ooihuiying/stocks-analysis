import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# import mplfinance as mpf
import matplotlib.pyplot as plt

def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    """
    Computes the squared sum of differences between a line and the price data.
    
    Args:
        support (bool): True for a support line, False for a resistance line.
        pivot (int): The index of the pivot point the line passes through.
        slope (float): The slope of the line.
        y (np.array): The price data.

    Returns:
        float: The squared sum of differences, or -1.0 if the line is invalid.
    """
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    # For support, all price points must be above or on the line.
    # For resistance, all price points must be below or on the line.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    """
    Optimizes the slope of a trendline to minimize the squared error while
    respecting the support/resistance constraints.
    
    Args:
        support (bool): True for a support line, False for a resistance line.
        pivot (int): The index of the pivot point.
        init_slope (float): The initial slope to start the optimization.
        y (np.array): The price data.

    Returns:
        tuple: A tuple containing the optimized slope and intercept.
    """
    # Amount to change slope by. Multiplied by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optimization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation to determine direction for slope change.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            
            # If increasing by a small amount fails, try decreasing
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data.")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        
        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # Slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # Test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimization is done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    """
    Fits support and resistance trendlines to a single series of data.
    """
    # Find line of best fit (least squares)
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    """
    Fits support and resistance trendlines using high, low, and close prices.
    """
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


def plot_support_resistance(data: pd.DataFrame):
    """
    Plots the closing price and the calculated trendlines.
    
    Args:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' prices.
        lookback (int): The number of data points to use for trendline calculation.
    """
    st.subheader("Plotting Trendlines")  
    candles = data.copy()  # Work on a copy of the data

    # Take natural log of data to resolve price scaling issues
    candles['High'] = np.log(candles['High'])
    candles['Low'] = np.log(candles['Low'])
    candles['Close'] = np.log(candles['Close'])

    # Fit trendlines
    support_coefs, resist_coefs = fit_trendlines_high_low(
        candles['High'].values, 
        candles['Low'].values, 
        candles['Close'].values
    )
    
    # Create the trendline data points
    x = np.arange(len(candles))
    support_line = support_coefs[0] * x + support_coefs[1]
    resist_line = resist_coefs[0] * x + resist_coefs[1]

    # Convert the resistance line array to a Pandas Series for easy shifting
    resistance_series = pd.Series(resist_line, index=candles.index)
    
    # Detect breakouts (when price crosses above resistance line)
    candles['Breakout_Signal'] = np.where(
        (candles['Close'].shift(1) < resistance_series.shift(1)) & 
        (candles['Close'] >= resistance_series),
        candles['Close'],  # Mark the breakout point with the closing price
        np.nan # Use NaN for non-breakout points
    )

    # Create the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the log-scaled close price
    ax.plot(candles.index, candles['Close'], label='Log Close Price', color='white', linewidth=2)
    
    # Plot the trendlines
    ax.plot(candles.index, support_line, label='Support Trendline', color='green', linestyle='--', linewidth=2)
    ax.plot(candles.index, resist_line, label='Resistance Trendline', color='red', linestyle='--', linewidth=2)
    
    # Plot the breakout signals
    ax.scatter(candles.index, candles['Breakout_Signal'], marker='^', color='red', s=200, label='Breakout Signal', zorder=5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Price')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_basic_trend(data):
    st.subheader("Raw Data and Line Trend")
    # Raw closing price chart
    st.line_chart(data['Close'])
    last_close = data['Close'].iloc[-1].item()
    first_close = data['Close'].iloc[0].item()

    if last_close > first_close:
        trend = "Uptrend 📈"
    else:
        trend = "Downtrend 📉"
    st.info(f"The long-term trend appears to be a **{trend}**.")

def plot_moving_averages(data):
    st.subheader("Double Moving Averages")
    st.write("Typical combis: 5 and 20 days, 10 and 50 days. When the shorter line crosses above the longer line, it is a bullish sign.")
    st.write("The crossover is simply a confirmation that the new upward momentum is gaining strength. It is a lagging indicator.")
    col1, col2 = st.columns(2)
    with col1:
        window1 = st.number_input("Short-term MA Window", min_value=1, value=5, step=1, key="ma_window1")
    with col2:
        window2 = st.number_input("Long-term MA Window", min_value=1, value=20, step=1, key="ma_window2")

    # Calculate moving averages based on user input
    ma_data = data.copy()
    ma_data['MA_Short'] = ma_data['Close'].rolling(window=window1).mean()
    ma_data['MA_Long'] = ma_data['Close'].rolling(window=window2).mean()

    # Create the plot using Matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the Close price and the two moving averages
    ax.plot(ma_data.index, ma_data['Close'], label='Close Price', color='white', linestyle='--', linewidth=0.5)
    ax.plot(ma_data.index, ma_data['MA_Short'], label=f'MA {window1}', color='pink', linewidth=2)
    ax.plot(ma_data.index, ma_data['MA_Long'], label=f'MA {window2}', color='orange', linewidth=2)

    # Detect buy signals (bullish crossovers)
    ma_data['Buy_Signal'] = np.where(
        (ma_data['MA_Short'].shift(1) < ma_data['MA_Long'].shift(1)) & 
        (ma_data['MA_Short'] >= ma_data['MA_Long']),
        ma_data['Close'],  # Mark the crossover point with the closing price
        np.nan # Use NaN for non-crossover points
    )
    
    # Plot the buy signals on the chart
    ax.scatter(ma_data.index, ma_data['Buy_Signal'], marker='^', color='green', s=200, label='Buy Signal', zorder=5)

    ax.set_title(f'Moving Averages for {window1} and {window2} periods with Buy Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

def show_big_picture_trend(data):
    """Displays the main price chart and a basic trend analysis."""
    st.header("1. Big Picture: Market Context")
    st.write("Refer to TradingView's Lux Algo for a comprehensive resistance and support analysis.")
    plot_basic_trend(data)
    plot_support_resistance(data)
    plot_moving_averages(data)
