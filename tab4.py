import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from typing import List
from collections import deque
from dataclasses import dataclass
from rolling_window import rw_top, rw_bottom
from trendline_automation import fit_trendlines_single

# --- The Rest of the Head and Shoulders Code ---
@dataclass
class HSPattern:
    inverted: bool
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1
    l_shoulder_p: float = -1
    r_shoulder_p: float = -1
    l_armpit_p: float = -1
    r_armpit_p: float = -1
    head_p: float = -1
    start_i: int = -1
    break_i: int = -1
    break_p: float = -1
    neck_start: float = -1
    neck_end: float = -1
    neck_slope: float = -1
    head_width: float = -1
    head_height: float = -1
    pattern_r2: float = -1

def compute_pattern_r2(data: np.array, pat: HSPattern):
    line0_slope = (pat.l_shoulder_p - pat.neck_start) / (pat.l_shoulder - pat.start_i)
    line0 = pat.neck_start + np.arange(pat.l_shoulder - pat.start_i) * line0_slope
    line1_slope = (pat.l_armpit_p - pat.l_shoulder_p) / (pat.l_armpit - pat.l_shoulder)
    line1 = pat.l_shoulder_p + np.arange(pat.l_armpit - pat.l_shoulder) * line1_slope
    line2_slope = (pat.head_p - pat.l_armpit_p) / (pat.head - pat.l_armpit)
    line2 = pat.l_armpit_p + np.arange(pat.head - pat.l_armpit) * line2_slope
    line3_slope = (pat.r_armpit_p - pat.head_p) / (pat.r_armpit - pat.head)
    line3 = pat.head_p + np.arange(pat.r_armpit - pat.head) * line3_slope
    line4_slope = (pat.r_shoulder_p - pat.r_armpit_p) / (pat.r_shoulder - pat.r_armpit)
    line4 = pat.r_armpit_p + np.arange(pat.r_shoulder - pat.r_armpit) * line4_slope
    line5_slope = (pat.break_p - pat.r_shoulder_p) / (pat.break_i - pat.r_shoulder)
    line5 = pat.r_shoulder_p + np.arange(pat.break_i - pat.r_shoulder) * line5_slope
    
    raw_data = data[pat.start_i:pat.break_i]
    hs_model = np.concatenate([line0, line1, line2, line3, line4, line5])
    
    if len(raw_data) != len(hs_model):
        return np.nan

    mean = np.mean(raw_data)
    ss_res = np.sum((raw_data - hs_model)**2.0)
    ss_tot = np.sum((raw_data - mean)**2.0)
    
    if ss_tot == 0:
        return np.nan
        
    r2 = 1.0 - ss_res / ss_tot
    return r2

def check_hs_pattern(extrema_indices: List[int], data: np.array, i: int, early_find: bool = False) -> HSPattern:
    l_shoulder, l_armpit, head, r_armpit = extrema_indices[0], extrema_indices[1], extrema_indices[2], extrema_indices[3]
    if i - r_armpit < 2: return None
    r_shoulder = r_armpit + data[r_armpit + 1: i].argmax() + 1
    if data[head] <= max(data[l_shoulder], data[r_shoulder]): return None
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] < r_midpoint or data[r_shoulder] < l_midpoint: return None
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time: return None
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    if neck_run == 0: return None
    neck_slope = neck_rise / neck_run
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    if early_find:
        if data[i] > r_midpoint: return None
    else:
        if data[i] > neck_val: return None
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0: return None
        if data[l_shoulder - j] < neck:
            pat_start, neck_start = l_shoulder - j, neck
            break
    if pat_start == -1: return None
    pat = HSPattern(inverted=False)
    pat.l_shoulder, pat.r_shoulder, pat.l_armpit, pat.r_armpit, pat.head = l_shoulder, r_shoulder, l_armpit, r_armpit, head
    pat.l_shoulder_p, pat.r_shoulder_p, pat.l_armpit_p, pat.r_armpit_p, pat.head_p = data[l_shoulder], data[r_shoulder], data[l_armpit], data[r_armpit], data[head]
    pat.start_i, pat.break_i, pat.break_p = pat_start, i, data[i]
    pat.neck_start, pat.neck_end = neck_start, neck_val
    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    return pat

def check_ihs_pattern(extrema_indices: List[int], data: np.array, i: int, early_find: bool = False) -> HSPattern:
    l_shoulder, l_armpit, head, r_armpit = extrema_indices[0], extrema_indices[1], extrema_indices[2], extrema_indices[3]
    if i - r_armpit < 2: return None
    r_shoulder = r_armpit + data[r_armpit+1: i].argmin() + 1
    if data[head] >= min(data[l_shoulder], data[r_shoulder]): return None
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] > r_midpoint or data[r_shoulder] > l_midpoint: return None
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time: return None
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    if neck_run == 0: return None
    neck_slope = neck_rise / neck_run
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    if early_find:
        if data[i] < r_midpoint: return None
    else:
        if data[i] < neck_val: return None
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        if l_shoulder - j < 0: return None
        if data[l_shoulder - j] > neck:
            pat_start, neck_start = l_shoulder - j, neck
            break
    if pat_start == -1: return None
    pat = HSPattern(inverted=True)
    pat.l_shoulder, pat.r_shoulder, pat.l_armpit, pat.r_armpit, pat.head = l_shoulder, r_shoulder, l_armpit, r_armpit, head
    pat.l_shoulder_p, pat.r_shoulder_p, pat.l_armpit_p, pat.r_armpit_p, pat.head_p = data[l_shoulder], data[r_shoulder], data[l_armpit], data[r_armpit], data[head]
    pat.start_i, pat.break_i, pat.break_p = pat_start, i, data[i]
    pat.neck_start, pat.neck_end = neck_start, neck_val
    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    return pat

def find_hs_patterns(data: np.array, order: int, early_find: bool = False):
    assert(order >= 1)
    last_is_top = False
    recent_extrema = deque(maxlen=5)
    recent_types = deque(maxlen=5)
    hs_lock, ihs_lock = False, False
    ihs_patterns, hs_patterns = [], []
    for i in range(len(data)):
        if rw_top(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(1)
            ihs_lock, last_is_top = False, True
        if rw_bottom(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(-1)
            hs_lock, last_is_top = False, False
        if len(recent_extrema) < 5: continue
        hs_alternating, ihs_alternating = True, True
        if last_is_top:
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: ihs_alternating = False
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: hs_alternating = False
            ihs_extrema, hs_extrema = list(recent_extrema)[1:5], list(recent_extrema)[0:4]
        else:
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: hs_alternating = False
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: ihs_alternating = False
            ihs_extrema, hs_extrema = list(recent_extrema)[0:4], list(recent_extrema)[1:5]
        if ihs_lock or not ihs_alternating: ihs_pat = None
        else: ihs_pat = check_ihs_pattern(ihs_extrema, data, i, early_find)
        if hs_lock or not hs_alternating: hs_pat = None
        else: hs_pat = check_hs_pattern(hs_extrema, data, i, early_find)
        if hs_pat is not None:
            hs_lock, hs_patterns = True, hs_patterns + [hs_pat]
        if ihs_pat is not None:
            ihs_lock, ihs_patterns = True, ihs_patterns + [ihs_pat]
    return hs_patterns, ihs_patterns

def plot_hs(candle_data: pd.DataFrame, pat: HSPattern, pad: int = 2):
    if pad < 0: pad = 0
    idx = candle_data.index
    data = candle_data.iloc[pat.start_i:pat.break_i + 1 + pad]
    fig, ax = plt.subplots(figsize=(10, 6))
    l0 = [(idx[pat.start_i], pat.neck_start), (idx[pat.l_shoulder], pat.l_shoulder_p)]
    l1 = [(idx[pat.l_shoulder], pat.l_shoulder_p), (idx[pat.l_armpit], pat.l_armpit_p)]
    l2 = [(idx[pat.l_armpit], pat.l_armpit_p ), (idx[pat.head], pat.head_p)]
    l3 = [(idx[pat.head], pat.head_p ), (idx[pat.r_armpit], pat.r_armpit_p)]
    l4 = [(idx[pat.r_armpit], pat.r_armpit_p ), (idx[pat.r_shoulder], pat.r_shoulder_p)]
    l5 = [(idx[pat.r_shoulder], pat.r_shoulder_p ), (idx[pat.break_i], pat.neck_end)]
    neck = [(idx[pat.start_i], pat.neck_start), (idx[pat.break_i], pat.neck_end)]
    mpf.plot(data, alines=dict(alines=[l0, l1, l2, l3, l4, l5, neck], colors=['w', 'w', 'w', 'w', 'w', 'w', 'r']), type='candle', style='charles', ax=ax)
    return fig

def show_head_and_shoulders_trend(data):
    st.subheader("Head & Shoulders Patterns")
    st.info("Head & Shoulders (H&S) is a classic reversal pattern. A regular H&S suggests a bearish reversal, while an inverted (IHS) suggests a bullish reversal.")
    
    if len(data) < 100:
        st.info("Not enough data to check for Head & Shoulders patterns. A longer `period` is required.")
        return
    
    # Run the pattern detection
    hs_patterns, ihs_patterns = find_hs_patterns(data['Close'].to_numpy(), order=5)
    
    if hs_patterns:
        st.subheader("Bearish Head & Shoulders Pattern Found 📉")
        for pat in hs_patterns:
            fig = plot_hs(data, pat)
            st.pyplot(fig)
            st.success(f"Bearish H&S Pattern detected from {data.index[pat.start_i].strftime('%Y-%m-%d')} to {data.index[pat.break_i].strftime('%Y-%m-%d')}. This suggests a potential downtrend.")
    
    if ihs_patterns:
        st.subheader("Bullish (Inverted) Head & Shoulders Pattern Found 📈")
        for pat in ihs_patterns:
            fig = plot_hs(data, pat)
            st.pyplot(fig)
            st.success(f"Bullish IHS Pattern detected from {data.index[pat.start_i].strftime('%Y-%m-%d')} to {data.index[pat.break_i].strftime('%Y-%m-%d')}. This suggests a potential uptrend.")
            
    if not hs_patterns and not ihs_patterns:
        st.info("No Head & Shoulders patterns detected in the selected period.")

@dataclass
class FlagPattern:
    base_x: int
    base_y: float
    tip_x: int = -1
    tip_y: float = -1.
    conf_x: int = -1
    conf_y: float = -1.
    pennant: bool = False
    flag_width: int = -1
    flag_height: float = -1.
    pole_width: int = -1
    pole_height: float = -1.
    support_intercept: float = -1.
    support_slope: float = -1.
    resist_intercept: float = -1.
    resist_slope: float = -1.

def check_bull_pattern_trendline(pending: FlagPattern, data: np.array, i:int, order:int):
    # Check if data max less than pole tip
    if data[pending.tip_x + 1 : i].max() > pending.tip_y:
        return False

    flag_min = data[pending.tip_x:i].min()

    # Find flag/pole height and width
    pole_height = pending.tip_y - pending.base_y
    pole_width = pending.tip_x - pending.base_x
    
    flag_height = pending.tip_y - flag_min
    flag_width = i - pending.tip_x

    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    if flag_height > pole_height * 0.75: # Flag should smaller vertically than preceding trend
        return False

    # Find trendlines going from flag tip to the previous bar (not including current bar)
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]

    # Check for breakout of upper trendline to confirm pattern
    current_resist = resist_intercept + resist_slope * (flag_width + 1)
    if data[i] <= current_resist:
        return False

    # Pattern is confiremd, fill out pattern details in pending
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept
    
    return True

def check_bear_pattern_trendline(pending: FlagPattern, data: np.array, i:int, order:int):
    # Check if data max less than pole tip
    if data[pending.tip_x + 1 : i].min() < pending.tip_y:
        return False

    flag_max = data[pending.tip_x:i].max()

    # Find flag/pole height and width
    pole_height = pending.base_y - pending.tip_y
    pole_width = pending.tip_x - pending.base_x
    
    flag_height = flag_max - pending.tip_y
    flag_width = i - pending.tip_x

    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    if flag_height > pole_height * 0.75: # Flag should smaller vertically than preceding trend
        return False

    # Find trendlines going from flag tip to the previous bar (not including current bar)
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]

    # Check for breakout of lower trendline to confirm pattern
    current_support = support_intercept + support_slope * (flag_width + 1)
    if data[i] >= current_support:
        return False

    # Pattern is confiremd, fill out pattern details in pending
    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False

    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept
    
    return True

def find_flags_pennants_trendline(data: np.array, order:int):
    assert(order >= 3)
    pending_bull = None # Pending pattern
    pending_bear = None # Pending pattern
    last_bottom = -1
    last_top = -1
    bull_pennants = []
    bear_pennants = []
    bull_flags = []
    bear_flags = []
    
    for i in range(len(data)):
        if rw_top(data, i, order):
            last_top = i - order
            if last_bottom != -1:
                pending = FlagPattern(last_bottom, data[last_bottom])
                pending.tip_x = last_top
                pending.tip_y = data[last_top]
                pending_bull = pending
        
        if rw_bottom(data, i, order):
            last_bottom = i - order
            if last_top != -1:
                pending = FlagPattern(last_top, data[last_top])
                pending.tip_x = last_bottom
                pending.tip_y = data[last_bottom]
                pending_bear = pending

        if pending_bear is not None:
            if check_bear_pattern_trendline(pending_bear, data, i, order):
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)
                else:
                    bear_flags.append(pending_bear)
                pending_bear = None
        
        if pending_bull is not None:
            if check_bull_pattern_trendline(pending_bull, data, i, order):
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)
                else:
                    bull_flags.append(pending_bull)
                pending_bull = None
    return bull_flags, bear_flags, bull_pennants, bear_pennants

def plot_flag(candle_data: pd.DataFrame, pattern: FlagPattern, pad: int = 2):
    """
    Plots a detected flag or pennant pattern on a candlestick chart.
    
    Args:
        candle_data (pd.DataFrame): The full candlestick data.
        pattern (FlagPattern): The detected pattern object.
        pad (int): The number of data points to plot before and after the pattern.
    """
    if pad < 0:
        pad = 0
    start_i = pattern.base_x
    end_i = pattern.conf_x + 1
    dat = candle_data.iloc[start_i - pad:end_i + pad]
    idx = dat.index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate the plot indices for the pattern
    base_plot_idx = pattern.base_x - (start_i - pad)
    tip_plot_idx = pattern.tip_x - (start_i - pad)
    conf_plot_idx = pattern.conf_x - (start_i - pad)
    
    # Define the lines for plotting
    pole_line = [(idx[base_plot_idx], pattern.base_y), (idx[tip_plot_idx], pattern.tip_y)]
    
    # Calculate flag line endpoints based on relative index and slope
    resist_end_y = pattern.resist_intercept + pattern.resist_slope * (pattern.flag_width + 1)
    resist_line = [(idx[tip_plot_idx], pattern.resist_intercept), (idx[conf_plot_idx], resist_end_y)]
    
    support_end_y = pattern.support_intercept + pattern.support_slope * (pattern.flag_width + 1)
    support_line = [(idx[tip_plot_idx], pattern.support_intercept), (idx[conf_plot_idx], support_end_y)]

    mpf.plot(dat, alines=dict(alines=[pole_line, support_line, resist_line], colors=['w', 'b', 'b']), type='candle', style='charles', ax=ax)
    plt.title(f"{'Bullish' if pattern.pennant else 'Bearish'} {'Pennant' if pattern.pennant else 'Flag'} Pattern")
    return fig

def show_flag_and_pennant(data):
    """
    Analyzes and displays Flag and Pennant patterns in the given financial data.

    Args:
        data (pd.DataFrame): A DataFrame containing the candlestick data.
    """
    st.subheader("Flag and Pennant Patterns")
    st.info("Flag and Pennant patterns are short-term continuation patterns that form a small consolidation after a sharp, linear move (the 'pole'). They signal that the preceding trend is likely to continue.")
    
    if len(data) < 100:
        st.info("Not enough data to check for Flag and Pennant patterns. A longer `period` is required.")
        return
    
    # Run the pattern detection
    data_close = data['Close'].to_numpy()
    
    # We will use a smaller `order` to make the detection more sensitive.
    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(data_close, order=10)
    
    # Display Bullish Patterns
    if bull_flags or bull_pennants:
        st.subheader("Bullish Continuation Patterns Found 📈")
        for pat in bull_flags:
            st.success(f"Bullish Flag Pattern detected, suggesting a continuation of the uptrend.")
            fig = plot_flag(data, pat)
            st.pyplot(fig)
        for pat in bull_pennants:
            st.success(f"Bullish Pennant Pattern detected, suggesting a continuation of the uptrend.")
            fig = plot_flag(data, pat)
            st.pyplot(fig)
    
    # Display Bearish Patterns
    if bear_flags or bear_pennants:
        st.subheader("Bearish Continuation Patterns Found 📉")
        for pat in bear_flags:
            st.success(f"Bearish Flag Pattern detected, suggesting a continuation of the downtrend.")
            fig = plot_flag(data, pat)
            st.pyplot(fig)
        for pat in bear_pennants:
            st.success(f"Bearish Pennant Pattern detected, suggesting a continuation of the downtrend.")
            fig = plot_flag(data, pat)
            st.pyplot(fig)
            
    if not (bull_flags or bull_pennants or bear_flags or bear_pennants):
        st.error("No Flag or Pennant patterns detected in the selected period.")

def show_reversal_continuation_patterns(data):
    st.header("4a. Reversal Patterns")
    show_head_and_shoulders_trend(data)
    st.header("4b. Continuation Patterns")
    show_flag_and_pennant(data)