"""
grams_tools.py
LArTPC Analysis Toolkit
-----------------------
Library for subsytem-dependent baseline correction algorithms. 

Author: Jonathan LeyVa
Date: Jan 15,2026
"""

import numpy as np

# --- CONFIGURATION ---
def get_pre_trigger_samples(res_ns, safe_window_us):
    """Calculates number of samples to use for baseline estimation."""
    # This value must be passed by the user as it will change for different runs and configs
    if safe_window_us is None:
        raise ValueError("Please define the pre-trigger window used for this data!")
    return int(safe_window_us / (res_ns * 1e-3))

# --- SiPM BC (Iterative Clipping) ---
def baseline_iterative_clipping_RT(segment, sigma=2.5, max_iter=10):
    """
    Tailored for room temperature SiPM channel baseline correction. Designed to 
    handle events with pre-trigger activity (pileup) or channels with large and 
    frequent dark counts (e.g., overbiased)
    
    Args:
        segment (array): The pre-trigger waveform data.
        sigma (float): Threshold for outlier rejection.
        max_iter (int): Maximum iterations.
        
    Returns:
        float: The estimated baseline voltage to add or subtract from raw trace (mV).
    """
    clean_seg = segment.copy()
    
    for i in range(max_iter):
        # Calculate mean and standard deviation of samples
        mu = np.median(clean_seg)
        std = np.std(clean_seg)
        
        if std == 0: break # Safety catch for flat lines
        
        # Define clipping bounds 
        lower_bound = mu - sigma * std
        upper_bound = mu + sigma * std 
        
        # Keep only data within bounds
        mask = (clean_seg >= lower_bound) & (clean_seg <= upper_bound)
        
        # Convergence check: if no new points removed, stop short of max iterations
        if np.sum(mask) == len(clean_seg):
            break
            
        clean_seg = clean_seg[mask]
        
    if len(clean_seg) == 0: return 0.0 # Prevents crash if mask removes all samples 
    return np.mean(clean_seg)

# --- CHARGE BC ---
def baseline_charge(segment, fine_tune_window=2.0):
    """
    Calculates baseline correction for CSP channels.
    Uses Median to reject switching noise or microphonics, then fine-tunes on quiet samples.
    
    Args:
        segment (array): The pre-trigger waveform data.
        fine_tune_window (float): Window (+/- mV) around median to include in mean.
        
    Returns:
        float: The estimated baseline voltage (mV).
    """
    # Median (Robust against transients)
    rough_bl = np.median(segment)
    
    # Shift the waveform close to zero mV
    residuals = segment - rough_bl
    
    # Remove any part of the waveform outside of +/- [fine_tune_window]  mV
    mask_quiet = (np.abs(residuals) < fine_tune_window)
    
    # Prevent crash or wild result if mask removes too many samples
    if np.sum(mask_quiet) > 10:
        fine_adjustment = np.mean(residuals[mask_quiet])
    else:
        fine_adjustment = 0.0
        
    return rough_bl + fine_adjustment

def get_freq_axis(n_samples, sample_rate_ns):
    """
    Returns the frequency axis for an rfft.
    
    Args:
        n_samples (int): Length of the trace.
        sample_rate_ns (float): Sampling period in nanoseconds (e.g., 2.0 for 500MS/s).
    """
    dt = sample_rate_ns * 1e-9 
    # rfftfreq returns the correct bins for real-input FFT
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    return freqs

def compute_psd_trace(trace_data, sample_rate_ns, window_type='hanning', imp_ohm=50.0):
    """
    Computes the Power Spectral Density (dBm/Hz) for a single trace.
    
    Args:
        trace_data (array): Voltage data (Linear, usually mV).
        sample_rate_ns (float): Sampling period in ns.
        window_type (str): 'hanning', 'blackman', or 'rect'.
        imp_ohm (float): Input impedance (default 50 Ohm for CAEN).
        
    Returns:
        array: PSD values in dBm/Hz.
    """
    # Define time bin in seconds
    dt_sec = sample_rate_ns * 1e-9
    N = len(trace_data)
    
    # 1. Generate window to mitigate spectral leakage
    if window_type == 'hanning':
        w = np.hanning(N)
    elif window_type == 'blackman':
        w = np.blackman(N)
    else:
        w = np.ones(N) 
    
    # Apply Window
    windowed_trace = w * trace_data
    
    # 2. FFT
    # CAUTION: Standard np.fft.rfft output is unnormalized.
    fhat = np.fft.rfft(windowed_trace)
    
    # 3. Energy Extraction
    # Extract energy (careful, this only accounts for energy up to Nyquist freq)
    mag_sq = np.real(fhat * np.conj(fhat))
    
    # Compensate for negative frequency bins (Parseval's theorem)
    if N % 2 == 0:
        mag_sq[1:-1] *= 2 
    else:
        mag_sq[1:] *= 2
    
    # 4. Power Calculation
    # P_avg = sum(x^2)/N.  In freq domain: sum(|X|^2)/N^2
    power_rms = mag_sq / (N**2)
    
    # Convert to power using Ohm's law (P = V^2 / R)
    # If trace is mV, power_rms is mV^2. 
    # Watts = (Volts^2) / R = (mV * 1e-3)^2 / R
    power_Watts = (power_rms * 1e-6) / imp_ohm
    
    # 5. Normalize by Frequency Bin Width -> Density (Watts/Hz)
    df = 1 / (N * dt_sec)
    PSD_Watts_Hz = power_Watts / df
    
    # 6. Convert to dBm/Hz
    # dBm = 10 * log10(P_watts / 1mW)
    # Avoid log(0)
    with np.errstate(divide='ignore'):
        PSD_dBm_Hz = 10 * np.log10(PSD_Watts_Hz / 1e-3)
        
    return PSD_dBm_Hz