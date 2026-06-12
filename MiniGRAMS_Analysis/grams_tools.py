"""
grams_tools.py
LArTPC Analysis Toolkit
-----------------------
Library for subsytem-dependent baseline correction algorithms. 

Author: Jonathan LeyVa
Date: Jan 15,2026
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

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


# --- PULSE SHAPE DISCRIMINATION (PSD) ---

def calculate_fprompt(trace, res_ns, trigger_idx, prompt_ns=90, total_us=7.0, pre_trigger_ns=20):
    """
    Calculates F_prompt and total integrated iharge for a single waveform.
    
    Args:
        trace (array): Voltage data (mV).
        res_ns (float): Sampling resolution (ns).
        trigger_idx (int): Index where the trigger occurred.
        prompt_ns (float): Duration of the prompt window (ns).
        total_us (float): Duration of the total integration window (us).
        pre_trigger_ns (float): Buffer before trigger point to ensure full rising edge is caught (ns).
        
    Returns:
        tuple: (f_prompt_ratio, total_integral_mV_ns)
    """
    # Convert times to sample counts
    # Pre-trigger buffer
    pre_samples = int(pre_trigger_ns / res_ns)
    
    # Prompt window width
    prompt_samples = int(prompt_ns / res_ns)
    
    # Total window width
    total_samples = int((total_us * 1e3) / res_ns)
    
    # 2. Define Indices
    # Start integration slightly before trigger
    # max() prevents negative indices
    start_idx = max(0, trigger_idx - pre_samples)
    
    # End of Prompt
    end_prompt_idx = start_idx + prompt_samples
    
    # End of Total
    end_total_idx = start_idx + total_samples
    
    # Safety Check: Don't run off the array
    if end_total_idx > len(trace):
        end_total_idx = len(trace)
    if end_prompt_idx > end_total_idx:
        end_prompt_idx = end_total_idx

    # 3. Integration (Riemann Sum)
    # Integral = Sum(Voltage) * dt
    # dt = res_ns
    
    # Slicing
    prompt_slice = trace[start_idx:end_prompt_idx]
    total_slice = trace[start_idx:end_total_idx]
    
    # F_prompt Calculation
    Q_prompt = np.abs(np.sum(prompt_slice)) * res_ns
    Q_total = np.abs(np.sum(total_slice)) * res_ns
    
    # Avoid Divide by Zero
    if Q_total == 0:
        return 0.0, 0.0
        
    f_prompt = Q_prompt / Q_total
    
    return f_prompt, Q_total

def find_charge_crest_time(trace, res_ns, pre_trigger_us=0.0):
    """
    Determines the peaking time of a charge waveform in TPC without anode mesh 
    
    Algorithm:
    1. Smooth trace (Gaussian filter).
    2. Compute Discrete Derivative (Ramo Current).
    3. Find the 'end' of the induced current pulse.
    
    Args:
        trace (array): Voltage trace (mV).
        res_ns (float): Resolution in ns.
        pre_trigger_us (float): Time offset to subtract.
        
    Returns:
        float: Drift time in microseconds (us) relative to trigger.
               Returns None if no significant pulse found.
    """
    # 1. Smooth the ramp to remove digitizer noise
    # Sigma=4.0 samples is roughly 8-10ns smoothing, good for slow drifts
    smooth_trace = gaussian_filter1d(trace, sigma=4.0)
    
    # 2. Derivative (V/s -> Current)
    deriv = np.gradient(smooth_trace)
    
    # 3. Thresholding
    # We look for a significant current pulse.
    # Noise floor estimation on first 20 samples of derivative
    noise_floor = np.std(deriv[:20]) * 5.0 
    
    # Check if we have a pulse at all
    if np.max(deriv) < noise_floor:
        return None
    
    # 4. Find the Peak Current (Inflection point of ramp)
    peak_idx = np.argmax(deriv)
    
    # 5. Find the "Crest" (End of current pulse)
    # We search forward from the peak until the derivative drops back 
    # below a fraction of the peak (e.g., 10%) or hits the noise floor.
    cutoff_thresh = max(deriv[peak_idx] * 0.10, noise_floor)
    
    crest_idx = peak_idx
    for i in range(peak_idx, len(deriv)):
        if deriv[i] < cutoff_thresh:
            crest_idx = i
            break
            
    # Convert to Time (us)
    t_ns = crest_idx * res_ns
    t_us = (t_ns / 1000.0) - pre_trigger_us
    
    return t_us

import struct

def calculate_daq_offsets(master_file, slave1_file, slave2_file, search_depth=150):
    """
    Calculates the true Ethernet start stagger for stochastic particle data.
    Uses Pairwise Difference Voting to find the true coincidence offset.
    """
    import struct
    import numpy as np
    
    def get_ts_array(filepath, n=search_depth):
        ts_list = []
        header_format = "<IQIQi"
        header_size = struct.calcsize(header_format)
        try:
            with open(filepath, 'rb') as f:
                for _ in range(n):
                    h_bytes = f.read(header_size)
                    if len(h_bytes) < header_size: break
                    _, ts, n_samp, _, n_chan = struct.unpack(header_format, h_bytes)
                    ts_list.append(ts)
                    # Skip payload
                    f.seek((n_chan * 2) + (n_chan * n_samp * 4), 1)
        except Exception as e:
            print(f"Warning reading {filepath} for offsets: {e}")
        return np.array(ts_list, dtype=np.int64)

    m_ts = get_ts_array(master_file)
    s1_ts = get_ts_array(slave1_file)
    s2_ts = get_ts_array(slave2_file)
    
    def find_true_offset(master, slave):
        if len(master) == 0 or len(slave) == 0: return 0
        
        # Calculate all possible differences (Slave - Master)
        diffs = slave[:, None] - master[None, :]
        diffs = diffs.flatten()
        
        # Bin the differences (10-tick bins) to account for slight phase jitter
        rounded = np.round(diffs / 10.0) * 10.0
        
        # Find the most frequent difference
        values, counts = np.unique(rounded, return_counts=True)
        best_bin = values[np.argmax(counts)]
        
        # Get the exact median of the raw differences within that winning bin
        best_diffs = diffs[np.abs(diffs - best_bin) <= 10]
        return int(np.median(best_diffs))

    offset_75 = find_true_offset(m_ts, s1_ts)
    offset_85 = find_true_offset(m_ts, s2_ts)
    
    return {'75': offset_75, '85': offset_85}

# --- 2D HEATMAP ALGORITHMS ---

def calc_geometric_mean_pos_integral(trace_x, trace_y, res_ns):
    """
    Calculates the geometric mean of the positive-only Riemann sum.
    Isolates the rising-edge charge injection response.
    """
    dt_us = res_ns / 1000.0
    
    # Mask out negative values, then integrate
    int_x = np.sum(trace_x[trace_x > 0]) * dt_us
    int_y = np.sum(trace_y[trace_y > 0]) * dt_us
    
    return np.sqrt(int_x * int_y)

def calc_geometric_mean_neg_integral(trace_x, trace_y, res_ns):
    """
    Calculates the geometric mean of the negative-only Riemann sum.
    Isolates the falling-edge discharge response and flips the sign for easy viewing.
    """
    dt_us = res_ns / 1000.0
    
    # Mask out positive values, integrate, and take absolute value
    int_x = np.abs(np.sum(trace_x[trace_x < 0])) * dt_us
    int_y = np.abs(np.sum(trace_y[trace_y < 0])) * dt_us
    
    return np.sqrt(int_x * int_y)

def calc_geometric_mean_peak(trace_x, trace_y):
    """
    Calculates the geometric mean of the peak absolute amplitudes.
    """
    peak_x = np.max(np.abs(trace_x))
    peak_y = np.max(np.abs(trace_y))
    
    return np.sqrt(peak_x * peak_y)