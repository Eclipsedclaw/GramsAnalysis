import h5py
import numpy as np
import sys

fname = sys.argv[1]
with h5py.File(fname, 'r') as f:
    # Look at Channel 32 (A Charge Channel)
    # 0-based index. 
    # If using Osaka map, Ch32 is physically Ch34 (Bank C).
    
    std_vals = f['baseline_std_mV'][:, 32] 
    
    print(f"--- SANITY CHECK: {fname} ---")
    print(f"Channel 32 RMS (First 5 events): {std_vals[:5]}")
    print(f"Average RMS: {np.mean(std_vals):.4f} mV")
    
    if np.mean(std_vals) < 0.2:
        print("❌ STATUS: STILL LOW (Pre-Trigger Only)")
    else:
        print("✅ STATUS: UPDATED (Full Trace Drift Included)")