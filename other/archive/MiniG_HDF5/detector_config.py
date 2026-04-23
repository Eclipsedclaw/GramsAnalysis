"""
detector_config.py
Configuration script for MiniGRAMS LArTPC channel mapping
"""

def get_channel_map(run_config_name):
    """
    Returns a dictionary of channel lists based on the run configuration name.
    
    Structure:
    {
        "VUV": [list of channels],
        "VIS": [list of channels],
        "Charge": [list of channels]
    }
    """
    mapping = {"VUV": [], "VIS": [], "Charge": []}
    
    # Normalize input string
    cfg = run_config_name.lower()
    
    # --- CONFIGS ---
    
    # LIGHT ONLY (Config 1 & 2)
    # User confirms: All 32 channels are light. CAEN Even=VUV, CAEN Odd=VIS.
    if "light_only" in cfg or "config_1" in cfg or "config_2" in cfg:
        light_range = range(0, 32)
        charge_range = []
        
    # COMBO 24 Light + 8 Charge on a CAEN V7230
    # Light: 0-23 -> Even/Odd rule applies
    # Charge: 24-31 (CSP Bank A & B)
    elif "combo" in cfg:
        light_range = range(0, 24)
        charge_range = range(24, 32)
        
    # 3. UNKNOWN / FALLBACK
    else:
        print(f"[WARN] Unknown Config '{run_config_name}'. Returning empty map.")
        return mapping

    # --- GENERATE LISTS ---
    
    # Apply Even/Odd rule for light channels
    for ch in light_range:
        if ch % 2 == 0:
            mapping["VUV"].append(ch)
        else:
            mapping["VIS"].append(ch)
            
    # Assign charge channels
    mapping["Charge"] = list(charge_range)
    
    return mapping

def get_channel_type(ch_idx, run_config_name): 
    """
    Reverse lookup: Given a channel ID, what is it?
    """
    m = get_channel_map(run_config_name)
    if ch_idx in m["VUV"]: return "VUV"
    if ch_idx in m["VIS"]: return "VIS"
    if ch_idx in m["Charge"]: return "Charge"
    return "Unknown"