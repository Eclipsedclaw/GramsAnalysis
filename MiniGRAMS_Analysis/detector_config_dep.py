"""
detector_config.py
Configuration script for MiniGRAMS LArTPC channel mapping.
Centralizes logic for run mode detection and physical plotting maps.
"""

def detect_run_mode(path):
    """
    Scans the target path string to identify the run configuration.
    Returns: 'COMBO' or 'LIGHT_ONLY'
    """
    path_lower = path.lower()
    if "combo" in path_lower:
        return 'COMBO'
    elif "light_only" in path_lower or "scintdata" in path_lower or "light-sig" in path_lower:
        return 'LIGHT_ONLY'
    else:
        # Default fallback
        print("[WARN] Run Config Unidentified. Defaulting to LIGHT_ONLY.")
        return 'LIGHT_ONLY'

def get_channel_map(run_config_name):
    """
    Returns lists of channels for VUV, VIS, and Charge.
    Used for Mass Plotting / PSD.
    """
    mapping = {"VUV": [], "VIS": [], "Charge": []}
    cfg = run_config_name.lower()
    
    if "light_only" in cfg or "config_1" in cfg or "config_2" in cfg:
        light_range = range(0, 32)
        charge_range = []
    elif "combo" in cfg:
        light_range = range(0, 24)
        charge_range = range(24, 32)
    #################
    elif "UPS" in cfg:
        # UPS Config: 4 Light (30-33, 29 Charge (34-62)
        light_range = range(30,34)
        charge_range = range(34, 63)
    #################
    else:
        # Fallback
        light_range = range(0, 32)
        charge_range = []


    for ch in light_range:
        if ch % 2 == 0: mapping["VUV"].append(ch)
        else: mapping["VIS"].append(ch)
            
    mapping["Charge"] = list(charge_range)
    return mapping

def get_channel_type(ch_idx, run_config_name): 
    m = get_channel_map(run_config_name)
    if ch_idx in m["VUV"]: return "VUV"
    if ch_idx in m["VIS"]: return "VIS"
    if ch_idx in m["Charge"]: return "Charge"
    return "Unknown"

def get_physical_map(mode):
    """
    Returns the Physical Location Mapping for the 6x6 Grid Plot.
    Returns Dictionary: { CAEN_CH : {'mb': INT, 'pcb': INT, 'label': STR} }
    """
    mapping = {}
    
    if mode == 'COMBO':
        # --- COMBO MODE (24 SiPMs + 8 Charge) ---
        # MB1 (Mid): Full 1-12 (CAEN 0-11)
        for i in range(12):
            mapping[i] = {'mb': 1, 'pcb': i+1, 'label': f"MB1-CH{i+1}"}
            
        # MB2 (Bot): Partial 1-8 (CAEN 12-19)
        for i in range(8):
            caen = 12 + i
            mapping[caen] = {'mb': 2, 'pcb': i+1, 'label': f"MB2-CH{i+1}"}
            
        # MB3 (Top): Center Only 5-8 (CAEN 20-23)
        for i in range(4):
            caen = 20 + i
            pcb_ch = 5 + i
            mapping[caen] = {'mb': 3, 'pcb': pcb_ch, 'label': f"MB3-CH{pcb_ch}"}

    elif mode == 'LIGHT_ONLY':
        # --- LIGHT ONLY MODE (32 SiPMs) ---
        # MB1 (Mid): Full 1-12 (CAEN 0-11)
        for i in range(12):
            mapping[i] = {'mb': 1, 'pcb': i+1, 'label': f"MB1-CH{i+1}"}
            
        # MB2 (Bot): Full 1-12 (CAEN 12-23)
        for i in range(12):
            caen = 12 + i
            mapping[caen] = {'mb': 2, 'pcb': i+1, 'label': f"MB2-CH{i+1}"}
            
        # MB3 (Top): Partial 1-8 (CAEN 24-31)
        # Assuming MB3 is fully populated or taking the upper block
        for i in range(8):
            caen = 24 + i
            # Mapping 24-31 to PCB 1-8 or 5-12? 
            # Defaulting to 1-8 for generic Light Only unless specified otherwise
            mapping[caen] = {'mb': 3, 'pcb': i+1, 'label': f"MB3-CH{i+1}"}
            
    return mapping