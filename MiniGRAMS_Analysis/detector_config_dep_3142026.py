"""
detector_config.py
Configuration script for MiniGRAMS & MicroGRAMS & 3-Board LArTPC.
"""

def detect_run_mode(path):
    path_lower = path.lower()
    if "osaka" in path_lower: return 'OSAKA'
    if "ups1" in path_lower: return 'UPS1'
    if "ups2" in path_lower: return 'UPS2'
    if "ups" in path_lower: return 'UPS1' # Default Legacy
    if "minig" in path_lower: return 'MINIG' # Added for new 3-board setup
    if "light" in path_lower: return 'LIGHT_ONLY'
    return 'LIGHT_ONLY'

def get_channel_map(run_config_name):
    # Added Charge_E and Charge_F to the initialization dictionary
    mapping = {"VUV": [], "VIS": [], "Charge": [], 
               "Charge_A": [], "Charge_B": [], "Charge_C": [], 
               "Charge_D": [], "Charge_E": [], "Charge_F": []}
    
    cfg = run_config_name.upper()
    
    # --- OSAKA (Run5 / Juggernaut) ---
    if "OSAKA" in cfg:
        # Data Stream starts at CAEN Ch 2 (Index shift -2)
        mapping["Charge_A"] = list(range(0, 14))   # Phys 2-15
        mapping["Charge_B"] = list(range(14, 28))  # Phys 16-29
        mapping["VUV"] = [28, 29]
        mapping["VIS"] = [30, 31]
        mapping["Charge_C"] = list(range(32, 47))  # Phys 34-48
        mapping["Charge_D"] = list(range(47, 61))  # Phys 49-62
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + mapping["Charge_D"]

    # --- UPS1 (Green Table / AB Board) ---
    elif "UPS1" in cfg:
        # Data 0-3 -> SiPMs (Phys 30-33)
        mapping["VUV"] = [0, 1]  # Phys 30, 31
        mapping["VIS"] = [2, 3]  # Phys 32, 33
        
        # Data 4-18 -> Bank A (Phys 1-15)
        mapping["Charge_A"] = list(range(4, 19))
        
        # Data 19-32 -> Bank B (Phys 16-29)
        mapping["Charge_B"] = list(range(19, 33))
        
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"]

    # --- UPS2 (Blue Table / CD Board) ---
    elif "UPS2" in cfg:
        # Data 0-3 -> SiPMs (Phys 30-33)
        mapping["VUV"] = [0, 1]  # Phys 30, 31
        mapping["VIS"] = [2, 3]  # Phys 32, 33
        
        # Data 4-18 -> Bank C (Phys 34-48)
        mapping["Charge_C"] = list(range(4, 19))
        
        # Data 19-32 -> Bank D (Phys 49-62)
        mapping["Charge_D"] = list(range(19, 33))
        
        mapping["Charge"] = mapping["Charge_C"] + mapping["Charge_D"]
        
    # --- MiniG First 6 CSP Bank Config ---
    # elif "MINIG" in cfg:
    #     # Global Indexing: 
    #     # Fast CAEN (72) = 0-31
    #     # Slow CAEN 1 (75) = 32-95
    #     # Slow CAEN 2 (85) = 96-159
        
    #     off75 = 32
    #     off85 = 96
        
    #     # --- Light (CAEN 72) ---
    #     mapping["VUV"] = [i for i in range(32) if i % 2 == 0]
    #     mapping["VIS"] = [i for i in range(32) if i % 2 != 0]
        
    #     # --- Charge (CAEN 75 & 85) ---
    #     # Bank A (PID 75, Ch 0-14) - Removed SIG14 (Ch 14)
    #     mapping["Charge_A"] = [off75 + i for i in range(0, 15) if i != 14]
        
    #     # Bank B (PID 75, Ch 15-28) - Removed SIG2 (Ch 17)
    #     mapping["Charge_B"] = [off75 + i for i in range(15, 29) if i != 17]
        
    #     # Bank C (PID 75, Ch 29-43) - Removed SIG0 (Ch 29)
    #     mapping["Charge_C"] = [off75 + i for i in range(29, 44) if i != 29]
        
    #     # Bank D (PID 75, Ch 44-57)
    #     mapping["Charge_D"] = [off75 + i for i in range(44, 58)]
        
    #     # Bank E (PID 75, Ch 58-63) AND (PID 85, Ch 0-8)
    #     mapping["Charge_E"] = [off75 + i for i in range(58, 64)] + [off85 + i for i in range(0, 9)]
        
    #     # Bank F (PID 85, Ch 9-22) - Removed SIG0 (Ch 9) and SIG11 (Ch 20)
    #     mapping["Charge_F"] = [off85 + i for i in range(9, 23) if i not in [9, 20]]
        
    #     mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + \
    #                         mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]
    
    # --- MiniG First 6 CSP Bank Config (GAr Run 6) ---
    elif "MINIG" in cfg:
        # Global Indexing: 
        # Fast CAEN (72) = 0-31
        # Slow CAEN 1 (75) = 32-95
        # Slow CAEN 2 (85) = 96-159
        
        off75 = 32
        off85 = 96
        
        # --- Light (CAEN 72) ---
        mapping["VUV"] = [i for i in range(32) if i % 2 == 0]
        mapping["VIS"] = [i for i in range(32) if i % 2 != 0]
        
        # --- Overflow Light (CAEN 85 Ch 60-63) ---
        mapping["VUV"].extend([off85 + 60, off85 + 62])
        mapping["VIS"].extend([off85 + 61, off85 + 63])
        
        # --- Charge (CAEN 75 & 85) ---
        mapping["Charge_A"] = [off75 + i for i in range(0, 15) if i != 14]
        mapping["Charge_B"] = [off75 + i for i in range(15, 29) if i != 17]
        mapping["Charge_C"] = [off75 + i for i in range(29, 44) if i != 29]
        mapping["Charge_D"] = [off75 + i for i in range(44, 58)]
        mapping["Charge_E"] = [off75 + i for i in range(58, 64)] + [off85 + i for i in range(0, 9)]
        mapping["Charge_F"] = [off85 + i for i in range(9, 23) if i not in [9, 20]]
        
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + \
                            mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]
            
    else:
        light_range = range(0, 32)
        for ch in light_range:
            if ch % 2 == 0: mapping["VUV"].append(ch)
            else: mapping["VIS"].append(ch)
            
    return mapping

def get_channel_type(ch_idx, run_config_name): 
    m = get_channel_map(run_config_name)
    if ch_idx in m["VUV"]: return "VUV"
    if ch_idx in m["VIS"]: return "VIS"
    if ch_idx in m["Charge"]: return "Charge"
    return "Unknown"