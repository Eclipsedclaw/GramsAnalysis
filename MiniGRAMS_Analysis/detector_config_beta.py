"""
detector_config_beta.py
Configuration script for MiniGRAMS & MicroGRAMS & 3-Board LArTPC.
BETA VERSION: Includes all physical channels (including removed/shorted CSPs) 
for DAQ noise floor and baseline sanity checks.
"""

def detect_run_mode(path):
    path_lower = path.lower()
    if "osaka" in path_lower: return 'OSAKA'
    if "ups1" in path_lower: return 'UPS1'
    if "ups2" in path_lower: return 'UPS2'
    if "ups" in path_lower: return 'UPS1' # Default Legacy
    if "minig" in path_lower: return 'MINIG' # Permanent 3-board setup
    if "light" in path_lower: return 'LIGHT_ONLY'
    return 'LIGHT_ONLY'

def get_channel_map(run_config_name):
    mapping = {"VUV": [], "VIS": [], "Charge": [], 
               "Charge_A": [], "Charge_B": [], "Charge_C": [], 
               "Charge_D": [], "Charge_E": [], "Charge_F": []}
    
    cfg = run_config_name.upper()
    
    # --- OSAKA (Run5 / Juggernaut) ---
    if "OSAKA" in cfg:
        mapping["Charge_A"] = list(range(0, 14))
        mapping["Charge_B"] = list(range(14, 28))
        mapping["VUV"] = [28, 29]
        mapping["VIS"] = [30, 31]
        mapping["Charge_C"] = list(range(32, 47))
        mapping["Charge_D"] = list(range(47, 61))
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + mapping["Charge_D"]

    # --- UPS1 (Green Table / AB Board) ---
    elif "UPS1" in cfg:
        mapping["VUV"] = [0, 1]
        mapping["VIS"] = [2, 3]
        mapping["Charge_A"] = list(range(4, 19))
        mapping["Charge_B"] = list(range(19, 33))
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"]

    # --- UPS2 (Blue Table / CD Board) ---
    elif "UPS2" in cfg:
        mapping["VUV"] = [0, 1]
        mapping["VIS"] = [2, 3]
        mapping["Charge_C"] = list(range(4, 19))
        mapping["Charge_D"] = list(range(19, 33))
        mapping["Charge"] = mapping["Charge_C"] + mapping["Charge_D"]
        
    # --- MiniG Permanent 6 CSP Bank Config (BETA - ALL CHANNELS INCLUDED) ---
    elif "MINIG" in cfg:
        # Global Indexing: 
        # Master CAEN (Board 72) = 0-31
        # Slave 1 CAEN (Board 85) = 32-95
        # Slave 2 CAEN (Board 75) = 96-159
        
        off_s1 = 32
        off_s2 = 96
        
        # --- Light (Master CAEN) ---
        mapping["VUV"] = [i for i in range(32) if i % 2 == 0]
        mapping["VIS"] = [i for i in range(32) if i % 2 != 0]
        
        # --- Overflow Light (Slave 2 CAEN Ch 60-63) ---
        mapping["VUV"].extend([off_s2 + 60, off_s2 + 62])
        mapping["VIS"].extend([off_s2 + 61, off_s2 + 63])
        
        # --- Charge (Slave 1 & Slave 2) ---
        # Bank A: Slave 1 Ch 0-14 (Includes removed SIG14 / Ch 14)
        mapping["Charge_A"] = [off_s1 + i for i in range(0, 15)]
        
        # Bank B: Slave 1 Ch 15-28 (Includes removed SIG2 / Ch 17)
        mapping["Charge_B"] = [off_s1 + i for i in range(15, 29)]
        
        # Bank C: Slave 1 Ch 29-43 (Includes removed SIG0 / Ch 29)
        mapping["Charge_C"] = [off_s1 + i for i in range(29, 44)]
        
        # Bank D: Slave 1 Ch 44-57
        mapping["Charge_D"] = [off_s1 + i for i in range(44, 58)]
        
        # Bank E: Slave 1 Ch 58-63 AND Slave 2 Ch 0-8
        mapping["Charge_E"] = [off_s1 + i for i in range(58, 64)] + [off_s2 + i for i in range(0, 9)]
        
        # Bank F: Slave 2 Ch 9-22 (Includes removed SIG0 / Ch 9 and SIG11 / Ch 20)
        mapping["Charge_F"] = [off_s2 + i for i in range(9, 23)]
        
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