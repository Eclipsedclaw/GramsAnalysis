"""
detector_config.py
Configuration script for MiniGRAMS & MicroGRAMS.
Updated: Explicit 'LIGHT_ONLY' block added to support standalone Fast CAEN acquisitions.
Fix: Matches exact Even/Odd VUV/VIS physical mapping for MB1, MB2, and MB3.
"""

def detect_run_mode(path):
    path_lower = path.lower()
    if "osaka" in path_lower: return 'OSAKA'
    if "ups1" in path_lower: return 'UPS1'
    if "ups2" in path_lower: return 'UPS2'
    if "ups" in path_lower: return 'UPS1' 
    if "run9" in path_lower: return 'MINIG_RUN9_BANDAID' 
    if "minig" in path_lower: return 'MINIG' 
    if "light" in path_lower: return 'LIGHT_ONLY'
    if 'lightstudy' in path_lower: return 'MICROG_PURITY_STUDY'
    return 'LIGHT_ONLY'

def get_channel_map(run_config_name):
    mapping = {"VUV": [], "VIS": [], "Charge": [], 
               "Charge_A": [], "Charge_B": [], "Charge_C": [], 
               "Charge_D": [], "Charge_E": [], "Charge_F": []}
    
    cfg = run_config_name.upper()
    
    # --- LIGHT ONLY (Standalone Fast CAEN Run) ---
    if "LIGHT_ONLY" in cfg:
        # Exactly 32 channels on Board 72. Evens = VUV, Odds = VIS.
        mapping["VUV"] = [i for i in range(32) if i % 2 == 0]
        mapping["VIS"] = [i for i in range(32) if i % 2 != 0]
        # Charge lists remain empty automatically.

    # --- OSAKA (Run5 / Juggernaut) ---
    elif "OSAKA" in cfg:
        mapping["Charge_A"] = list(range(0, 14))
        mapping["Charge_B"] = list(range(14, 28))
        mapping["VUV"] = [28, 29]
        mapping["VIS"] = [30, 31]
        mapping["Charge_C"] = list(range(32, 46))
        mapping["Charge_D"] = list(range(46, 60))
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + mapping["Charge_D"]
        
    # --- UPS1 & UPS2 ---
    elif "UPS1" in cfg or "UPS2" in cfg:
        mapping["VUV"] = [0, 1]
        mapping["VIS"] = [2, 3]
        mapping["Charge_A"] = list(range(4, 18))
        mapping["Charge_B"] = list(range(18, 32))
        mapping["Charge_C"] = list(range(32, 46))
        mapping["Charge_D"] = list(range(46, 60))
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + mapping["Charge_D"]

    # --- MINIG RUN 9 BANDAID (DB50 Swap) ---
    elif "MINIG_RUN9_BANDAID" in cfg:
        off_s1 = 32  # Board 85
        off_s2 = 96  # Board 75
        
        # LIGHT: Fast CAEN (Board 72) + Slave 2 overflow
        mapping["VUV"] = [i for i in range(32) if i % 2 == 0] + [off_s2 + 60, off_s2 + 62]
        mapping["VIS"] = [i for i in range(32) if i % 2 != 0] + [off_s2 + 61, off_s2 + 63]
        
        # X-DIRECTION (Banks A, B, C)
        mapping["Charge_A"] = [off_s1 + i for i in range(0, 15)]           
        mapping["Charge_B"] = [off_s1 + i for i in range(15, 29)]          
        mapping["Charge_C"] = [off_s1 + i for i in range(58, 64)] + [off_s2 + i for i in range(0, 9)]            
        
        # Y-DIRECTION (Banks D, E, F)
        mapping["Charge_D"] = [off_s2 + i for i in range(9, 23)]           
        mapping["Charge_E"] = [off_s1 + i for i in range(29, 44)]          
        mapping["Charge_F"] = [off_s1 + i for i in range(44, 58)]          
        
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + \
                            mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]

    # --- MINIG (Permanent 160-Channel 3-Board Layout) ---
    elif "MINIG" in cfg:
        off_s1 = 32  
        off_s2 = 96  
        
        mapping["VUV"] = [i for i in range(32) if i % 2 == 0] + [off_s2 + 60, off_s2 + 62]
        mapping["VIS"] = [i for i in range(32) if i % 2 != 0] + [off_s2 + 61, off_s2 + 63]
        
        mapping["Charge_A"] = [off_s1 + i for i in range(0, 15)]
        mapping["Charge_B"] = [off_s1 + i for i in range(15, 29)]
        mapping["Charge_C"] = [off_s1 + i for i in range(29, 44)]
        mapping["Charge_D"] = [off_s1 + i for i in range(44, 58)]
        mapping["Charge_E"] = [off_s1 + i for i in range(58, 64)] + [off_s2 + i for i in range(0, 9)]
        mapping["Charge_F"] = [off_s2 + i for i in range(9, 23)]
        
        mapping["Charge"] = mapping["Charge_A"] + mapping["Charge_B"] + mapping["Charge_C"] + \
                            mapping["Charge_D"] + mapping["Charge_E"] + mapping["Charge_F"]

    return mapping

def get_channel_type(global_ch, run_config_name):
    """
    Reverse-lookup function. Takes an absolute channel index and returns its physical label.
    Used by the Event Builder to stamp HDF5 metadata arrays.
    """
    ch_map = get_channel_map(run_config_name)
    for bank_name in ["Charge_A", "Charge_B", "Charge_C", "Charge_D", "Charge_E", "Charge_F", "VUV", "VIS"]:
        if global_ch in ch_map.get(bank_name, []):
            return bank_name
    if global_ch in ch_map.get("Charge", []):
        return "Charge"
    return "NC"