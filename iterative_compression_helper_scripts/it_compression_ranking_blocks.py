import pandas as pd

def calculate_pruning_ranking(cmmd_single, cmmd_double, 
                              idx_pruned_s, ratios_s, 
                              idx_pruned_d, ratios_d):
    # Constants
    PARAM_DOUBLE = 340.0  # Mio
    PARAM_SINGLE = 141.0  # Mio
    PRUNE_FACTOR = 0.60   # Every block is pruned by another 60%
    
    # Convert lists to dictionaries for fast lookup
    pruned_single_map = dict(zip(idx_pruned_s, ratios_s))
    pruned_double_map = dict(zip(idx_pruned_d, ratios_d))
    
    data = []

    # Process Double Blocks (ID 1-18 as per your metrics)
    # Range starts at 1 to match your provided CMMD list index
    for i in range(len(cmmd_double)):
        cmmd = cmmd_double[i]
        ratio = pruned_double_map.get(i, 0.0)
        
        current_params = PARAM_DOUBLE * (1 - ratio)
        params_to_remove = current_params * PRUNE_FACTOR
        
        efficiency = cmmd / params_to_remove if params_to_remove > 0 else float('inf')
        
        data.append({
            "Type": "Double", "ID": i, "CMMD": cmmd, 
            "Prev_Ratio": ratio, "Delta_Mio": params_to_remove, "Efficiency": efficiency
        })

    # Process Single Blocks (ID 0-37)
    for i, cmmd in enumerate(cmmd_single):
        ratio = pruned_single_map.get(i, 0.0)
        
        current_params = PARAM_SINGLE * (1-ratio)
        params_to_remove = current_params * PRUNE_FACTOR
        
        efficiency = cmmd / params_to_remove if params_to_remove > 0 else float('inf')
        
        data.append({
            "Type": "Single", "ID": i, "CMMD": cmmd, 
            "Prev_Ratio": ratio, "Delta_Mio": params_to_remove, "Efficiency": efficiency
        })

    # Create DataFrame and sort by Efficiency (Lower is Better)
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by="Efficiency", ascending=True).reset_index(drop=True)
    return df_sorted

# --- INPUT AREA ---

# 1. Paste your CMMD scores here
cmmd_single_scores  = [
    0.01227856, 0.01454353, 0.01347065, 0.01525879, 0.01823902, 
    0.01406670, 0.01609325, 0.01692772, 0.01502037, 0.01931190, 
    0.01716614, 0.01716614, 0.01561642, 0.01406670, 0.01788139, 
    0.01478195, 0.01645088, 0.01895428, 0.01490116, 0.01513958, 
    0.01835823, 0.01680851, 0.01430511, 0.01978874, 0.01895428, 
    0.01680851, 0.01537800, 0.01633167, 0.01525879, 0.01871586, 
    0.01323223, 0.01621246, 0.01788139, 0.01275539, 0.01621246, 
    0.02312660, 0.01752377, 0.01585484
]

# Note: Double index 0 is empty/dummy based on your data
cmmd_double_scores  = [
    0.02133846, 0.01907349, 0.94139576, 0.01370907, 0.01525879, 
    0.01370907, 0.01478195, 0.01513958, 0.01382828, 0.01680851, 
    0.01597404, 0.01609325, 0.01609325, 0.01299381, 0.01490116, 
    0.01454353, 0.01358986, 0.01370907, 0.01490116
]
# 2. Pruned Double Blocks
doubles_idx = []
doubles_rat = []

# 3. Pruned Single Blocks
singles_idx = []
singles_rat = []

# --- EXECUTION ---
ranking = calculate_pruning_ranking(cmmd_single_scores, cmmd_double_scores, 
                                    singles_idx, singles_rat, 
                                    doubles_idx, doubles_rat)

print(ranking.to_string())
print(list(ranking["ID"]))