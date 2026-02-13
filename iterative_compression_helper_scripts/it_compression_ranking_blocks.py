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
cmmd_single_scores =[
    0.012278556823730469, 0.014543533325195312, 0.013470649719238281, 0.0152587890625, 
    0.01823902130126953, 0.014066696166992188, 0.01609325408935547, 0.016927719116210938, 
    0.015020370483398438, 0.019311904907226562, 0.0171661376953125, 0.0171661376953125, 
    0.015616416931152344, 0.014066696166992188, 0.017881393432617188, 0.014781951904296875, 
    0.016450881958007812, 0.01895427703857422, 0.014901161193847656, 0.015139579772949219, 
    0.018358230590820312, 0.016808509826660156, 0.01430511474609375, 0.019788742065429688, 
    0.01895427703857422, 0.016808509826660156, 0.015377998352050781, 0.01633167266845703, 
    0.0152587890625, 0.018715858459472656, 0.013232231140136719, 0.01621246337890625, 
    0.017881393432617188, 0.012755393981933594, 0.01621246337890625, 0.023126602172851562, 
    0.017523765563964844, 0.015854835510253906
]

# Note: Double index 0 is empty/dummy based on your data
cmmd_double_scores  = [
    0.021338462829589844, 0.019073486328125, 0.9413957595825195, 0.013709068298339844, 
    0.0152587890625, 0.013709068298339844, 0.014781951904296875, 0.015139579772949219, 
    0.013828277587890625, 0.016808509826660156, 0.015974044799804688, 0.01609325408935547, 
    0.01609325408935547, 0.012993812561035156, 0.014901161193847656, 0.014543533325195312, 
    0.013589859008789062, 0.013709068298339844, 0.014901161193847656
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