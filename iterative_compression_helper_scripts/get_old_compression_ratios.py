def get_old_ratios_from_ranking(best_40_blocks, 
                                 idx_d, rat_d, 
                                 idx_s, rat_s):
    """
    best_40_blocks: List of tuples [("Double", 16), ("Single", 16), ...]
    idx_d, rat_d: Current double block indices and their ratios
    idx_s, rat_s: Current single block indices and their ratios
    """
    
    # 1. Create Lookup Dictionaries for the current state
    # This allows us to find the ratio for any block instantly
    double_state = dict(zip(idx_d, rat_d))
    single_state = dict(zip(idx_s, rat_s))
    
    old_ratios_output = []
    
    # 2. Iterate through the ranking list
    for block_type, block_id in best_40_blocks:
        if block_type == "Double":
            # Get ratio if exists, otherwise 0.0
            ratio = double_state.get(block_id, 0.0)
        else:
            # Get ratio if exists, otherwise 0.0
            ratio = single_state.get(block_id, 0.0)
            
        old_ratios_output.append(ratio)
    
    return old_ratios_output

# --- INPUT DATA ---

# Your four state lists
doubles_idx = [13, 16, 5, 17, 3, 8, 15, 6, 14, 18, 7, 4, 10, 11, 12, 9, 1]
doubles_rat = [0.53, 0.5, 0.47, 0.455, 0.44, 0.425, 0.38, 0.365, 0.35, 0.335, 0.32, 0.305, 0.29, 0.245, 0.23, 0.215, 0.17]

singles_idx = [0, 33, 30, 2, 5, 13, 1, 15, 18, 8, 19, 3, 28, 26, 12, 37, 6]
singles_rat = [0.515, 0.485, 0.41, 0.395, 0.275, 0.26, 0.2, 0.185, 0.155, 0.14, 0.125, 0.11, 0.095, 0.08, 0.065, 0.05, 0.035]

# Your Ranking (the 40 best blocks in order)
# Format: (Type, ID)
ranking_order = [
    ("Double", 0), ("Double", 11), ("Double", 9), ("Double", 12), ("Double", 10),
    ("Double", 7), ("Double", 14), ("Double", 6), ("Double", 4), ("Double", 18),
    ("Double", 1), ("Double", 15), ("Double", 3), ("Double", 8), ("Double", 17),
    ("Double", 16), ("Double", 5), ("Single", 16), ("Single", 23), ("Single", 22),
    ("Double", 13), ("Single", 14), ("Single", 26), ("Single", 31), ("Single", 17),
    ("Single", 21), ("Single", 20), ("Single", 6), ("Single", 28), ("Single", 11),
    ("Single", 1), ("Single", 37), ("Single", 24), ("Single", 27), ("Single", 19),
    ("Single", 4), ("Single", 3), ("Single", 7), ("Single", 12), ("Single", 25)
]

# --- EXECUTION ---
result = get_old_ratios_from_ranking(ranking_order, doubles_idx, doubles_rat, singles_idx, singles_rat)

print("Python List of Old Ratios:")
print(result)