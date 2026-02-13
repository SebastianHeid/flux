import numpy as np

def calculate_advanced_linear_compression(
    all_blocks_ranked, 
    num_top_blocks, 
    prev_compression_ratios, 
    block_param_counts,
    start_ratio, 
    target_to_remove
):
    """
    all_blocks_ranked: Liste der Strings ["single_block_11", "double_block_0", ...]
    num_top_blocks: Wie viele der besten Blöcke sollen gepruned werden (z.B. 40)
    prev_compression_ratios: Dict {block_name: ratio} der bereits erfolgten Kompression
    block_param_counts: Dict {block_name: count} der ORIGINALEN Parameteranzahl
    start_ratio: Gewünschte Ratio für den allerbesten Block (Rang 0)
    target_to_remove: Gesamtanzahl der Parameter, die in DIESEM Schritt weg sollen
    """
    
    selected_blocks = all_blocks_ranked[:num_top_blocks]
    n = len(selected_blocks)
    
    current_params = []
    for name in selected_blocks:
        p_orig = block_param_counts.get(name, 0)
        p_prev_ratio = prev_compression_ratios.get(name, 0.0)
        current_params.append(p_orig * (1 - p_prev_ratio))
    
    numerator = sum(p * start_ratio for p in current_params) - target_to_remove
    denominator = sum(p * i for i, p in enumerate(current_params))
    
    if denominator == 0:
        x = 0
    else:
        x = numerator / denominator

    new_ratios_dict = {}
    for i, name in enumerate(selected_blocks):
        ratio = start_ratio - (i * x)
        new_ratios_dict[name] = max(0.0, ratio)
        
    return x, new_ratios_dict



NUM_DOUBLE_BLOCKS = 19  
NUM_SINGLE_BLOCKS = 38  


PARAM_DOUBLE = 340_000_000
PARAM_SINGLE = 141_000_000


param_counts = {}


for i in range(NUM_DOUBLE_BLOCKS):
    param_counts[f"double_block_{i}"] = PARAM_DOUBLE

for i in range(NUM_SINGLE_BLOCKS):
    param_counts[f"single_block_{i}"] = PARAM_SINGLE

# ----------------------------- Parameters to specify ------------------------------------------------
ranked_list = ['double_block_14', 'double_block_5', 'double_block_9', 'double_block_10', 'double_block_13', 'double_block_12', 'double_block_7', 'double_block_6', 'double_block_11', 'double_block_17', 'double_block_4', 'double_block_15', 'single_block_27', 'double_block_3', 'double_block_1', 'single_block_28', 'single_block_22', 'single_block_14', 'single_block_23', 'double_block_18', 'double_block_16', 'single_block_0', 'single_block_20', 'single_block_31', 'single_block_18', 'single_block_4', 'single_block_25', 'single_block_13', 'single_block_21', 'double_block_8', 'single_block_11', 'single_block_37', 'single_block_17', 'single_block_34', 'single_block_12', 'single_block_19', 'double_block_0', 'single_block_24', 'single_block_3', 'single_block_26', 'single_block_10', 'single_block_32', 'single_block_36', 'single_block_16', 'single_block_5', 'single_block_8', 'single_block_1', 'single_block_2', 'single_block_9', 'single_block_30', 'single_block_33', 'single_block_15', 'single_block_6', 'single_block_7', 'single_block_35', 'single_block_29'] 
N_TOP = 30           # Wir betrachten die ersten 40 Blöcke aus dem Ranking
START_VAL = 0.35     # Bester Block soll um weitere 30% (des Rests) reduziert werden
TOTAL_TARGET = 1.19*10**9  # Anzahl Parameter die entfernt werden sollen


single_blocks_ids = [0, 33, 30, 2, 5, 13, 22, 1, 15, 18, 8, 19, 16, 6, 26, 7, 21, 27, 12, 24, 3, 17, 37]

double_blocks_ids = [13, 16, 5, 3, 17, 8, 15, 6, 14, 18, 7, 4, 10, 12, 11, 9, 1, 0]

single_ratios = [0.1004, 0.2799, 0.3046, 0.2098, 0.0671, 0.0588, 0.0505, 0.1879, 0.2904, 0.1802, 0.0172, 0.133, 0.2968, 0.2571, 0.2812, 0.1079, 0.102, 0.0961, 0.0902, 0.0843, 0.0784, 0.1184, 0.1117]
double_ratios = [0.5492, 0.5632, 0.4991, 0.4923, 0.529, 0.569, 0.4978, 0.5573, 0.4747, 0.4832, 0.4886, 0.4961, 0.5029, 0.4555, 0.5029, 0.4405, 0.4136, 0.5321]
# ---------------------------------------------------------------------------------------------------

prev_compression_ratios = {}

for block_id, ratio in zip(single_blocks_ids, single_ratios):
    prev_compression_ratios[f"single_block_{block_id}"] = ratio

for block_id, ratio in zip(double_blocks_ids, double_ratios):
    prev_compression_ratios[f"double_block_{block_id}"] = ratio



x_val, new_ratios = calculate_advanced_linear_compression(
    ranked_list, N_TOP, prev_compression_ratios, param_counts, START_VAL, TOTAL_TARGET
)

single_block_ids = []
single_blocks_compression_ratios = []
double_block_ids = []
double_blocks_compression_ratios = []


for name, ratio_step in new_ratios.items():
    if ratio_step > 0: 
        r_prev = prev_compression_ratios.get(name, 0.0)
        total_ratio = 1 - (1 - r_prev) * (1 - ratio_step)
        
        if "single_block_" in name:
            single_block_ids.append(int(name.replace("single_block_", "")))
            single_blocks_compression_ratios.append(round(total_ratio, 4))
        elif "double_block_" in name:
            double_block_ids.append(int(name.replace("double_block_", "")))
            double_blocks_compression_ratios.append(round(total_ratio, 4))


print(f"Berechneter Abnahmefaktor x: {x_val:.6f}\n")

print("Single Block IDs:")
print(single_block_ids)
print("\nSingle Blocks Compression Ratios:")
print(single_blocks_compression_ratios)

print("\n" + "="*30 + "\n")

print("Double Block IDs:")
print(double_block_ids)
print("\nDouble Blocks Compression Ratios:")
print(double_blocks_compression_ratios)


total_removed = 0
for name, ratio in new_ratios.items():
    p_curr = param_counts[name] * (1 - prev_compression_ratios.get(name, 0))
    total_removed += p_curr * ratio
print(f"\nCheck - Gesamt zusätzlich entfernt: {total_removed:,.0f} / Ziel: {TOTAL_TARGET:,.0f}")
