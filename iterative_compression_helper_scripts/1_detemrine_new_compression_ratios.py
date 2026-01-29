# def calculate_linear_compression(parameter_counts, start_ratio, target_to_remove):
#     """
#     parameter_counts: Liste der Parameteranzahl pro Block (sortiert nach Ranking)
#     start_ratio: Verkleinerung des besten Blocks (z.B. 0.5 für 50%)
#     target_to_remove: Gesamtanzahl der Parameter, die weg sollen
#     """
#     n = len(parameter_counts)
    
#     # Zähler: Summe(P_i * R_start) - Target
#     numerator = sum(p * start_ratio for p in parameter_counts) - target_to_remove
    
#     # Nenner: Summe(P_i * (i-1))
#     denominator = sum(p * i for i, p in enumerate(parameter_counts))
    
#     if denominator == 0:
#         return 0, [start_ratio] * n

#     x = numerator / denominator
    
#     # Neue Ratios berechnen
#     new_ratios = []
#     for i in range(n):
#         ratio = start_ratio - (i * x)
#         new_ratios.append(max(0, ratio)) # Ratio kann nicht negativ sein
        
#     return x, new_ratios

# # BEISPIELWERTE (Bitte mit deinen echten Daten füllen)
# # Angenommen, das sind die Parameterzahlen deiner Top-Blöcke (in der Reihenfolge des Rankings)
# param_counts = [
#     340.0, 256.7, 266.9, 261.8, 241.4, 231.2, 221.0, 215.9, 236.3, 226.1, 
#     282.2, 210.8, 190.4, 195.5, 185.3, 170.0, 180.2, 140.0, 140.0, 140.0, 
#     159.8, 140.0, 128.8, 140.0, 140.0, 140.0, 140.0, 135.1, 126.7, 140.0, 
#     112.0, 133.0, 140.0, 140.0, 122.5, 140.0, 124.6, 140.0, 130.9, 140.0
# ]
# start_val = 0.4  # 50%
# total_target = 1400  # Wie viele Parameter insgesamt weg sollen

# x_val, ratios = calculate_linear_compression(param_counts, start_val, total_target)

# print(f"Berechneter Abnahmefaktor x: {x_val:.6f}")
# print("-" * 30)
# for i, r in enumerate(ratios):
#     print(f"Block {i+1} (Rang {i}): Neue Ratio = {r:.4f} ({r*100:.2f}%)")

# # Validierung
# total_removed = sum(p * r for p, r in zip(param_counts, ratios))
# print("-" * 30)
# print(f"Tatsächlich entfernte Parameter: {total_removed:,.0f}")

# print(ratios)







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
    
    # 1. Auswahl der relevanten Blöcke für diesen Schritt
    selected_blocks = all_blocks_ranked[:num_top_blocks]
    n = len(selected_blocks)
    
    # 2. Berechnung der aktuell noch vorhandenen Parameter (Substanz)
    # P_curr = P_orig * (1 - prev_ratio)
    current_params = []
    for name in selected_blocks:
        p_orig = block_param_counts.get(name, 0)
        p_prev_ratio = prev_compression_ratios.get(name, 0.0)
        current_params.append(p_orig * (1 - p_prev_ratio))
    
    # 3. Mathematische Herleitung von x (analog zu deiner Formel)
    # Sum(P_curr * (start_ratio - i*x)) = target
    # Sum(P_curr * start_ratio) - x * Sum(P_curr * i) = target
    
    numerator = sum(p * start_ratio for p in current_params) - target_to_remove
    denominator = sum(p * i for i, p in enumerate(current_params))
    
    if denominator == 0:
        x = 0
    else:
        x = numerator / denominator

    # 4. Neue Ratios berechnen und mit Namen verknüpfen
    new_ratios_dict = {}
    for i, name in enumerate(selected_blocks):
        # Lineare Abnahme der Ratio basierend auf dem Rang
        ratio = start_ratio - (i * x)
        new_ratios_dict[name] = max(0.0, ratio)
        
    return x, new_ratios_dict

# --- BEISPIEL FÜR DIE ANWENDUNG ---


# 2. Die originalen Parametergrößen (Beispielwerte)
# Hinweis: Double Blocks sind meist größer als Single Blocks
NUM_DOUBLE_BLOCKS = 19  # 0 bis 18
NUM_SINGLE_BLOCKS = 38  # 0 bis 37

# Parameter-Werte in Millionen (als Float oder Integer)
PARAM_DOUBLE = 340_000_000
PARAM_SINGLE = 141_000_000

# Dictionary initialisieren
param_counts = {}

# Double Blocks (0-18) befüllen
for i in range(NUM_DOUBLE_BLOCKS):
    param_counts[f"double_block_{i}"] = PARAM_DOUBLE

# Single Blocks (0-37) befüllen
for i in range(NUM_SINGLE_BLOCKS):
    param_counts[f"single_block_{i}"] = PARAM_SINGLE

# 3. Bereits erfolgte Kompression (was wurde in vorherigen Schritten getan?)
# ----------------------------- Parameters to specify ------------------------------------------------

ranked_list =['double_block_7', 'double_block_0', 'double_block_17', 'double_block_4', 'double_block_10', 'double_block_9', 'double_block_11', 'double_block_15', 'double_block_6', 'double_block_12', 'double_block_18', 'double_block_3', 'double_block_8', 'double_block_14', 'double_block_5', 'double_block_13', 'double_block_16', 'double_block_1', 'single_block_15', 'single_block_5', 'single_block_18', 'single_block_6', 'single_block_3', 'single_block_16', 'single_block_28', 'single_block_22', 'single_block_33', 'single_block_26', 'single_block_0', 'single_block_21', 'single_block_30', 'single_block_27', 'single_block_1', 'single_block_12', 'single_block_11', 'single_block_37', 'single_block_13', 'single_block_10', 'single_block_2', 'single_block_8', 'single_block_14', 'single_block_20', 'single_block_31', 'single_block_24', 'single_block_7', 'single_block_19', 'single_block_34', 'single_block_17', 'single_block_4', 'single_block_36', 'single_block_25', 'single_block_23', 'single_block_9', 'single_block_32', 'single_block_35', 'single_block_29', 'double_block_2'] 

N_TOP = 20            # Wir betrachten die ersten 4 Blöcke aus dem Ranking
START_VAL = 0.3      # Bester Block soll um weitere 40% (des Rests) reduziert werden
TOTAL_TARGET = 1.19*10**9  # Wir wollen insgesamt 1500 weitere Parameter entfernen


single_blocks_ids = [0, 33]
double_blocks_ids = [13, 16, 5, 3, 17, 8, 15, 6, 14, 18, 7, 4, 10, 12, 11, 9, 1, 0]

single_ratios = [0.0702, 0.0575]
double_ratios = [0.3, 0.2872, 0.2745, 0.2617, 0.2489, 0.2362, 0.2234, 0.2106, 0.1979, 0.1851, 0.1724, 0.1596, 0.1468, 0.1341, 0.1213, 0.1085, 0.0958, 0.083]
# ---------------------------------------------------------------------------------------------------
# 2. Dictionary erstellen
prev_compression_ratios = {}

# Single Blocks hinzufügen
for block_id, ratio in zip(single_blocks_ids, single_ratios):
    prev_compression_ratios[f"single_block_{block_id}"] = ratio

# Double Blocks hinzufügen
for block_id, ratio in zip(double_blocks_ids, double_ratios):
    prev_compression_ratios[f"double_block_{block_id}"] = ratio



x_val, new_ratios = calculate_advanced_linear_compression(
    ranked_list, N_TOP, prev_compression_ratios, param_counts, START_VAL, TOTAL_TARGET
)

# --- EXTRAKTION DER VIER LISTEN ---
single_block_ids = []
single_blocks_compression_ratios = []
double_block_ids = []
double_blocks_compression_ratios = []

# Wir gehen die new_ratios durch (sie enthalten nur die N_TOP Blöcke)
for name, ratio_step in new_ratios.items():
    if ratio_step > 0:  # Exkludiere Blöcke mit Ratio 0
        r_prev = prev_compression_ratios.get(name, 0.0)
        # Berechnung der kumulativen Gesamtratio
        total_ratio = 1 - (1 - r_prev) * (1 - ratio_step)
        
        if "single_block_" in name:
            single_block_ids.append(int(name.replace("single_block_", "")))
            single_blocks_compression_ratios.append(round(total_ratio, 4))
        elif "double_block_" in name:
            double_block_ids.append(int(name.replace("double_block_", "")))
            double_blocks_compression_ratios.append(round(total_ratio, 4))

# --- FINALE AUSGABE ---
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

# Validierung der Gesamtsumme
total_removed = 0
for name, ratio in new_ratios.items():
    p_curr = param_counts[name] * (1 - prev_compression_ratios.get(name, 0))
    total_removed += p_curr * ratio
print(f"\nCheck - Gesamt zusätzlich entfernt: {total_removed:,.0f} / Ziel: {TOTAL_TARGET:,.0f}")

# import numpy as np

# # Beispiel-Daten (dein bereitgestelltes Dict für Single Blocks)
# single_blocks = {0: np.float32(0.0004447671), 1: np.float32(0.00021464769), 2: np.float32(0.00028852306), 3: np.float32(0.00023121116), 4: np.float32(0.00022990363), 5: np.float32(0.00029166148), 6: np.float32(0.0002088295), 7: np.float32(0.00023132279), 8: np.float32(0.00027558074), 9: np.float32(0.000251191), 10: np.float32(0.00023983774), 11: np.float32(0.00021429289), 12: np.float32(0.0002337437), 13: np.float32(0.0002761605), 14: np.float32(0.000197263), 15: np.float32(0.00025597087), 16: np.float32(0.00017455647), 17: np.float32(0.0002071971), 18: np.float32(0.0002502421), 19: np.float32(0.00022706532), 20: np.float32(0.00020861626), 21: np.float32(0.00020861626), 22: np.float32(0.00018732889), 23: np.float32(0.00018590972), 24: np.float32(0.00021996952), 25: np.float32(0.00023416111), 26: np.float32(0.0001974481), 27: np.float32(0.00022138868), 28: np.float32(0.00021326577), 29: np.float32(0.0003704003), 30: np.float32(0.00028383164), 31: np.float32(0.00019868214), 32: np.float32(0.00027247838), 33: np.float32(0.00041334706), 34: np.float32(0.0002412569), 35: np.float32(0.00035053206), 36: np.float32(0.00025544848), 37: np.float32(0.0002151145)} 


# # Dummy-Daten für Double Blocks (da du dieses Dict nur erwähnt hast)
# double_blocks =  {0: np.float32(9.7587996e-05), 1: np.float32(0.00013024875), 2: np.float32(0.00352953), 3: np.float32(0.00013878533), 4: np.float32(0.00012275748), 5: np.float32(0.00016758982), 6: np.float32(0.00012147312), 7: np.float32(0.0001177312), 8: np.float32(0.00013922995), 9: np.float32(0.000103472536), 10: np.float32(0.00011604882), 11: np.float32(9.984416e-05), 12: np.float32(0.000114595125), 13: np.float32(0.0001902276), 14: np.float32(0.00012136692), 15: np.float32(0.00013854969), 16: np.float32(0.00016478931), 17: np.float32(0.0001479662), 18: np.float32(0.00012390173)} 

# def get_sorted_block_names(single_dict, double_dict):
#     combined_list = []

#     # 1. Single Blocks verarbeiten
#     for block_nr, score in single_dict.items():
#         combined_list.append({
#             "name": f"single_block_{block_nr}",
#             "score": float(score)
#         })

#     # 2. Double Blocks verarbeiten
#     for block_nr, score in double_dict.items():
#         combined_list.append({
#             "name": f"double_block_{block_nr}",
#             "score": float(score)
#         })

#     # 3. Sortieren nach dem Score (kleinster CMMD zuerst)
#     # Falls du absteigend sortieren willst (größter zuerst), setze reverse=True
#     sorted_data = sorted(combined_list, key=lambda x: x["score"])

#     # 4. Nur die Namen (Strings) in ein Array extrahieren
#     result_array = [item["name"] for item in sorted_data]
    
#     return result_array

# # Skript ausführen
# sorted_blocks = get_sorted_block_names(single_blocks, double_blocks)

# # Ausgabe zur Kontrolle
# print("Top 5 Blöcke (niedrigster CMMD):")
# print(sorted_blocks)

# # Wenn du auch die Scores sehen willst während der Entwicklung:
# # for block in sorted_blocks:
# #     print(block)