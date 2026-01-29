# --- KONFIGURATION ---
params_per_double_block_mio = 340.0
params_per_single_block_mio = 122.0

# --- DATEN (Aus vorherigem Schritt) ---
double_blocks = [13, 16, 5, 17, 3, 8, 15, 6, 14, 18, 7, 4, 10, 11, 12, 9, 1, 0]
double_blocks_compression_ratio =[0.602799, 0.608084, 0.578074, 0.579491, 0.581644, 0.563391, 0.544418, 0.564525, 0.562204, 0.527651, 0.550332, 0.51486, 0.539195, 0.537747, 0.50969, 0.509759, 0.40028, 0.4]


single_blocks =  [0, 33, 30, 2, 5, 13, 1, 15, 18, 8, 19, 3, 28, 26, 12, 37, 6, 16, 23, 22, 14, 31, 17, 21, 20, 11, 24]

single_blocks_compression_ratio = [0.515, 0.485, 0.41, 0.395, 0.275, 0.26, 0.22587, 0.185, 0.155, 0.14, 0.125, 0.11, 0.146448, 0.199951, 0.065, 0.069078, 0.101685, 0.191658, 0.179403, 0.167147, 0.142636, 0.118126, 0.10587, 0.093615, 0.081359, 0.044593, 0.007827]

# --- BERECHNUNG ---

# 1. Berechnung für Double Blocks
# Summe aller Ratios * Parameter pro Block
total_removed_double = sum(double_blocks_compression_ratio) * params_per_double_block_mio

# 2. Berechnung für Single Blocks
total_removed_single = sum(single_blocks_compression_ratio) * params_per_single_block_mio

# 3. Gesamtsumme
total_removed_global = total_removed_double + total_removed_single

# --- AUSGABE ---
print(f"{'--- ERGEBNISANALYSE ---':^40}")
print(f"Anzahl bearbeiteter Double Blöcke: {len(double_blocks)}")
print(f"Anzahl bearbeiteter Single Blöcke: {len(single_blocks)}")
print("-" * 40)

print(f"Entfernt aus Double Blöcken: {total_removed_double:10.2f} Mio Parameter")
print(f"Entfernt aus Single Blöcken: {total_removed_single:10.2f} Mio Parameter")
print("=" * 40)
print(f"GESAMT ENTFERNT:             {total_removed_global:10.2f} Mio Parameter")
print(f"                             {total_removed_global / 1000:10.2f} Mrd Parameter")
print("=" * 40)

# Optional: Durchschnittliche Einsparung pro Blocktyp
avg_double = total_removed_double / len(double_blocks)
avg_single = total_removed_single / len(single_blocks)

print(f"\n--- DURCHSCHNITTE ---")
print(f"Ø Entfernung pro Double Block: {avg_double:.2f} Mio")
print(f"Ø Entfernung pro Single Block: {avg_single:.2f} Mio")