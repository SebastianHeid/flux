import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Daten definieren
single_block_cmmd_ = {
    0: 0.00013765835, 1: 0.00014901161, 2: 0.00014475414, 3: 0.00014617329, 
    4: 0.00013765835, 5: 0.00013623919, 6: 0.00015043077, 7: 0.00013482003, 
    8: 0.00012772424, 9: 0.00013623919, 10: 0.00014049666, 11: 0.00014475414, 
    12: 0.00013056255, 13: 0.00017313730, 14: 0.00019158635, 15: 0.00015468824, 
    16: 0.00019158635, 17: 0.00013340087, 18: 0.00014049666, 19: 0.00019584384, 
    20: 0.00012062845, 21: 0.00016178403, 22: 0.00014901161, 23: 0.00015043077, 
    24: 0.00010785602, 25: 0.00033917881, 26: 0.00020577793, 27: 0.00031505313, 
    28: 0.00021713121, 29: 0.00022280784, 30: 0.00021713121, 31: 0.00018023310, 
    32: 0.00031647229, 33: 0.00018449056, 34: 0.00021713121, 35: 0.00028241248, 
    36: 0.00024409521, 37: 0.00015610740
}

double_block_cmmd_ = {
    0: 0.00012154673, 1: 0.00004499566, 2: 0.00499568740, 3: 0.00006018900, 
    4: 0.00006603260, 5: 0.00005726721, 6: 0.00005785157, 7: 0.00005434541, 
    8: 0.00006252644, 9: 0.00006018900, 10: 0.00005902029, 11: 0.00006135772, 
    12: 0.00006427952, 13: 0.00007655106, 14: 0.00006252644, 15: 0.00005492977, 
    16: 0.00005434541, 17: 0.00006194208, 18: 0.00005726721
}
constant_single = 140*0.6
single_block_cmmd = {k: v * constant_single for k, v in single_block_cmmd_.items()}

constant_double = 340*0.6
double_block_cmmd = {k: v * constant_double for k, v in double_block_cmmd_.items()}
# 2. Listen für Plotting zusammenstellen
labels, values, colors = [], [], []

# Double Blocks (D-1 bis D-18)
for k in sorted(double_block_cmmd.keys()):
    labels.append(f"D-{k}")
    values.append(double_block_cmmd[k])
    colors.append('#FF7F50') # Koralle

# Single Blocks (S-0 bis S-37)
for k in sorted(single_block_cmmd.keys()):
    labels.append(f"S-{k}")
    values.append(single_block_cmmd[k])
    colors.append('#5DADE2') # Blau

# 3. Plot erstellen
plt.figure(figsize=(18, 7))
plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)

# Legende und Beschriftung
legend_elements = [
    Line2D([0], [0], color='#FF7F50', lw=6, label='Double Blocks'),
    Line2D([0], [0], color='#5DADE2', lw=6, label='Single Blocks')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.yscale('log') # Logarithmisch wegen des D-2 Ausreißers
plt.ylabel('CMMD (Log-Scale)', fontsize=14)
plt.xlabel('Block Index', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/hd/hd_hd/hd_om233/SVD/images/ranking/flux_cmmd_barplot_1k.png')
plt.show()