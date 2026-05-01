import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Daten definieren
# single_block_cmmd_ = {
#     0: 0.00013765835, 1: 0.00014901161, 2: 0.00014475414, 3: 0.00014617329, 
#     4: 0.00013765835, 5: 0.00013623919, 6: 0.00015043077, 7: 0.00013482003, 
#     8: 0.00012772424, 9: 0.00013623919, 10: 0.00014049666, 11: 0.00014475414, 
#     12: 0.00013056255, 13: 0.00017313730, 14: 0.00019158635, 15: 0.00015468824, 
#     16: 0.00019158635, 17: 0.00013340087, 18: 0.00014049666, 19: 0.00019584384, 
#     20: 0.00012062845, 21: 0.00016178403, 22: 0.00014901161, 23: 0.00015043077, 
#     24: 0.00010785602, 25: 0.00033917881, 26: 0.00020577793, 27: 0.00031505313, 
#     28: 0.00021713121, 29: 0.00022280784, 30: 0.00021713121, 31: 0.00018023310, 
#     32: 0.00031647229, 33: 0.00018449056, 34: 0.00021713121, 35: 0.00028241248, 
#     36: 0.00024409521, 37: 0.00015610740
# }

# double_block_cmmd_ = {
#     0: 0.00012154673, 1: 0.00004499566, 2: 0.00499568740, 3: 0.00006018900, 
#     4: 0.00006603260, 5: 0.00005726721, 6: 0.00005785157, 7: 0.00005434541, 
#     8: 0.00006252644, 9: 0.00006018900, 10: 0.00005902029, 11: 0.00006135772, 
#     12: 0.00006427952, 13: 0.00007655106, 14: 0.00006252644, 15: 0.00005492977, 
#     16: 0.00005434541, 17: 0.00006194208, 18: 0.00005726721
# }
# constant_single = 140*0.6
# single_block_cmmd = {k: v * constant_single for k, v in single_block_cmmd_.items()}

# constant_double = 340*0.6
# double_block_cmmd = {k: v * constant_double for k, v in double_block_cmmd_.items()}

# print("Double Blocks:", double_block_cmmd)

# print("Single Blocks:", single_block_cmmd)

# single_block_cmmd = {
#     0: 0.012278556823730469,
#     1: 0.014543533325195312,
#     2: 0.013470649719238281,
#     3: 0.0152587890625,
#     4: 0.01823902130126953,
#     5: 0.014066696166992188,
#     6: 0.01609325408935547,
#     7: 0.016927719116210938,
#     8: 0.015020370483398438,
#     9: 0.019311904907226562,
#     10: 0.0171661376953125,
#     11: 0.0171661376953125,
#     12: 0.015616416931152344,
#     13: 0.014066696166992188,
#     14: 0.017881393432617188,
#     15: 0.014781951904296875,
#     16: 0.016450881958007812,
#     17: 0.01895427703857422,
#     18: 0.014901161193847656,
#     19: 0.015139579772949219,
#     20: 0.018358230590820312,
#     21: 0.016808509826660156,
#     22: 0.01430511474609375,
#     23: 0.019788742065429688,
#     24: 0.01895427703857422,
#     25: 0.016808509826660156,
#     26: 0.015377998352050781,
#     27: 0.01633167266845703,
#     28: 0.0152587890625,
#     29: 0.018715858459472656,
#     30: 0.013232231140136719,
#     31: 0.01621246337890625,
#     32: 0.017881393432617188,
#     33: 0.012755393981933594,
#     34: 0.01621246337890625,
#     35: 0.023126602172851562,
#     36: 0.017523765563964844,
#     37: 0.015854835510253906
# }

# double_block_cmmd = {
#     0: 0.021338462829589844,
#     1: 0.019073486328125,
#     2: 0.9413957595825195,
#     3: 0.013709068298339844,
#     4: 0.0152587890625,
#     5: 0.013709068298339844,
#     6: 0.014781951904296875,
#     7: 0.015139579772949219,
#     8: 0.013828277587890625,
#     9: 0.016808509826660156,
#     10: 0.015974044799804688,
#     11: 0.01609325408935547,
#     12: 0.01609325408935547,
#     13: 0.012993812561035156,
#     14: 0.014901161193847656,
#     15: 0.014543533325195312,
#     16: 0.013589859008789062,
#     17: 0.013709068298339844,
#     18: 0.014901161193847656
# }

# #2. Listen für Plotting zusammenstellen
# labels, values, colors = [], [], []

# # Double Blocks (D-1 bis D-18)
# for k in sorted(double_block_cmmd.keys()):
#     labels.append(f"D-{k}")
#     values.append(double_block_cmmd[k])
#     colors.append('#FF7F50') # Koralle

# # Single Blocks (S-0 bis S-37)
# for k in sorted(single_block_cmmd.keys()):
#     labels.append(f"S-{k}")
#     values.append(single_block_cmmd[k])
#     colors.append('#5DADE2') # Blau

# # 3. Plot erstellen
# plt.figure(figsize=(18, 7))
# plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)

# # Legende und Beschriftung
# legend_elements = [
#     Line2D([0], [0], color='#FF7F50', lw=6, label='Double Blocks'),
#     Line2D([0], [0], color='#5DADE2', lw=6, label='Single Blocks')
# ]
# plt.legend(handles=legend_elements, loc='upper right')

# plt.yscale('log') # Logarithmisch wegen des D-2 Ausreißers
# plt.ylabel('CMMD (Log-Scale)', fontsize=14)
# plt.xlabel('Block Index', fontsize=12)
# plt.xticks(rotation=90)
# plt.grid(axis='y', linestyle='--', alpha=0.3, which='both')

# plt.tight_layout()
# plt.savefig('/home/hd/hd_hd/hd_om233/SVD/images/ranking/flux_cmmd_barplot_100_2.png')
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# --- 1. DATEN VORBEREITEN ---
single_block_cmmd = {
    0: 0.012278556823730469, 1: 0.014543533325195312, 2: 0.013470649719238281, 3: 0.0152587890625,
    4: 0.01823902130126953, 5: 0.014066696166992188, 6: 0.01609325408935547, 7: 0.016927719116210938,
    8: 0.015020370483398438, 9: 0.019311904907226562, 10: 0.0171661376953125, 11: 0.0171661376953125,
    12: 0.015616416931152344, 13: 0.014066696166992188, 14: 0.017881393432617188, 15: 0.014781951904296875,
    16: 0.016450881958007812, 17: 0.01895427703857422, 18: 0.014901161193847656, 19: 0.015139579772949219,
    20: 0.018358230590820312, 21: 0.016808509826660156, 22: 0.01430511474609375, 23: 0.019788742065429688,
    24: 0.01895427703857422, 25: 0.016808509826660156, 26: 0.015377998352050781, 27: 0.01633167266845703,
    28: 0.0152587890625, 29: 0.018715858459472656, 30: 0.013232231140136719, 31: 0.01621246337890625,
    32: 0.017881393432617188, 33: 0.012755393981933594, 34: 0.01621246337890625, 35: 0.023126602172851562,
    36: 0.017523765563964844, 37: 0.015854835510253906
}

double_block_cmmd = {
    0: 0.021338462829589844, 1: 0.019073486328125, 2: 0.9413957595825195, 3: 0.013709068298339844,
    4: 0.0152587890625, 5: 0.013709068298339844, 6: 0.014781951904296875, 7: 0.015139579772949219,
    8: 0.013828277587890625, 9: 0.016808509826660156, 10: 0.015974044799804688, 11: 0.01609325408935547,
    12: 0.01609325408935547, 13: 0.012993812561035156, 14: 0.014901161193847656, 15: 0.014543533325195312,
    16: 0.013589859008789062, 17: 0.013709068298339844, 18: 0.014901161193847656
}

labels, values, colors = [], [], []
outlier_key = 2
current_idx = 0
outlier_index = 0

for k in sorted(double_block_cmmd.keys()):
    labels.append(f"D-{k+1}") 
    values.append(double_block_cmmd[k])
    colors.append('#FF7F50')
    if k == outlier_key:
        outlier_index = current_idx
    current_idx += 1

for k in sorted(single_block_cmmd.keys()):
    labels.append(f"S-{k+1}")
    values.append(single_block_cmmd[k])
    colors.append('#5DADE2')

# --- 2. PLOT SETUP ---
plot_ratio = 4
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 7), 
                               gridspec_kw={'height_ratios': [1, plot_ratio], 'hspace': 0.1})

ax1.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
ax2.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)

ax1.set_ylim(0.90, 0.96)
ax2.set_ylim(0.00, 0.03)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()

# --- 3. DIE STRICHE (BREAK MARKERS) - NEU DEFINIERT ---

def draw_break_slashes(axis_top, axis_bottom, x_rel_pos, ratio, width=0.01, single_cut=False):
    """
    Zeichnet die Unterbrechungsstriche.
    single_cut=True: Zeichnet nur EINEN zentrierten Strich (für den Balken).
    single_cut=False: Zeichnet ZWEI parallele Striche (für die Achsen).
    """
    d = width
    kwargs = dict(transform=axis_top.transAxes, color='k', clip_on=False, lw=1.0)

    # Bestimmen der vertikalen Offsets für die Striche
    if single_cut:
        # Ein Strich, genau in der Mitte
        offsets = [0]
    else:
        # Zwei Striche, leicht nach oben und unten versetzt für den "//" Effekt
        v_off = d * 0.8 
        offsets = [-v_off, v_off]

    # --- Oben (Unten am Plot, y=0) ---
    for off in offsets:
        # Die Linie geht von x-d bis x+d und y-d bis y+d, verschoben um 'off'
        axis_top.plot((x_rel_pos - d, x_rel_pos + d), (-d + off, +d + off), **kwargs)

    # --- Unten (Oben am Plot, y=1) ---
    kwargs.update(transform=axis_bottom.transAxes)
    d_scaled = d / ratio
    
    if single_cut:
        offsets_scaled = [0]
    else:
        v_off_scaled = v_off / ratio
        offsets_scaled = [-v_off_scaled, v_off_scaled]

    for off in offsets_scaled:
         axis_bottom.plot((x_rel_pos - d, x_rel_pos + d), (1 - d_scaled + off, 1 + d_scaled + off), **kwargs)


# A) Striche an den Achsen (Doppelt = Standard)
draw_break_slashes(ax1, ax2, x_rel_pos=0, ratio=plot_ratio, single_cut=True)
draw_break_slashes(ax1, ax2, x_rel_pos=1, ratio=plot_ratio, single_cut=True)

# B) Strich am Balken (Einzeln!)
x_min, x_max = ax1.get_xlim()
total_span = x_max - x_min
bar_rel_pos = (outlier_index - x_min) / total_span

# Hier setzen wir single_cut=True für den Balken
draw_break_slashes(ax1, ax2, x_rel_pos=bar_rel_pos, ratio=plot_ratio, width=0.012, single_cut=True)


# --- 4. FINISH ---
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.xticks(rotation=90, fontsize=10)
plt.xlabel('Block Index', fontsize=12)
fig.text(0.005, 0.55, 'CMMD Value', va='center', rotation='vertical', fontsize=14)

legend_elements = [
    Line2D([0], [0], color='#FF7F50', lw=6, label='Double Blocks'),
    Line2D([0], [0], color='#5DADE2', lw=6, label='Single Blocks')
]
ax1.legend(handles=legend_elements, loc='upper right')

plt.savefig('flux_cmmd_broken_axis_single_cut.png', bbox_inches='tight', dpi=300)
plt.show()


# plt.tight_layout() # Careful with tight_layout on broken axes, usually manual adjustment is safer
plt.savefig('/home/hd/hd_hd/hd_om233/SVD/images/ranking/flux_cmmd_barplot_broken_axis.png', bbox_inches='tight')
plt.show()