import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# --- 1. DATEN ---
#### Based on Original Model
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

# --- SKALIERUNGS-FAKTOREN ---
SCALE_SINGLE = 85.2
SCALE_DOUBLE = 204


### Based on 30 Blocks 70% model
single_block_cmmd = {0: np.float32(0.011535186), 1: np.float32(0.008410494), 2: np.float32(0.011914386), 3: np.float32(0.009834486), 4: np.float32(0.011621913), 5: np.float32(0.010568822), 6: np.float32(0.0076548858), 7: np.float32(0.009930115), 8: np.float32(0.012119596), 9: np.float32(0.010471907), 10: np.float32(0.010626995), 11: np.float32(0.00831305), 12: np.float32(0.0122165065), 13: np.float32(0.009876339), 14: np.float32(0.012473319), 15: np.float32(0.009108556), 16: np.float32(0.011666492), 17: np.float32(0.008026959), 18: np.float32(0.010434464), 19: np.float32(0.011271096), 20: np.float32(0.010112445), 21: np.float32(0.01257209), 22: np.float32(0.010920977), 23: np.float32(0.011492959), 24: np.float32(0.011910346), 25: np.float32(0.0078559965), 26: np.float32(0.0079061175), 27: np.float32(0.0076467497), 28: np.float32(0.009306714), 29: np.float32(0.008791685), 30: np.float32(0.008580386), 31: np.float32(0.006595854), 32: np.float32(0.008480929), 33: np.float32(0.011759732), 34: np.float32(0.0106986165), 35: np.float32(0.005918551), 36: np.float32(0.006422112), 37: np.float32(0.012030661)} 

double_block_cmmd = {0: np.float32(0.013403796), 1: np.float32(0.010214197), 2: np.float32(0.011744453), 3: np.float32(0.0071424777), 4: np.float32(0.006664335), 5: np.float32(0.012594119), 6: np.float32(0.011563259), 7: np.float32(0.007333486), 8: np.float32(0.011848994), 9: np.float32(0.011615319), 10: np.float32(0.010793511), 11: np.float32(0.010609705), 12: np.float32(0.011922435), 13: np.float32(0.012547749), 14: np.float32(0.011447617), 15: np.float32(0.011818012), 16: np.float32(0.012288216), 17: np.float32(0.012426178), 18: np.float32(0.012662428)} 

# --- SKALIERUNGS-FAKTOREN ---
SCALE_SINGLE = 1
SCALE_DOUBLE = 1

labels, values, colors = [], [], []
outlier_key = 2
current_idx = 0
outlier_index = 0

# Werte für Limits sammeln
outlier_value_scaled = 0
normal_values_scaled = []

# Double Blocks
for k in sorted(double_block_cmmd.keys()):
    labels.append(f"D-{k+1}")
    
    # HIER WIRD SKALIERT (Division)
    val = double_block_cmmd[k] / SCALE_DOUBLE 
    values.append(val)
    colors.append('#FF7F50')
    
    if k == outlier_key:
        outlier_index = current_idx
        outlier_value_scaled = val
    else:
        normal_values_scaled.append(val)
    current_idx += 1

# Single Blocks
for k in sorted(single_block_cmmd.keys()):
    labels.append(f"S-{k+1}")
    
    # HIER WIRD SKALIERT (Division)
    val = single_block_cmmd[k] / SCALE_SINGLE
    values.append(val)
    colors.append('#5DADE2')
    
    normal_values_scaled.append(val)

# --- 2. PLOT SETUP (Dynamische Limits) ---
plot_ratio = 4
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18, 7), 
                               gridspec_kw={'height_ratios': [1, plot_ratio], 'hspace': 0.1})

ax1.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
ax2.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)

# --- DYNAMISCHE LIMITS BERECHNEN ---
# Wir setzen die Limits basierend auf den neuen, skalierten Werten
max_normal = max(normal_values_scaled)

# Oben: Fokus auf den skalierten Ausreißer (+/- ein bisschen Luft)
margin_top = outlier_value_scaled * 0.02
ax1.set_ylim(outlier_value_scaled - margin_top, outlier_value_scaled + margin_top)

# Unten: Fokus auf den Rest (+ etwas Luft nach oben)
ax2.set_ylim(0.00, max_normal * 1.1)

ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(labeltop=False, bottom=False)
ax2.xaxis.tick_bottom()

# --- 3. DIE STRICHE (BREAK MARKERS) ---
def draw_break_slashes(axis_top, axis_bottom, x_rel_pos, ratio, width=0.01, single_cut=False):
    d = width
    kwargs = dict(transform=axis_top.transAxes, color='k', clip_on=False, lw=1.0)

    if single_cut:
        offsets = [0]
    else:
        v_off = d * 0.8 
        offsets = [-v_off, v_off]

    for off in offsets:
        axis_top.plot((x_rel_pos - d, x_rel_pos + d), (-d + off, +d + off), **kwargs)

    kwargs.update(transform=axis_bottom.transAxes)
    d_scaled = d / ratio
    
    if single_cut:
        offsets_scaled = [0]
    else:
        v_off_scaled = v_off / ratio
        offsets_scaled = [-v_off_scaled, v_off_scaled]

    for off in offsets_scaled:
         axis_bottom.plot((x_rel_pos - d, x_rel_pos + d), (1 - d_scaled + off, 1 + d_scaled + off), **kwargs)


# A) Striche an den Achsen (single_cut=True für sauberen Look oder False für Doppellinie)
draw_break_slashes(ax1, ax2, x_rel_pos=0, ratio=plot_ratio, single_cut=True)
draw_break_slashes(ax1, ax2, x_rel_pos=1, ratio=plot_ratio, single_cut=True)

# B) Strich am Balken
x_min, x_max = ax1.get_xlim()
total_span = x_max - x_min
bar_rel_pos = (outlier_index - x_min) / total_span

draw_break_slashes(ax1, ax2, x_rel_pos=bar_rel_pos, ratio=plot_ratio, width=0.012, single_cut=True)


# --- 4. FINISH ---
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.xticks(rotation=90, fontsize=10)
plt.xlabel('Block Index', fontsize=12)

# Beschriftung angepasst (Scaled)
fig.text(0.005, 0.55, 'Parameter Weighted CMMD', va='center', rotation='vertical', fontsize=14)

legend_elements = [
    Line2D([0], [0], color='#FF7F50', lw=6, label='Double Blocks'),
    Line2D([0], [0], color='#5DADE2', lw=6, label='Single Blocks')
]
ax1.legend(handles=legend_elements, loc='upper right')

# Da die Werte jetzt sehr klein sind, formatieren wir die Y-Achse optional wissenschaftlich
from matplotlib.ticker import ScalarFormatter
y_formatter = ScalarFormatter(useMathText=True)
y_formatter.set_powerlimits((-2, 2))
ax2.yaxis.set_major_formatter(y_formatter)
ax1.yaxis.set_major_formatter(y_formatter)

plt.savefig('/home/hd/hd_hd/hd_om233/SVD/images/ranking/flux_cmmd_broken_axis_scaled_block_40_30.png', bbox_inches='tight', dpi=300)
plt.show()