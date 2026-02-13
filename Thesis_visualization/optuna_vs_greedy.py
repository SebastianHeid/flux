import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------
# 1. DATEN (Deine exakten Werte)
# ---------------------------------------------------------
num_blocks = 38
iterations = ["Iteration 1", "Iteration 2", "Iteration 3", "Iteration 4"]

# CMMD Werte für die Annotation (Delta zeigen!)
cmmd_greedy = [0.122, 0.128, 0.188, 0.551]
cmmd_optuna = [0.086, 0.097, 0.171, 0.292]

greedy_data = [
    [0, 1, 2, 3, 4, 5, 6, 16, 21,23, 27],
    [5,8,18,20],
    [9,11,18],
    [10,18,23]
]

optuna_data = [
    [0,1,2,3,4,6,7,16, 17,25,27],
    [5,8,19,20],
    [9,11,21],
    [14,18,22]
]

# ---------------------------------------------------------
# 2. MATRIX ERSTELLEN (Pro Iteration, NICHT kumulativ)
# ---------------------------------------------------------
# 0 = Kept (Hintergrund)
# 1 = Beide (Konsens)
# 2 = Greedy Only (Die "falsche" Wahl der Heuristik)
# 3 = Optuna Only (Die "bessere" Wahl)

matrix = np.zeros((len(iterations), num_blocks))

for i in range(len(iterations)):
    g_set = set(greedy_data[i])
    o_set = set(optuna_data[i])
    
    # Schnittmenge (Beide)
    for block in g_set.intersection(o_set):
        matrix[i, block] = 1
        
    # Greedy Only (Was Greedy fälschlicherweise entfernen wollte)
    for block in g_set - o_set:
        matrix[i, block] = 2
        
    # Optuna Only (Was Optuna stattdessen entfernt hat)
    for block in o_set - g_set:
        matrix[i, block] = 3

# ---------------------------------------------------------
# 3. PLOTTING
# ---------------------------------------------------------
# Farben: Hellgrau (Hintergrund), Dunkelgrau (Konsens), Blau (Greedy), Rot (Optuna)
colors = ['#F5F5F5', '#505050', '#1f77b4', '#d62728'] 
cmap = mcolors.ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

# Gitter und Achsen
ax.set_xticks(np.arange(num_blocks))
ax.set_xticklabels(np.arange(num_blocks), fontsize=9, rotation=90)
ax.set_xlabel("Transformer Block Index", fontsize=12, fontweight='bold')

ax.set_yticks(np.arange(len(iterations)))
ax.set_yticklabels(iterations, fontsize=11, fontweight='bold')

# Weiße Trennlinien
ax.set_xticks(np.arange(num_blocks + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(iterations) + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# Titel
ax.set_title("Vergleich der Block-Selektion: Greedy Baseline vs. Optuna Optimierung", fontsize=14, pad=20)

# ---------------------------------------------------------
# 4. ANNOTATION DER CMMD VERBESSERUNG
# ---------------------------------------------------------
# Wir schreiben rechts neben jede Zeile, wie viel besser Optuna war
for i in range(len(iterations)):
    delta = cmmd_greedy[i] - cmmd_optuna[i]
    text = f"Optuna Vorteil:\nCMMD -{delta:.3f}"
    # Positionierung rechts neben dem Plot
    ax.text(num_blocks + 0.5, i, text, va='center', ha='left', fontsize=10, fontweight='bold', color='#2ca02c')

# ---------------------------------------------------------
# 5. AUSSAGEKRÄFTIGE LEGENDE
# ---------------------------------------------------------
legend_elements = [
    Patch(facecolor='#505050', label='Konsens (Beide wählen diesen Block)'),
    Patch(facecolor='#1f77b4', label='Greedy Exklusiv (Schlechtere Wahl)'),
    Patch(facecolor='#d62728', label='Optuna Exklusiv (Bessere Wahl)')
]

ax.legend(handles=legend_elements, 
          loc='upper center', 
          bbox_to_anchor=(0.5, -0.25), 
          ncol=3, 
          frameon=False,
          fontsize=11)

plt.tight_layout()
plt.show()
# plt.savefig("optuna_vs_greedy_stepwise.pdf", bbox_inches='tight')