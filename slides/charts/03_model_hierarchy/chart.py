"""Loan State Transition Diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'figure.figsize': (10, 6), 'figure.dpi': 150
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# State positions
states = {
    'Performing': (2, 5),
    '30 DPD': (5, 5),
    '60 DPD': (5, 3.5),
    '90 DPD': (5, 2),
    'Default': (8, 2),
    'Prepaid': (2, 3),
    'Matured': (2, 1.5),
}

# Colors
C_GOOD = '#27ae60'
C_DELQ = '#f39c12'
C_BAD = '#e74c3c'
C_EXIT = '#3498db'

state_colors = {
    'Performing': C_GOOD,
    '30 DPD': C_DELQ,
    '60 DPD': C_DELQ,
    '90 DPD': '#e67e22',
    'Default': C_BAD,
    'Prepaid': C_EXIT,
    'Matured': C_EXIT,
}

# Draw states
for name, (x, y) in states.items():
    circle = plt.Circle((x, y), 0.45, color=state_colors[name], alpha=0.85)
    ax.add_patch(circle)
    ax.text(x, y, name, ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')

# Transitions
transitions = [
    ('Performing', '30 DPD', '1.5%'),
    ('Performing', 'Prepaid', '0.8%'),
    ('Performing', 'Matured', ''),
    ('30 DPD', '60 DPD', '30%'),
    ('30 DPD', 'Performing', '40%'),
    ('60 DPD', '90 DPD', '40%'),
    ('60 DPD', 'Performing', '20%'),
    ('90 DPD', 'Default', '50%'),
    ('90 DPD', 'Performing', '10%'),
]

for from_state, to_state, prob in transitions:
    x1, y1 = states[from_state]
    x2, y2 = states[to_state]

    # Offset for circle radius
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/dist, dy/dist

    x1_adj = x1 + 0.5 * dx
    y1_adj = y1 + 0.5 * dy
    x2_adj = x2 - 0.5 * dx
    y2_adj = y2 - 0.5 * dy

    ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    if prob:
        mid_x = (x1 + x2) / 2 + 0.2
        mid_y = (y1 + y2) / 2 + 0.15
        ax.text(mid_x, mid_y, prob, fontsize=8, color='#34495e')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=C_GOOD, label='Performing'),
    mpatches.Patch(facecolor=C_DELQ, label='Delinquent'),
    mpatches.Patch(facecolor=C_BAD, label='Default'),
    mpatches.Patch(facecolor=C_EXIT, label='Exit'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

ax.set_title('Markov State Transitions (Monthly)', fontsize=14, pad=10)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: 03_model_hierarchy/chart.pdf")
