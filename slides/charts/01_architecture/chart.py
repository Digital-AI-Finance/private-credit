"""Hierarchical Deep Generative Architecture Diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    'figure.figsize': (10, 6), 'figure.dpi': 150,
    'font.family': 'sans-serif'
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
C_MACRO = '#3498db'
C_COHORT = '#2ecc71'
C_LOAN = '#e74c3c'
C_PORTFOLIO = '#9b59b6'
C_ARROW = '#7f8c8d'

# Level boxes
boxes = [
    (1.5, 5.0, 7, 0.8, 'Level 1: Macro VAE', C_MACRO, 'Generate correlated macro paths'),
    (1.5, 3.5, 7, 0.8, 'Level 2: Transition Transformer', C_COHORT, 'Cohort-level dynamics'),
    (1.5, 2.0, 7, 0.8, 'Level 3: Loan Trajectory Model', C_LOAN, 'Individual loan paths'),
    (1.5, 0.5, 7, 0.8, 'Level 4: Portfolio Aggregator', C_PORTFOLIO, 'Waterfall & loss distribution'),
]

for x, y, w, h, title, color, desc in boxes:
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=color, edgecolor='white', linewidth=2, alpha=0.85
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.1, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color='white')
    ax.text(x + w/2, y + h/2 - 0.2, desc, ha='center', va='center',
            fontsize=10, color='white', alpha=0.9)

# Arrows
for y_start, y_end in [(4.95, 4.35), (3.45, 2.85), (1.95, 1.35)]:
    ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2))

# Labels
ax.text(0.3, 5.4, 'GDP, Unemp,\nSpreads', fontsize=9, ha='center', color=C_MACRO)
ax.text(0.3, 3.9, 'Transition\nMatrices', fontsize=9, ha='center', color=C_COHORT)
ax.text(0.3, 2.4, 'State &\nPayment Seq', fontsize=9, ha='center', color=C_LOAN)
ax.text(0.3, 0.9, 'VaR, CVaR\nTranche IRR', fontsize=9, ha='center', color=C_PORTFOLIO)

ax.text(9.5, 5.4, 'Conditional\nVAE', fontsize=9, ha='center', color=C_MACRO)
ax.text(9.5, 3.9, 'Transformer\nEncoder', fontsize=9, ha='center', color=C_COHORT)
ax.text(9.5, 2.4, 'AR Decoder\n+ Diffusion', fontsize=9, ha='center', color=C_LOAN)
ax.text(9.5, 0.9, 'Differentiable\nWaterfall', fontsize=9, ha='center', color=C_PORTFOLIO)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: 01_architecture/chart.pdf")
