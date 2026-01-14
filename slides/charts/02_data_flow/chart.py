"""SPV Data Flow Diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Colors
C1 = '#1abc9c'  # Origination
C2 = '#3498db'  # SPV
C3 = '#e74c3c'  # Tranches
C4 = '#f39c12'  # Investors

def draw_box(ax, x, y, w, h, text, color, fontsize=11):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03",
        facecolor=color, edgecolor='white', alpha=0.85
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold')

# Originators
draw_box(ax, 0.3, 4.5, 1.8, 0.8, 'Originator 1', C1)
draw_box(ax, 0.3, 3.3, 1.8, 0.8, 'Originator 2', C1)
draw_box(ax, 0.3, 2.1, 1.8, 0.8, 'Originator N', C1)

# SPV
draw_box(ax, 3.5, 2.5, 2.5, 2.5, 'SPV\n(Loan Pool)', C2, fontsize=14)

# Tranches
draw_box(ax, 7.2, 4.3, 2.3, 0.7, 'Senior A (70%)', C3, fontsize=10)
draw_box(ax, 7.2, 3.4, 2.3, 0.7, 'Mezz B (15%)', C3, fontsize=10)
draw_box(ax, 7.2, 2.5, 2.3, 0.7, 'Junior C (10%)', C3, fontsize=10)
draw_box(ax, 7.2, 1.6, 2.3, 0.7, 'Equity (5%)', C3, fontsize=10)

# Arrows
arrow_style = dict(arrowstyle='->', color='#7f8c8d', lw=2)
ax.annotate('', xy=(3.4, 3.75), xytext=(2.2, 4.9), arrowprops=arrow_style)
ax.annotate('', xy=(3.4, 3.75), xytext=(2.2, 3.7), arrowprops=arrow_style)
ax.annotate('', xy=(3.4, 3.75), xytext=(2.2, 2.5), arrowprops=arrow_style)

ax.annotate('', xy=(7.1, 4.65), xytext=(6.1, 3.75), arrowprops=arrow_style)
ax.annotate('', xy=(7.1, 3.75), xytext=(6.1, 3.75), arrowprops=arrow_style)
ax.annotate('', xy=(7.1, 2.85), xytext=(6.1, 3.75), arrowprops=arrow_style)
ax.annotate('', xy=(7.1, 1.95), xytext=(6.1, 3.75), arrowprops=arrow_style)

# Labels
ax.text(2.8, 5.3, 'Loan\nPortfolios', fontsize=9, ha='center')
ax.text(6.6, 5.0, 'Waterfall\nDistribution', fontsize=9, ha='center')

# Data labels
ax.text(4.75, 1.8, 'Loan Tape\n+ Monthly Panel', fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: 02_data_flow/chart.pdf")
