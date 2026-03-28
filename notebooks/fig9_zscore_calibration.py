"""
Figure 9: Z-Score Calibration and Timescale Selectivity

Paper Figure 9 — (a) Z-scores for Lorenz-Lorenz vs Rössler-Lorenz at coupling=0.3,
showing the method's selective sensitivity to heterogeneous-timescale coupling.
(b) Structural baseline: binding score at coupling=0 grows with data length.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'

# --- Load data ---

# Z-scores by system type
zscores = pd.read_csv('../results/cross_system_zscores.csv')

# Structural baseline
with open('../results/finite_sample_effect.json') as f:
    baseline = json.load(f)

# --- Panel (a): Z-score comparison ---

lorenz = zscores[zscores['system'] == 'coupled_lorenz']
rossler = zscores[zscores['system'] == 'coupled_rossler_lorenz']

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
x = np.arange(5)
width = 0.35

bars1 = ax.bar(x - width/2, lorenz['z_score'].values, width,
               color='#EF5350', alpha=0.85, label='Lorenz–Lorenz (same timescale)')
bars2 = ax.bar(x + width/2, rossler['z_score'].values, width,
               color='#42A5F5', alpha=0.85, label='Rössler–Lorenz (heterogeneous)')

ax.axhline(y=1.96, color='k', linestyle='--', linewidth=1, alpha=0.6)
ax.text(4.5, 2.1, '$p = 0.05$', fontsize=9, ha='right', alpha=0.6)
ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)

ax.set_xlabel('Random seed')
ax.set_ylabel('Z-score (vs. surrogate null)')
ax.set_title('(a) Z-scores at coupling $\\epsilon = 0.3$')
ax.set_xticks(x)
ax.set_xticklabels([f'seed {i}' for i in range(5)])
ax.legend(fontsize=8, frameon=True, fancybox=False, loc='lower left')
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(-5, 4)

# Annotate significance (p <= 0.05)
for i, (z_r, p_r) in enumerate(zip(rossler['z_score'].values, rossler['p_value'].values)):
    if p_r <= 0.05:
        ax.text(i + width/2, z_r + 0.15, '*', fontsize=14, ha='center', fontweight='bold',
                color='#1565C0')

# --- Panel (b): Structural baseline growth ---

ax = axes[1]
n_steps = [b['n_steps'] for b in baseline]
means = [b['mean_score'] for b in baseline]
stds = [b['std_score'] for b in baseline]

ax.errorbar(n_steps, means, yerr=stds, fmt='ko-', linewidth=2, markersize=7,
            capsize=5, capthick=1.5, elinewidth=1.5)
ax.fill_between(n_steps, [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)], alpha=0.15, color='gray')

ax.set_xlabel('Time series length ($n$)')
ax.set_ylabel('Binding score at $\\epsilon = 0$')
ax.set_title('(b) Structural baseline grows with data')
ax.set_xticks(n_steps)
ax.set_xticklabels(['3k', '10k', '20k'])
ax.grid(True, alpha=0.2)

# Annotate values
for n, m, s in zip(n_steps, means, stds):
    ax.annotate(f'{m:.0f} ± {s:.0f}', (n, m + s + 5),
                fontsize=8, ha='center', alpha=0.7)

plt.tight_layout()
fig.savefig('../figures/fig9_zscore_calibration.pdf', bbox_inches='tight')
fig.savefig('../figures/fig9_zscore_calibration.png', bbox_inches='tight')
plt.show()
print("Saved to figures/fig9_zscore_calibration.{pdf,png}")
