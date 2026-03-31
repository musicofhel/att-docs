"""Compare baseline vs Lindner-aware topological sparsity results.

Reads JSON results from both runs and produces a side-by-side comparison table.
Saves to data/transformer/comparison_results.txt.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(REPO_ROOT, "data", "transformer", "baseline_results.json")
LINDNER_PATH = os.path.join(REPO_ROOT, "data", "transformer", "lindner_results.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "transformer", "comparison_results.txt")

with open(BASELINE_PATH) as f:
    baseline = json.load(f)
with open(LINDNER_PATH) as f:
    lindner = json.load(f)

lines = []

def out(s=""):
    print(s)
    lines.append(s)

out("=" * 90)
out("COMPARISON: BASELINE (threshold=1e4) vs LINDNER (dimension-aware)")
out("=" * 90)

# Main comparison table
out("\n## Degeneracy Rate Comparison (Experiment 2)")
out("")
out("Level | Old Degen% | New Degen% | Delta  | Mean Cond | DimWarn? | PE(H1) Baseline | PE(H1) Lindner")
out("------|-----------|-----------|--------|-----------|---------|-----------------|---------------")

old_degens = []
new_degens = []
for lv in range(1, 6):
    b = baseline["exp2_summary"][str(lv)]
    l = lindner["exp2_summary"][str(lv)]
    old_d = b["degen_rate"]
    new_d = l["degen_rate"]
    delta = new_d - old_d
    mean_cond = b["cond_number"]  # same data, same condition numbers
    dim_warn = lindner["exp1_dim_warnings"][str(lv)]
    pe_b = b.get("traj_pe_h1", b.get("pers_entropy_h1", float("nan")))
    pe_l = l.get("traj_pe_h1", l.get("pers_entropy_h1", float("nan")))
    old_degens.append(old_d)
    new_degens.append(new_d)

    out(f"  {lv}   | {old_d:>8.0%} | {new_d:>8.0%} | {delta:>+5.0%} | {mean_cond:>9.1f} | {'YES' if dim_warn else 'no':>7s} | {pe_b:>15.4f} | {pe_l:>14.4f}")

# Spearman of New Degen% with difficulty
r_degen, p_degen = stats.spearmanr(range(1, 6), new_degens)
out(f"\nSpearman(difficulty, New Degen%): r={r_degen:.3f}, p={p_degen:.4f}")

# Old degen Spearman
r_old, p_old = stats.spearmanr(range(1, 6), old_degens)
out(f"Spearman(difficulty, Old Degen%): r={r_old:.3f}, p={p_old:.4f}")

# TopologyDimensionalityWarning gradient
out("\n## TopologyDimensionalityWarning Gradient")
dim_warns = [lindner["exp1_dim_warnings"][str(lv)] for lv in range(1, 6)]
sparsity_t10 = [lindner["sparsity_metrics"][str(lv)]["top10_energy"] for lv in range(1, 6)]
out(f"Warnings by level: {['YES' if w else 'no' for w in dim_warns]}")
out(f"Top10% Energy:     {[f'{v:.4f}' for v in sparsity_t10]}")
warn_tracks_sparsity = all(dim_warns[i] <= dim_warns[i+1] for i in range(4)) or all(dim_warns[i] >= dim_warns[i+1] for i in range(4))
out(f"Warning gradient tracks sparsity gradient: {'YES' if warn_tracks_sparsity else 'NO (non-monotone)'}")

# Effective ranks from Exp 1
out("\n## Effective Rank (Experiment 1 PH)")
for lv in range(1, 6):
    er = lindner["exp1_eff_ranks"][str(lv)]
    out(f"  Level {lv}: eff_rank={er}")

# Permutation test
out("\n## Permutation Test (Experiment 4)")
out(f"Observed mean Wasserstein (baseline): {baseline['exp4_observed_mean_dist']:.4f}")
out(f"Observed mean Wasserstein (lindner):  {lindner['exp4_observed_mean_dist']:.4f}")
out(f"Baseline p-value: {baseline['exp4_p_value']:.4f} ({'SIGNIFICANT' if baseline['exp4_significant'] else 'not significant'})")
out(f"Lindner p-value:  {lindner['exp4_p_value']:.4f} ({'SIGNIFICANT' if lindner['exp4_significant'] else 'not significant'})")
out(f"Baseline z-score: {baseline['exp4_z_score']:.2f}")
out(f"Lindner z-score:  {lindner['exp4_z_score']:.2f}")

# PE(H1) correlation with Top10% Energy
out("\n## PE(H1) vs Top10% Energy Correlation")
pe_b_vals = [baseline["exp1_pe_h1"][str(lv)] for lv in range(1, 6)]
pe_l_vals = [lindner["exp1_pe_h1"][str(lv)] for lv in range(1, 6)]
r_pe_b, p_pe_b = stats.pearsonr(pe_b_vals, sparsity_t10)
r_pe_l, p_pe_l = stats.pearsonr(pe_l_vals, sparsity_t10)
out(f"Baseline PE(H1) vs Top10%E: r={r_pe_b:.3f}, p={p_pe_b:.4f}")
out(f"Lindner  PE(H1) vs Top10%E: r={r_pe_l:.3f}, p={p_pe_l:.4f}")

# Sparsity baselines (same for both runs)
out("\n## Sparsity Baselines (shared)")
out(f"Spearman(difficulty, Top10%E): r={baseline['spearman_difficulty_top10e']['r']:.3f}, p={baseline['spearman_difficulty_top10e']['p']:.4f}")

# Experiment 3 terminal ratios
out("\n## Layer-wise Transition Ratios (Experiment 3)")
for lv_str in ["1", "5"]:
    b3 = baseline["exp3_terminal_ratios"].get(lv_str, {})
    l3 = lindner["exp3_terminal_ratios"].get(lv_str, {})
    if b3:
        out(f"Level {lv_str}: terminal/non-terminal = {b3.get('ratio', '?'):.2f}x (baseline) / {l3.get('ratio', '?'):.2f}x (lindner)")

out("\n" + "=" * 90)
out("END COMPARISON")
out("=" * 90)

with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved to {OUTPUT_PATH}")
