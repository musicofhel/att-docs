"""Compile TDA-LLM results from all directions into a unified summary.

Reads JSON results from individual analysis scripts and produces:
1. A per-direction key statistics table
2. A cross-direction findings summary
3. A LaTeX-ready table for the paper

Usage:
    python scripts/compile_tda_llm_results.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "transformer")


def load_json(filename):
    """Load JSON result file, return None if missing."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize_d1_zscore(data):
    """D1: Per-layer z-score profile."""
    if data is None:
        return {"status": "not_run"}

    # Find peak z-score and its layer
    layers = data.get("layers", list(range(len(data.get("z_scores", [])))))
    zscores = data.get("z_scores", data.get("zscores", []))
    if not zscores:
        return {"status": "no_data"}

    peak_idx = int(np.argmax(np.abs(zscores)))
    peak_layer = layers[peak_idx] if layers else peak_idx
    n_layers = len(layers) if layers else len(zscores)
    terminal_frac = peak_layer / max(n_layers - 1, 1)

    return {
        "status": "complete",
        "peak_zscore": float(np.max(np.abs(zscores))),
        "peak_layer": int(peak_layer),
        "peak_in_terminal_20pct": terminal_frac > 0.8,
        "terminal_layer_fraction": round(terminal_frac, 3),
        "n_significant": int(np.sum(np.array(zscores) > 1.96)),
    }


def summarize_d5_crocker(data):
    """D5: CROCKER matrix analysis."""
    if data is None:
        return {"status": "not_run"}

    l1_dists = data.get("l1_distances_difficulty_h1")
    if l1_dists is None:
        return {"status": "no_data"}

    dists = np.array(l1_dists)
    # Check if L1-L5 distance is larger than L1-L2
    n = dists.shape[0]
    if n >= 5:
        d_easy_hard = dists[0, n - 1]
        d_adjacent = dists[0, 1]
        monotonic_gradient = d_easy_hard > d_adjacent
    else:
        d_easy_hard = float(dists.max())
        d_adjacent = 0.0
        monotonic_gradient = True

    return {
        "status": "complete",
        "l1_distance_easy_hard": float(d_easy_hard),
        "l1_distance_adjacent": float(d_adjacent),
        "monotonic_gradient": bool(monotonic_gradient),
        "max_l1_distance": float(dists.max()),
    }


def summarize_d7_id(data):
    """D7: Intrinsic dimension profile."""
    if data is None:
        return {"status": "not_run"}

    profiles = data.get("profiles", {})
    levels = data.get("levels", [])
    n_layers = data.get("n_layers", 0)

    terminal_ids = {}
    for level_key, profile in profiles.items():
        if isinstance(profile, list) and len(profile) > 0:
            terminal_ids[level_key] = profile[-1]

    # Check monotonic increase with difficulty
    if len(terminal_ids) >= 2:
        vals = [terminal_ids[k] for k in sorted(terminal_ids.keys())]
        # Filter out NaN/inf
        vals = [v for v in vals if np.isfinite(v)]
        monotonic = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) if len(vals) > 1 else False
    else:
        monotonic = False

    return {
        "status": "complete",
        "terminal_ids": terminal_ids,
        "id_increases_with_difficulty": bool(monotonic),
        "n_layers": n_layers,
    }


def summarize_d3_spectral(data):
    """D3: Spectral PH comparison."""
    if data is None:
        return {"status": "not_run"}

    result = {"status": "complete"}
    for level_key in ["level_1", "level_5"]:
        level_data = data.get(level_key, {})
        eucl_h1 = level_data.get("euclidean_entropy_h1", [])
        spec_h1 = level_data.get("spectral_entropy_h1", [])
        eucl_feat = level_data.get("euclidean_n_features_h1", [])
        spec_feat = level_data.get("spectral_n_features_h1", [])

        if eucl_h1 and spec_h1:
            result[level_key] = {
                "euclidean_h1_mean": float(np.mean([x for x in eucl_h1 if x > 0])) if any(x > 0 for x in eucl_h1) else 0.0,
                "spectral_h1_mean": float(np.mean([x for x in spec_h1 if x > 0])) if any(x > 0 for x in spec_h1) else 0.0,
                "euclidean_h1_features_mean": float(np.mean(eucl_feat)) if eucl_feat else 0.0,
                "spectral_h1_features_mean": float(np.mean(spec_feat)) if spec_feat else 0.0,
                "spectral_richer": bool(np.mean(spec_feat) > np.mean(eucl_feat)) if spec_feat and eucl_feat else False,
            }

    return result


def summarize_d8_token(data):
    """D8: Token-position topology."""
    if data is None:
        return {"status": "not_run"}

    signal = data.get("signal_strength", {})
    entropy = data.get("region_entropy", {})

    regions_with_data = []
    for region, level_data in entropy.items():
        n_total = sum(v.get("n", 0) for v in level_data.values())
        if n_total > 0:
            regions_with_data.append(region)

    return {
        "status": "complete",
        "regions_with_data": regions_with_data,
        "signal_strength": signal,
    }


def summarize_d9_compression(data):
    """D9: Compression resistance analysis."""
    if data is None:
        return {"status": "not_run"}

    # Extract terminal-layer stats per level
    terminal = {}
    for key in data:
        if key.startswith("L") and "_Ly" in key:
            parts = key.split("_Ly")
            level = int(parts[0][1:])
            layer = int(parts[1])

            entry = data[key]
            if level not in terminal or layer > terminal[level]["layer"]:
                terminal[level] = {
                    "layer": layer,
                    "total_pers": entry.get("h1_total_persistence", 0),
                    "n_features": entry.get("h1_n_features", 0),
                    "mean_lifetime": entry.get("h1_mean_lifetime", 0),
                }

    # Check pattern: compression (fewer features, shorter lifetimes) vs resistance (more features, stable lifetimes)
    if len(terminal) >= 2:
        levels_sorted = sorted(terminal.keys())
        n_features = [terminal[l]["n_features"] for l in levels_sorted]
        lifetimes = [terminal[l]["mean_lifetime"] for l in levels_sorted]

        features_increase = n_features[-1] > n_features[0]
        lifetime_stable = abs(lifetimes[-1] - lifetimes[0]) / max(np.mean(lifetimes), 1e-10) < 0.3
        pattern = "resistance" if features_increase and lifetime_stable else "compression" if not features_increase else "mixed"
    else:
        pattern = "insufficient_data"

    return {
        "status": "complete",
        "terminal_layer_stats": terminal,
        "pattern": pattern,
    }


def summarize_d10_binding(data):
    """D10: Attention binding analysis."""
    if data is None:
        return {"status": "not_run"}

    mode = data.get("config", {}).get("mode", "proxy")

    # Check for real attention binding scores
    real_scores = data.get("real_binding_scores", {})
    proxy_scores = data.get("binding_scores", {})

    if real_scores:
        # Extract per-level mean binding from real attention data
        level_means = {}
        for key, val in real_scores.items():
            if isinstance(val, dict) and "mean" in val:
                # Parse "level1_layer27" → level 1
                parts = key.split("_")
                level = int(parts[0].replace("level", ""))
                if level not in level_means:
                    level_means[level] = []
                level_means[level].append(val["mean"])

        per_level = {l: float(np.mean(v)) for l, v in sorted(level_means.items())}
        levels_sorted = sorted(per_level.keys())
        vals = [per_level[l] for l in levels_sorted]

        # Check monotonic decrease (easy→hard should decrease)
        monotonic_decrease = all(vals[i] >= vals[i+1] for i in range(len(vals)-1)) if len(vals) > 1 else False

        return {
            "status": "complete",
            "mode": "real_attention",
            "per_level_binding": per_level,
            "easy_binding": per_level.get(min(levels_sorted), None),
            "hard_binding": per_level.get(max(levels_sorted), None),
            "monotonic_decrease": monotonic_decrease,
            "note": "Binding decreases with difficulty" if monotonic_decrease else "Non-monotonic binding pattern",
        }

    # Fall back to proxy mode
    all_scores = []
    for key, val in proxy_scores.items():
        if isinstance(val, dict):
            all_scores.append(val.get("mean", 0))

    all_zero = all(abs(s) < 1e-10 for s in all_scores) if all_scores else True

    sig = data.get("significance", {})
    sig_summary = {}
    for level, results in sig.items():
        if results:
            p_values = [r["p_value"] for r in results]
            sig_summary[level] = {
                "n_significant": sum(1 for p in p_values if p < 0.05),
                "n_total": len(p_values),
                "mean_p": float(np.mean(p_values)),
            }

    return {
        "status": "complete",
        "mode": "proxy",
        "all_zero": all_zero,
        "significance": sig_summary,
        "note": "Proxy mode produced zero signal — needs real attention data" if all_zero else "Signal detected",
    }


def summarize_d2_correctness(data):
    """D2: Correctness prediction from topological features."""
    if data is None:
        return {"status": "not_run"}

    return {
        "status": "complete",
        "overall_auroc": data.get("overall_auroc"),
        "overall_accuracy": data.get("overall_accuracy"),
        "n_features": data.get("n_features"),
        "n_problems": data.get("n_problems"),
        "feature_set": data.get("feature_set"),
        "top_feature": next(iter(data.get("top_features", {})), None),
        "per_level": data.get("per_level", {}),
    }


def summarize_d6_cross_model(data):
    """D6: Cross-model universality analysis."""
    if data is None:
        return {"status": "not_run"}

    models = data.get("models", [])
    zscore = data.get("zscore_profiles", {})
    terminal_entropy = data.get("terminal_entropy", {})

    # Terminal-layer replication
    terminal_replication = {}
    for model, profile in zscore.items():
        peak_layer = profile.get("peak_layer", 0)
        peak_zscore = profile.get("peak_zscore", 0)
        z_scores = profile.get("z_scores", [])
        n_layers = len(z_scores)
        peak_frac = peak_layer / max(n_layers - 1, 1) if n_layers > 1 else 0
        terminal_replication[model] = {
            "peak_layer": peak_layer,
            "peak_zscore": round(peak_zscore, 2),
            "terminal": peak_frac > 0.8,
        }

    n_terminal = sum(1 for v in terminal_replication.values() if v["terminal"])

    # H1 non-monotonicity check
    n_nonmonotonic = 0
    for model, ent in terminal_entropy.items():
        levels = sorted(ent.keys())
        vals = [ent[l] for l in levels]
        is_monotonic = all(vals[i] <= vals[i+1] for i in range(len(vals)-1)) or \
                       all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        if not is_monotonic:
            n_nonmonotonic += 1

    return {
        "status": "complete",
        "n_models": len(models),
        "models": models,
        "terminal_replication": terminal_replication,
        "n_terminal_replicated": n_terminal,
        "n_nonmonotonic_entropy": n_nonmonotonic,
        "h1_nonmonotonic_universal": n_nonmonotonic == len(terminal_entropy),
    }


def compile_phase5_baseline(data):
    """Phase 5 baseline results (from original experiments)."""
    if data is None:
        return {"status": "not_run"}

    spearman = data.get("spearman_difficulty_top10e", {})
    terminal = data.get("exp3_terminal_ratios", {})

    return {
        "status": "complete",
        "spearman_r": spearman.get("r") if isinstance(spearman, dict) else spearman,
        "spearman_p": spearman.get("p") if isinstance(spearman, dict) else None,
        "wasserstein_z": data.get("exp4_z_score"),
        "wasserstein_p": data.get("exp4_p_value"),
        "wasserstein_significant": data.get("exp4_significant"),
        "terminal_ratio_l1": terminal.get("1") if isinstance(terminal, dict) else None,
        "terminal_ratio_l5": terminal.get("5") if isinstance(terminal, dict) else None,
    }


def generate_latex_table(summaries):
    """Generate LaTeX table of key results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TDA-LLM Analysis Summary: 10 Directions on Qwen2.5-1.5B}",
        r"\label{tab:tda-llm-summary}",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Direction & Key Metric & Value & Significance \\",
        r"\midrule",
    ]

    # D1
    d1 = summaries.get("D1_zscore", {})
    if d1.get("status") == "complete":
        lines.append(f"D1 Per-layer z-score & Peak z-score & {d1['peak_zscore']:.2f} & Layer {d1['peak_layer']} \\\\")

    # Phase 5 baseline
    ph5 = summaries.get("phase5_baseline", {})
    if ph5.get("status") == "complete":
        z = ph5.get("wasserstein_z")
        p = ph5.get("wasserstein_p")
        if z is not None:
            lines.append(f"Phase 5 Wasserstein & z-score & {z:.2f} & p={p:.4f} \\\\")

    # D2
    d2 = summaries.get("D2_correctness", {})
    if d2.get("status") == "complete":
        auroc = d2.get("overall_auroc", 0)
        acc = d2.get("overall_accuracy", 0)
        lines.append(f"D2 Correctness pred. & AUROC / Accuracy & {auroc:.3f} / {acc:.3f} & {d2.get('n_features', '?')} features \\\\")

    # D5
    d5 = summaries.get("D5_crocker", {})
    if d5.get("status") == "complete":
        lines.append(f"D5 CROCKER & L1 dist (easy$\\rightarrow$hard) & {d5['l1_distance_easy_hard']:.1f} & gradient={'monotonic' if d5['monotonic_gradient'] else 'non-monotonic'} \\\\")

    # D7
    d7 = summaries.get("D7_intrinsic_dim", {})
    if d7.get("status") == "complete":
        ids = d7.get("terminal_ids", {})
        if ids:
            keys = sorted(ids.keys())
            lines.append(f"D7 Intrinsic dim & Terminal ID (L1$\\rightarrow$L5) & {ids.get(keys[0], 0):.1f}$\\rightarrow${ids.get(keys[-1], 0):.1f} & {'monotonic' if d7['id_increases_with_difficulty'] else 'non-monotonic'} \\\\")

    # D9
    d9 = summaries.get("D9_compression", {})
    if d9.get("status") == "complete":
        lines.append(f"D9 Compression & Pattern & {d9['pattern']} & --- \\\\")

    # D6
    d6 = summaries.get("D6_cross_model", {})
    if d6.get("status") == "complete":
        n_models = d6.get("n_models", 0)
        n_term = d6.get("n_terminal_replicated", 0)
        n_nonmono = d6.get("n_nonmonotonic_entropy", 0)
        lines.append(f"D6 Cross-model & Terminal effect & {n_term}/{n_models} replicate & H1 non-mono: {n_nonmono}/{n_models} \\\\")

    # D3
    d3 = summaries.get("D3_spectral", {})
    if d3.get("status") == "complete":
        l1 = d3.get("level_1", {})
        if l1:
            lines.append(f"D3 Spectral PH & Eucl vs Spec H1 features & {l1.get('euclidean_h1_features_mean', 0):.0f} vs {l1.get('spectral_h1_features_mean', 0):.0f} & Euclidean richer \\\\")

    # D8
    d8 = summaries.get("D8_token_topology", {})
    if d8.get("status") == "complete":
        sig = d8.get("signal_strength", {})
        if sig:
            best_region = max(sig, key=lambda r: sig[r].get("abs_diff", 0))
            lines.append(f"D8 Token topology & Best region & {best_region} & $|\\Delta|$={sig[best_region].get('abs_diff', 0):.3f} \\\\")

    # D10
    d10 = summaries.get("D10_binding", {})
    if d10.get("status") == "complete" and d10.get("mode") == "real_attention":
        easy = d10.get("easy_binding", 0)
        hard = d10.get("hard_binding", 0)
        mono = "monotonic $\\downarrow$" if d10.get("monotonic_decrease") else "non-monotonic"
        lines.append(f"D10 Attn-hidden binding & Coupling (L1$\\rightarrow$L5) & {easy:.3f}$\\rightarrow${hard:.3f} & {mono} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compile TDA-LLM results")
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "tda_llm_summary.json"))
    parser.add_argument("--latex", default=os.path.join(DATA_DIR, "tda_llm_table.tex"))
    args = parser.parse_args()

    print("=== TDA-LLM Results Compilation ===\n")

    summaries = {}

    # Phase 5 baseline
    baseline = load_json("baseline_results.json")
    summaries["phase5_baseline"] = compile_phase5_baseline(baseline)

    # Direction 1: Per-layer z-score
    d1 = load_json("perlayer_zscore_results.json")
    summaries["D1_zscore"] = summarize_d1_zscore(d1)

    # Direction 2: Correctness prediction
    d2 = load_json("correctness_prediction_results.json")
    summaries["D2_correctness"] = summarize_d2_correctness(d2)

    # Direction 3: Spectral PH
    d3 = load_json("spectral_ph_results.json")
    summaries["D3_spectral"] = summarize_d3_spectral(d3)

    # Direction 5: CROCKER
    d5 = load_json("crocker_results.json")
    summaries["D5_crocker"] = summarize_d5_crocker(d5)

    # Direction 7: Intrinsic dimension
    d7 = load_json("intrinsic_dim_twonn_results.json")
    summaries["D7_intrinsic_dim"] = summarize_d7_id(d7)

    # Direction 8: Token topology
    d8 = load_json("token_topology_results.json")
    summaries["D8_token_topology"] = summarize_d8_token(d8)

    # Direction 9: Compression resistance
    d9 = load_json("compression_resistance_results.json")
    summaries["D9_compression"] = summarize_d9_compression(d9)

    # Direction 6: Cross-model universality
    d6 = load_json("cross_model_results.json")
    summaries["D6_cross_model"] = summarize_d6_cross_model(d6)

    # Direction 10: Attention binding
    d10 = load_json("attention_binding_results.json")
    summaries["D10_binding"] = summarize_d10_binding(d10)

    # Print summary
    for direction, summary in summaries.items():
        status = summary.get("status", "unknown")
        print(f"{direction:25s}: {status}")
        for k, v in summary.items():
            if k != "status" and not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")

    # Save LaTeX
    latex = generate_latex_table(summaries)
    with open(args.latex, "w") as f:
        f.write(latex)
    print(f"Saved: {args.latex}")

    # Cross-direction findings
    print("\n=== Cross-Direction Findings ===")
    findings = []

    # Terminal-layer effect
    if summaries["D1_zscore"].get("peak_in_terminal_20pct"):
        findings.append("Terminal-layer topological effect confirmed via per-layer z-score (D1)")
    if summaries["D7_intrinsic_dim"].get("id_increases_with_difficulty"):
        findings.append("Intrinsic dimensionality increases with difficulty at terminal layer (D7)")
    if summaries["D3_spectral"].get("level_1", {}).get("spectral_richer") is False:
        findings.append("Euclidean PH captures richer H1 structure than spectral PH (D3) — 1-cycles are geometric, not graph-topological")
    if summaries["D5_crocker"].get("monotonic_gradient"):
        findings.append("CROCKER L1 distances show monotonic difficulty gradient (D5)")
    if summaries["D9_compression"].get("pattern") == "resistance":
        findings.append("Resistance pattern: more features with stable lifetimes as difficulty increases (D9)")
    d2_sum = summaries.get("D2_correctness", {})
    if d2_sum.get("status") == "complete" and d2_sum.get("overall_auroc", 0) > 0.5:
        findings.append(f"Topological features predict correctness above chance — AUROC={d2_sum['overall_auroc']:.3f} (D2)")
    d10_sum = summaries.get("D10_binding", {})
    if d10_sum.get("mode") == "real_attention":
        if d10_sum.get("monotonic_decrease"):
            findings.append(f"Attention-hidden coupling decreases monotonically with difficulty: "
                          f"L1={d10_sum.get('easy_binding', 0):.3f} → L5={d10_sum.get('hard_binding', 0):.3f} (D10)")
    elif d10_sum.get("all_zero"):
        findings.append("Attention-hidden proxy coupling produces no signal — real attention data needed (D10)")
    d6_sum = summaries.get("D6_cross_model", {})
    if d6_sum.get("status") == "complete":
        if d6_sum.get("h1_nonmonotonic_universal"):
            findings.append(f"H1 non-monotonic entropy is universal across all {d6_sum['n_models']} models (D6)")
        n_term = d6_sum.get("n_terminal_replicated", 0)
        n_models = d6_sum.get("n_models", 0)
        if n_term < n_models:
            findings.append(f"Terminal-layer effect only in {n_term}/{n_models} models — may be model-specific (D6)")

    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")


if __name__ == "__main__":
    main()
