"""
Failure taxonomy and decision pattern analysis for benchmark runs.

Classifies LLM decision-making patterns from run traces:
  - Oscillation: flipping NPIs on/off frequently
  - Over-reaction: heavy NPIs when cases are low
  - Under-reaction: no NPIs when cases are spiking
  - Never-lifting: imposing restrictions and never removing them
  - Panic-lockdown: sudden jump from minimal to maximal NPIs
  - Scratchpad usage: whether the LLM uses its planning tools

Usage:
    pixi run python scripts/analyze_decisions.py
    pixi run python scripts/analyze_decisions.py --model nemotron
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from outbreakbench.io import load_runs
from outbreakbench.metrics import npi_active, npi_stringency, npi_vector


def analyze_run(run):
    """Compute decision pattern metrics for a single run."""
    decisions = run["decisions"]
    n = len(decisions)
    if n < 2:
        return {}

    stringencies = [npi_stringency(d["policy"]) for d in decisions]
    vectors = [npi_vector(d["policy"]) for d in decisions]
    n_active = [npi_active(d["policy"]) for d in decisions]

    # --- Oscillation ---
    flips = 0
    for i in range(1, n):
        for j in range(7):
            if vectors[i][j] != vectors[i - 1][j]:
                flips += 1
    oscillation_rate = flips / ((n - 1) * 7)

    # --- Stringency dynamics ---
    max_stringency = max(stringencies)
    min_stringency = min(stringencies)
    mean_stringency = np.mean(stringencies)

    # Week-over-week changes
    deltas = [stringencies[i] - stringencies[i - 1] for i in range(1, n)]
    max_escalation = max(deltas) if deltas else 0
    max_deescalation = min(deltas) if deltas else 0

    # --- Panic lockdown ---
    panic_weeks = []
    for i in range(1, n):
        if deltas[i - 1] >= 3:
            panic_weeks.append(i + 1)

    # --- Never-lifting ---
    first_active_week = None
    last_zero_week = None
    for i, s in enumerate(stringencies):
        if s > 0 and first_active_week is None:
            first_active_week = i
        if s == 0:
            last_zero_week = i
    if first_active_week is not None and last_zero_week is not None:
        never_lifts = last_zero_week < first_active_week
    elif first_active_week is not None:
        never_lifts = True
    else:
        never_lifts = False

    weeks_at_zero = sum(1 for s in stringencies if s == 0)
    weeks_at_max = sum(1 for s in stringencies if s >= 6)

    # --- Closure usage ---
    uses_closures = any(
        d["policy"]["workplaces"] != "open" or d["policy"]["stay_at_home"]
        for d in decisions
    )
    closure_weeks = sum(
        1
        for d in decisions
        if d["policy"]["workplaces"] != "open" or d["policy"]["stay_at_home"]
    )

    # --- Scratchpad usage ---
    notes_used = sum(1 for d in decisions if d.get("notes", "").strip())

    # --- Parse failures ---
    parse_failures = sum(
        1 for d in decisions if d["justification"].startswith("[PARSE FAILURE")
    )

    # --- Classify dominant pattern ---
    pattern = "balanced"
    if oscillation_rate > 0.15:
        pattern = "oscillating"
    elif never_lifts and weeks_at_zero == 0 and mean_stringency > 3:
        pattern = "never-lifting"
    elif panic_weeks:
        pattern = "panic-lockdown"
    elif mean_stringency < 1 and run["final_results"]["cum_deaths"] > 100:
        pattern = "under-reacting"
    elif mean_stringency > 4 and run["final_results"]["cum_deaths"] < 10:
        pattern = "over-reacting"

    return {
        "model": run.get("model", "unknown"),
        "scenario": run["scenario"],
        "framing": run["framing"],
        "seed": run["seed"],
        "cum_deaths": run["final_results"]["cum_deaths"],
        "cum_infections": run["final_results"]["cum_infections"],
        "pattern": pattern,
        "oscillation_rate": round(oscillation_rate, 3),
        "mean_stringency": round(mean_stringency, 2),
        "max_escalation": max_escalation,
        "panic_weeks": panic_weeks,
        "never_lifts": never_lifts,
        "weeks_at_zero": weeks_at_zero,
        "weeks_at_max": weeks_at_max,
        "uses_closures": uses_closures,
        "closure_weeks": closure_weeks,
        "n_decisions": n,
        "notes_used": notes_used,
        "parse_failures": parse_failures,
    }


def print_summary(analyses):
    """Print a summary table of decision patterns."""
    patterns = {}
    for a in analyses:
        p = a["pattern"]
        patterns[p] = patterns.get(p, 0) + 1

    print("\n=== Decision Pattern Distribution ===")
    for p, count in sorted(patterns.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(analyses)
        print(f"  {p:20s}  {count:3d} runs  ({pct:.0f}%)")

    print(f"\n=== Aggregate Metrics ({len(analyses)} runs) ===")
    osc = [a["oscillation_rate"] for a in analyses]
    print(f"  Oscillation rate:   mean={np.mean(osc):.3f}  max={np.max(osc):.3f}")

    str_vals = [a["mean_stringency"] for a in analyses]
    print(
        f"  Mean stringency:    mean={np.mean(str_vals):.2f}  std={np.std(str_vals):.2f}"
    )

    closures = [a["closure_weeks"] for a in analyses]
    print(f"  Closure weeks:      mean={np.mean(closures):.1f}  max={max(closures)}")

    notes = [a["notes_used"] for a in analyses]
    n_decisions = [a["n_decisions"] for a in analyses]
    print(
        f"  Scratchpad usage:   mean={np.mean(notes):.1f} weeks  (of {n_decisions[0]} possible)"
    )

    failures = [a["parse_failures"] for a in analyses]
    print(f"  Parse failures:     mean={np.mean(failures):.1f}  max={max(failures)}")

    # Per-scenario breakdown
    scenarios = sorted(set(a["scenario"] for a in analyses))
    print("\n=== Per-Scenario Pattern Breakdown ===")
    print(
        f"  {'scenario':25s} {'balanced':>10} {'oscillate':>10} {'never-lift':>10} {'panic':>10} {'under':>10} {'over':>10}"
    )
    for scenario in scenarios:
        subset = [a for a in analyses if a["scenario"] == scenario]
        counts = {}
        for a in subset:
            counts[a["pattern"]] = counts.get(a["pattern"], 0) + 1
        row = [
            counts.get("balanced", 0),
            counts.get("oscillating", 0),
            counts.get("never-lifting", 0),
            counts.get("panic-lockdown", 0),
            counts.get("under-reacting", 0),
            counts.get("over-reacting", 0),
        ]
        print(
            f"  {scenario:25s} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}"
        )


def plot_taxonomy(analyses, model_name):
    """Heatmap of decision patterns by scenario × framing."""
    scenarios = sorted(set(a["scenario"] for a in analyses))
    framings = sorted(set(a["framing"] for a in analyses))
    pattern_names = [
        "balanced",
        "oscillating",
        "never-lifting",
        "panic-lockdown",
        "under-reacting",
        "over-reacting",
    ]
    pattern_colors = {
        "balanced": "#2ecc71",
        "oscillating": "#e74c3c",
        "never-lifting": "#9b59b6",
        "panic-lockdown": "#e67e22",
        "under-reacting": "#3498db",
        "over-reacting": "#f1c40f",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(scenarios) * 0.8)))

    metrics = [
        ("oscillation_rate", "Oscillation Rate\n(NPI flips / week × 7 fields)"),
        ("mean_stringency", "Mean Stringency\n(0–7 scale)"),
        ("closure_weeks", "Closure Weeks\n(workplace/stay-at-home)"),
    ]

    for ax, (key, label) in zip(axes, metrics):
        data = np.zeros((len(scenarios), len(framings)))
        for i, scenario in enumerate(scenarios):
            for j, framing in enumerate(framings):
                subset = [
                    a[key]
                    for a in analyses
                    if a["scenario"] == scenario and a["framing"] == framing
                ]
                data[i, j] = np.mean(subset) if subset else 0

        import seaborn as sns

        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            xticklabels=framings,
            yticklabels=[s.replace("_", "\n") for s in scenarios],
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_title(label, fontsize=11)

    fig.suptitle(
        f"{model_name}: Decision Pattern Analysis", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze decision patterns")
    parser.add_argument("--model", default=None, help="Filter to one model")
    parser.add_argument("--output-dir", default="outputs/runs")
    args = parser.parse_args()

    runs = load_runs(args.output_dir, model_filter=args.model)
    if not runs:
        print("No runs found.")
        return

    analyses = [analyze_run(r) for r in runs]

    models = sorted(set(a["model"] for a in analyses))
    for model_name in models:
        model_analyses = [a for a in analyses if a["model"] == model_name]
        print(f"\n{'='*60}")
        print(f"  Model: {model_name} ({len(model_analyses)} runs)")
        print(f"{'='*60}")
        print_summary(model_analyses)

        fig = plot_taxonomy(model_analyses, model_name)
        os.makedirs("outputs", exist_ok=True)
        path = f"outputs/{model_name}_decision_patterns.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  Saved {path}")

    # Save full analysis as JSON
    out_path = os.path.join(args.output_dir, "decision_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analyses, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
