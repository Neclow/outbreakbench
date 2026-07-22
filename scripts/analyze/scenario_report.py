"""
Per-scenario report: frontier scatter, NPI stacked area, stringency ribbon.

Generates 3 PNGs per scenario:
  {scenario}_frontier.png    — deaths vs closure weeks, all models and framings
  {scenario}_npis.png        — stacked area of NPI activation over time (neutral framing)
  {scenario}_stringency.png  — stringency ribbon over time (neutral framing)

Usage:
    pixi run report
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from outbreakbench.io import load_runs
from outbreakbench.metrics import npi_stringency

_CLOSURE_FRAC = {"open": 1.0, "partial": 0.5, "full": 0.0}

NPI_KEYS = [
    ("schools", lambda v: v != "open"),
    ("workplaces", lambda v: v != "open"),
    ("masks", lambda v: v is True),
    ("mass_testing", lambda v: v is True),
    ("contact_tracing", lambda v: v is True),
    ("gathering_limits", lambda v: v != "none"),
    ("stay_at_home", lambda v: v is True),
]

NPI_LABELS = [k.replace("_", " ").title() for k, _ in NPI_KEYS]

MODEL_COLORS = {
    "nemotron-3-nano:30b": "#4C72B0",
    "qwen3:14b": "#55A868",
    "magistral:24b": "#C44E52",
    "phi4-reasoning:14b": "#8172B3",
}

MODEL_SHORT = {
    "nemotron-3-nano:30b": "Nemotron",
    "qwen3:14b": "Qwen 3",
    "magistral:24b": "Magistral",
    "phi4-reasoning:14b": "Phi-4",
}

FRAMING_MARKERS = {"economic": "D", "neutral": "o", "public_health": "s"}
FRAMING_LABELS = {"economic": "Economic", "neutral": "Neutral", "public_health": "Public health"}

NPI_COLORS = [
    "#E24A33",  # schools
    "#348ABD",  # workplaces
    "#988ED5",  # masks
    "#FBC15E",  # mass testing
    "#8EBA42",  # contact tracing
    "#FFB5B8",  # gathering limits
    "#777777",  # stay at home
]


def _closure_weeks(decisions):
    return sum(
        1
        for d in decisions
        if d["policy"]["workplaces"] != "open" or d["policy"]["stay_at_home"]
    )


def plot_frontier(runs, baselines, scenario):
    models = sorted(set(r.get("model", "unknown") for r in runs))
    framings = sorted(set(r["framing"] for r in runs))

    fig, ax = plt.subplots(figsize=(7, 5))

    # baselines — single averaged line
    sc_no_int = [b for b in baselines if b["scenario"] == scenario and b["policy"] == "no_intervention"]
    if sc_no_int:
        mean_deaths = np.mean([b["cum_deaths"] for b in sc_no_int])
        ax.axhline(mean_deaths, color="#999", ls="--", lw=0.8, zorder=1)
        ax.text(0.5, mean_deaths, "No intervention", color="#999", fontsize=8, va="bottom")

    for model in models:
        color = MODEL_COLORS.get(model, "#333")
        first = True
        for framing in framings:
            subset = [
                r for r in runs
                if r.get("model") == model
                and r["scenario"] == scenario
                and r["framing"] == framing
                and r["decisions"]
            ]
            if not subset:
                continue

            deaths = [r["final_results"]["cum_deaths"] for r in subset]
            closures = [_closure_weeks(r["decisions"]) for r in subset]

            ax.errorbar(
                np.mean(closures), np.mean(deaths),
                xerr=np.std(closures), yerr=np.std(deaths),
                fmt=FRAMING_MARKERS[framing], color=color, ms=8,
                capsize=3, capthick=1, elinewidth=1, alpha=0.85,
                label=MODEL_SHORT.get(model, model) if first else None,
                zorder=3,
            )
            first = False

    # framing legend (shapes) — separate from model colors
    for framing in framings:
        ax.scatter(
            [], [], marker=FRAMING_MARKERS[framing], color="#555", s=50,
            label=FRAMING_LABELS[framing],
        )

    ax.set_xlabel("Closure weeks (workplaces + stay-at-home)")
    ax.set_ylabel("Cumulative deaths")
    ax.set_title(scenario.replace("_", " ").title())
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-max(1, ax.get_ylim()[1] * 0.02))
    fig.tight_layout()
    return fig


def plot_npi_stacked(runs, scenario):
    models = sorted(set(r.get("model", "unknown") for r in runs))

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subset = [
            r for r in runs
            if r.get("model") == model
            and r["scenario"] == scenario
            and r["framing"] == "neutral"
            and r["decisions"]
        ]
        if not subset:
            ax.set_title(MODEL_SHORT.get(model, model))
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        min_len = min(len(r["decisions"]) for r in subset)
        weeks = np.arange(1, min_len + 1)

        layers = []
        for key, is_active in NPI_KEYS:
            fracs = np.zeros(min_len)
            for r in subset:
                for t in range(min_len):
                    if is_active(r["decisions"][t]["policy"][key]):
                        fracs[t] += 1
            fracs /= len(subset)
            layers.append(fracs)

        ax.stackplot(weeks, *layers, colors=NPI_COLORS, alpha=0.85)
        ax.set_title(MODEL_SHORT.get(model, model))
        ax.set_xlim(weeks[0], weeks[-1])
        ax.set_ylim(0, 7)
        ax.set_xlabel("Week")

    axes[0].set_ylabel("Active NPIs (fraction of seeds)")

    # shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85)
        for c in NPI_COLORS
    ]
    fig.legend(
        handles, NPI_LABELS,
        loc="upper center", ncol=4, fontsize=8,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.suptitle(
        f"{scenario.replace('_', ' ').title()} — NPI Composition (neutral framing)",
        fontsize=12, y=1.14,
    )
    fig.tight_layout()
    return fig


def plot_stringency_ribbon(runs, scenario):
    models = sorted(set(r.get("model", "unknown") for r in runs))

    fig, ax = plt.subplots(figsize=(8, 4))

    for model in models:
        subset = [
            r for r in runs
            if r.get("model") == model
            and r["scenario"] == scenario
            and r["framing"] == "neutral"
            and r["decisions"]
        ]
        if not subset:
            continue

        min_len = min(len(r["decisions"]) for r in subset)
        all_scores = np.array([
            [npi_stringency(r["decisions"][t]["policy"]) for t in range(min_len)]
            for r in subset
        ])

        weeks = np.arange(1, min_len + 1)
        mean = all_scores.mean(axis=0)
        std = all_scores.std(axis=0)
        color = MODEL_COLORS.get(model, "#333")

        ax.plot(weeks, mean, color=color, label=MODEL_SHORT.get(model, model), lw=2)
        ax.fill_between(weeks, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Week")
    ax.set_ylabel("Stringency (0–7)")
    ax.set_ylim(-0.2, 7.2)
    ax.set_title(
        f"{scenario.replace('_', ' ').title()} — Stringency Over Time (neutral framing)"
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Per-scenario report")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--plot-dir", default="outputs")
    args = parser.parse_args()

    runs = load_runs(args.output_dir)
    if not runs:
        print("No runs found.")
        return

    baselines_path = os.path.join(args.output_dir, "baselines.json")
    baselines = []
    if os.path.exists(baselines_path):
        with open(baselines_path) as f:
            baselines = json.load(f)

    os.makedirs(args.plot_dir, exist_ok=True)
    scenarios = sorted(set(r["scenario"] for r in runs))

    for scenario in scenarios:
        fig = plot_frontier(runs, baselines, scenario)
        path = os.path.join(args.plot_dir, f"{scenario}_frontier.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        fig = plot_npi_stacked(runs, scenario)
        path = os.path.join(args.plot_dir, f"{scenario}_npis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        fig = plot_stringency_ribbon(runs, scenario)
        path = os.path.join(args.plot_dir, f"{scenario}_stringency.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


if __name__ == "__main__":
    main()
