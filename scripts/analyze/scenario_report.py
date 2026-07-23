"""
Per-scenario report: frontier scatter, NPI stacked area, stringency ribbon,
and epidemic time series.

Generates PNGs per scenario under outputs/scenarios/{scenario}/:
  frontier.png           — deaths vs closure weeks, all models and framings
  npis.png               — stacked area of NPI activation over time (neutral)
  stringency.png         — stringency ribbon over time (neutral)
  active_infectious.png  — active infectious count over time (neutral)
  cum_deaths.png         — cumulative deaths over time (neutral)
  quarantine.png         — quarantined individuals over time (neutral)

The last three require time_series data in runs (added 2026-07-23).

Usage:
    pixi run analyze-scenarios
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def _display_name(model):
    return model.split("/")[-1] if "/" in model else model


def _closure_weeks(decisions):
    return sum(
        1
        for d in decisions
        if d["policy"]["workplaces"] != "open" or d["policy"]["stay_at_home"]
    )


def plot_frontier(runs, baselines, scenario):
    models = sorted(set(r.get("model", "unknown") for r in runs))
    framings = sorted(set(r["framing"] for r in runs))

    fig, ax = plt.subplots(figsize=(10, 5))

    # baselines — single averaged line
    sc_no_int = [b for b in baselines if b["scenario"] == scenario and b["policy"] == "no_intervention"]
    if sc_no_int:
        mean_deaths = np.mean([b["cum_deaths"] for b in sc_no_int])
        ax.axhline(mean_deaths, color="#999", ls="--", lw=0.8, zorder=1)
        ax.text(0.5, mean_deaths, "No intervention", color="#999", fontsize=8, va="bottom")

    plotted_models = set()
    for model in models:
        color = MODEL_COLORS.get(model, "#333")
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

            plotted_models.add(model)
            deaths = [r["final_results"]["cum_deaths"] for r in subset]
            closures = [_closure_weeks(r["decisions"]) for r in subset]

            ax.errorbar(
                np.mean(closures), np.mean(deaths),
                xerr=np.std(closures), yerr=np.std(deaths),
                fmt=FRAMING_MARKERS[framing], color=color, ms=8,
                capsize=3, capthick=1, elinewidth=1, alpha=0.85,
                zorder=3,
            )

    # Legend outside plot: models then framings, stacked on the right
    model_handles = [
        plt.Line2D([], [], color=MODEL_COLORS.get(m, "#333"), marker="o", ls="none", ms=8)
        for m in models if m in plotted_models
    ]
    model_labels = [_display_name(m) for m in models if m in plotted_models]

    framing_handles = [
        plt.Line2D([], [], marker=FRAMING_MARKERS[f], color="#555", ls="none", ms=8)
        for f in framings
    ]
    framing_labels = [FRAMING_LABELS[f] for f in framings]

    leg1 = fig.legend(
        model_handles, model_labels, fontsize=8, title="Model", title_fontsize=9,
        frameon=False, loc="upper left", bbox_to_anchor=(0.68, 0.95),
    )
    fig.legend(
        framing_handles, framing_labels, fontsize=8, title="Framing", title_fontsize=9,
        frameon=False, loc="upper left", bbox_to_anchor=(0.68, 0.45),
    )

    ax.set_xlabel("Closure weeks (workplaces + stay-at-home)")
    ax.set_ylabel("Cumulative deaths")
    ax.set_title(f"Scenario: {scenario.replace('_', ' ').title()}\nDeaths vs. Closure Weeks")
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-max(1, ax.get_ylim()[1] * 0.02))
    sns.despine(ax=ax)
    fig.subplots_adjust(right=0.65)
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
            ax.set_title(model)
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
        ax.set_title(model)
        ax.set_xlim(weeks[0], weeks[-1])
        ax.set_ylim(0, 7)
        ax.set_xlabel("Week")

    axes[0].set_ylabel("Active NPIs (fraction of seeds)")

    for a in axes:
        sns.despine(ax=a)

    # shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85)
        for c in NPI_COLORS
    ]
    fig.suptitle(
        f"Scenario: {scenario.replace('_', ' ').title()}\nNPI Composition (neutral framing)",
        fontsize=12, y=1.18,
    )
    fig.legend(
        handles, NPI_LABELS,
        loc="upper center", ncol=4, fontsize=8,
        bbox_to_anchor=(0.5, 1.05),
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

        ax.plot(weeks, mean, color=color, label=model, lw=2)
        ax.fill_between(weeks, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Week")
    ax.set_ylabel("Stringency (0–7)")
    ax.set_ylim(-0.2, 7.2)
    ax.set_title(
        f"Scenario: {scenario.replace('_', ' ').title()}\nStringency Over Time (neutral framing)"
    )
    ax.legend(
        fontsize=8, title="Model", title_fontsize=9,
        frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0),
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


TS_PLOTS = [
    ("n_infectious", "Active infectious", "Active Infectious Over Time"),
    ("cum_deaths", "Cumulative deaths", "Cumulative Deaths Over Time"),
    ("n_quarantined", "Quarantined individuals", "Quarantine Over Time"),
]


def plot_time_series(runs, scenario, ts_key, ylabel, title_suffix):
    models = sorted(set(r.get("model", "unknown") for r in runs))

    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False

    for model in models:
        subset = [
            r for r in runs
            if r.get("model") == model
            and r["scenario"] == scenario
            and r["framing"] == "neutral"
            and r["decisions"]
            and r.get("time_series")
        ]
        if not subset:
            continue

        min_len = min(len(r["time_series"][ts_key]) for r in subset)
        all_series = np.array([
            r["time_series"][ts_key][:min_len] for r in subset
        ], dtype=float)

        days = np.arange(min_len)
        mean = all_series.mean(axis=0)
        std = all_series.std(axis=0)
        color = MODEL_COLORS.get(model, "#333")

        ax.plot(days, mean, color=color, label=_display_name(model), lw=1.5)
        ax.fill_between(days, mean - std, mean + std, color=color, alpha=0.15)
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{scenario.replace('_', ' ').title()}:\n{title_suffix} (neutral framing)"
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        fontsize=8, title="Model", title_fontsize=9,
        frameon=False, loc="upper left", bbox_to_anchor=(0.72, 0.95),
    )
    sns.despine(ax=ax)
    fig.subplots_adjust(right=0.70)
    return fig


def plot_baseline_variance(runs, baselines):
    scenarios = sorted(set(b["scenario"] for b in baselines))
    all_models = sorted(set(r.get("model") for r in runs if r.get("model")))
    all_framings = sorted(set(r["framing"] for r in runs))

    sim_sds, llm_sds, labels = [], [], []
    for scenario in scenarios:
        no_int = [b["cum_deaths"] for b in baselines
                  if b["scenario"] == scenario and b["policy"] == "no_intervention"]
        if not no_int:
            continue

        combo_sds = []
        for model in all_models:
            for framing in all_framings:
                subset = [r for r in runs
                          if r.get("model") == model
                          and r["scenario"] == scenario
                          and r["framing"] == framing
                          and r["decisions"]]
                if len(subset) >= 2:
                    combo_sds.append(
                        np.std([r["final_results"]["cum_deaths"] for r in subset])
                    )

        if not combo_sds:
            continue

        sim_sds.append(np.std(no_int))
        llm_sds.append(np.mean(combo_sds))
        labels.append(scenario.replace("_", " ").title())

    if not labels:
        return None

    x = np.arange(len(labels))
    w = 0.35

    sim_sds = np.array(sim_sds, dtype=float)
    llm_sds = np.array(llm_sds, dtype=float)
    sim_sds[sim_sds == 0] = 0.1
    llm_sds[llm_sds == 0] = 0.1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, sim_sds, w, label="Fixed policy (no intervention)",
           color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, llm_sds, w, label="LLM-driven policy",
           color="#C44E52", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("SD of Cumulative Deaths (across seeds)")
    ax.set_title("Sources of Outcome Variability Across Seeds")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Per-scenario report")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--plot-dir", default="outputs/scenarios")
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

    if baselines and runs:
        fig = plot_baseline_variance(runs, baselines)
        if fig:
            os.makedirs(args.plot_dir, exist_ok=True)
            path = os.path.join(args.plot_dir, "baseline_variance.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")

    scenarios = sorted(set(r["scenario"] for r in runs))

    for scenario in scenarios:
        scenario_dir = os.path.join(args.plot_dir, scenario)
        os.makedirs(scenario_dir, exist_ok=True)

        fig = plot_frontier(runs, baselines, scenario)
        path = os.path.join(scenario_dir, "frontier.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        fig = plot_npi_stacked(runs, scenario)
        path = os.path.join(scenario_dir, "npis.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        fig = plot_stringency_ribbon(runs, scenario)
        path = os.path.join(scenario_dir, "stringency.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        for ts_key, ylabel, title_suffix in TS_PLOTS:
            fig = plot_time_series(runs, scenario, ts_key, ylabel, title_suffix)
            if fig:
                fname = {
                    "n_infectious": "active_infectious",
                    "cum_deaths": "cum_deaths",
                    "n_quarantined": "quarantine",
                }[ts_key]
                path = os.path.join(scenario_dir, f"{fname}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {path}")


if __name__ == "__main__":
    main()
