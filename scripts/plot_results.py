"""
Visualize benchmark results from outputs/runs/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(output_dir="outputs/runs"):
    results = []
    for model_dir in sorted(os.listdir(output_dir)):
        model_path = os.path.join(output_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(model_path, fname)) as f:
                results.append(json.load(f))
    return results


def plot_outcomes_by_scenario(results, model_name):
    """Bar chart: deaths and infections by scenario, grouped by framing."""
    scenarios = sorted(set(r["scenario"] for r in results))
    framings = sorted(set(r["framing"] for r in results))
    framing_colors = {"neutral": "#4C72B0", "public_health": "#DD8452", "economic": "#55A868"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric, label in [
        (axes[0], "cum_deaths", "Cumulative Deaths"),
        (axes[1], "cum_infections", "Cumulative Infections"),
        (axes[2], "peak_severe", "Peak Hospitalized"),
    ]:
        x = np.arange(len(scenarios))
        width = 0.25
        for i, framing in enumerate(framings):
            vals = []
            errs = []
            for scenario in scenarios:
                subset = [
                    r["final_results"][metric]
                    for r in results
                    if r["scenario"] == scenario and r["framing"] == framing
                ]
                vals.append(np.mean(subset))
                errs.append(np.std(subset))
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                yerr=errs,
                label=framing,
                color=framing_colors[framing],
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name}: Outcomes by Scenario and Framing (mean ± std over seeds)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_npi_usage(results, model_name):
    """Heatmap: fraction of weeks each NPI is active, by scenario × framing."""
    scenarios = sorted(set(r["scenario"] for r in results))
    framings = sorted(set(r["framing"] for r in results))
    npi_keys = [
        ("schools", lambda v: v != "open"),
        ("workplaces", lambda v: v != "open"),
        ("masks", lambda v: v is True),
        ("mass_testing", lambda v: v is True),
        ("contact_tracing", lambda v: v is True),
        ("gathering_limits", lambda v: v != "none"),
        ("stay_at_home", lambda v: v is True),
    ]

    rows = []
    row_labels = []
    for scenario in scenarios:
        for framing in framings:
            subset = [
                r for r in results
                if r["scenario"] == scenario and r["framing"] == framing
            ]
            fracs = []
            for key, is_active in npi_keys:
                active_weeks = 0
                total_weeks = 0
                for r in subset:
                    for d in r["decisions"]:
                        total_weeks += 1
                        if is_active(d["policy"][key]):
                            active_weeks += 1
                fracs.append(active_weeks / total_weeks if total_weeks > 0 else 0)
            rows.append(fracs)
            row_labels.append(f"{scenario}\n({framing[:4]})")

    data = np.array(rows)
    col_labels = [k for k, _ in npi_keys]

    fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.4)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".0%",
        xticklabels=[c.replace("_", "\n") for c in col_labels],
        yticklabels=row_labels,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Fraction of weeks active"},
    )
    ax.set_title(f"{model_name}: NPI Usage by Scenario and Framing", fontsize=13)
    plt.tight_layout()
    return fig


def plot_weekly_decisions(results, model_name, scenario="baseline"):
    """Line chart: NPI decisions over time for one scenario, all framings."""
    framings = sorted(set(r["framing"] for r in results))
    npi_keys = ["schools", "workplaces", "masks", "mass_testing",
                "contact_tracing", "gathering_limits", "stay_at_home"]

    def npi_score(policy):
        score = 0
        if policy["schools"] == "partial": score += 0.5
        if policy["schools"] == "full": score += 1
        if policy["workplaces"] == "partial": score += 0.5
        if policy["workplaces"] == "full": score += 1
        if policy["masks"]: score += 1
        if policy["mass_testing"]: score += 1
        if policy["contact_tracing"]: score += 1
        if policy["gathering_limits"] == "ban_large": score += 0.5
        if policy["gathering_limits"] == "ban_all": score += 1
        if policy["stay_at_home"]: score += 1
        return score

    framing_colors = {"neutral": "#4C72B0", "public_health": "#DD8452", "economic": "#55A868"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    for framing in framings:
        subset = [
            r for r in results
            if r["scenario"] == scenario and r["framing"] == framing
        ]
        all_scores = []
        for r in subset:
            scores = [npi_score(d["policy"]) for d in r["decisions"]]
            all_scores.append(scores)

        all_scores = np.array(all_scores)
        weeks = np.arange(1, all_scores.shape[1] + 1)
        mean = all_scores.mean(axis=0)
        std = all_scores.std(axis=0)
        ax.plot(weeks, mean, label=framing, color=framing_colors[framing])
        ax.fill_between(weeks, mean - std, mean + std, alpha=0.2, color=framing_colors[framing])

    ax.set_ylabel("NPI Stringency Score (0-7)")
    ax.set_title(f"{model_name}: NPI Stringency Over Time — {scenario}")
    ax.legend()
    ax.set_xlim(1, 25)

    ax = axes[1]
    for framing in framings:
        subset = [
            r for r in results
            if r["scenario"] == scenario and r["framing"] == framing
        ]
        all_inf = []
        for r in subset:
            inf = [d["policy"]["masks"] for d in r["decisions"]]
            all_inf.append(inf)

    # Show new infections from a single seed for reference
    ref = [r for r in results if r["scenario"] == scenario and r["framing"] == "neutral" and r["seed"] == 0]
    if ref:
        ax.text(0.5, 0.5, f"Final outcomes (seed 0, neutral): "
                f"{ref[0]['final_results']['cum_deaths']} deaths, "
                f"{ref[0]['final_results']['cum_infections']} infections",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_axis_off()

    plt.tight_layout()
    return fig


def npi_stringency(policy):
    score = 0
    if policy["schools"] == "partial": score += 0.5
    if policy["schools"] == "full": score += 1
    if policy["workplaces"] == "partial": score += 0.5
    if policy["workplaces"] == "full": score += 1
    if policy["masks"]: score += 1
    if policy["mass_testing"]: score += 1
    if policy["contact_tracing"]: score += 1
    if policy["gathering_limits"] == "ban_large": score += 0.5
    if policy["gathering_limits"] == "ban_all": score += 1
    if policy["stay_at_home"]: score += 1
    return score


def load_baselines(output_dir="outputs/runs"):
    path = os.path.join(output_dir, "baselines.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def plot_policy_frontier(results, baselines, model_name):
    """Cost-effectiveness plane: deaths vs NPI stringency, with baselines."""
    scenarios = sorted(set(r["scenario"] for r in results))
    framing_colors = {"neutral": "#4C72B0", "public_health": "#DD8452", "economic": "#55A868"}
    framing_markers = {"neutral": "o", "public_health": "s", "economic": "D"}

    n_cols = 3
    n_rows = (len(scenarios) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]

        # Baselines
        no_int = [b for b in baselines if b["scenario"] == scenario and b["policy"] == "no_intervention"]
        full_lock = [b for b in baselines if b["scenario"] == scenario and b["policy"] == "full_lockdown"]

        if no_int:
            ax.scatter(
                [0] * len(no_int),
                [b["cum_deaths"] for b in no_int],
                marker="X", s=100, c="gray", zorder=5, label="No intervention",
            )
        if full_lock:
            n_weeks = 25
            ax.scatter(
                [7 * n_weeks] * len(full_lock),
                [b["cum_deaths"] for b in full_lock],
                marker="X", s=100, c="black", zorder=5, label="Full lockdown",
            )

        # LLM runs
        for framing in sorted(framing_colors.keys()):
            subset = [
                r for r in results
                if r["scenario"] == scenario and r["framing"] == framing
            ]
            if not subset:
                continue
            stringencies = []
            deaths = []
            for r in subset:
                total = sum(npi_stringency(d["policy"]) for d in r["decisions"])
                stringencies.append(total)
                deaths.append(r["final_results"]["cum_deaths"])
            ax.scatter(
                stringencies, deaths,
                marker=framing_markers[framing], s=60,
                c=framing_colors[framing], alpha=0.8,
                label=framing, zorder=4,
            )

        ax.set_title(scenario.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Cumulative NPI Stringency\n(sum of weekly scores)")
        ax.set_ylabel("Cumulative Deaths")
        ax.legend(fontsize=7, loc="upper right")

    for idx in range(len(scenarios), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"{model_name}: Policy Frontier — Deaths vs Intervention Stringency",
        fontsize=14,
    )
    plt.tight_layout()
    return fig


def main():
    results = load_results()
    if not results:
        print("No results found in outputs/runs/")
        return

    baselines = load_baselines()
    models = sorted(set(r.get("model", "unknown") for r in results))
    os.makedirs("outputs", exist_ok=True)

    for model_name in models:
        model_results = [r for r in results if r.get("model") == model_name]
        print(f"\n{model_name}: {len(model_results)} runs")

        fig = plot_outcomes_by_scenario(model_results, model_name)
        path = f"outputs/{model_name}_outcomes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

        fig = plot_npi_usage(model_results, model_name)
        path = f"outputs/{model_name}_npi_usage.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

        if baselines:
            fig = plot_policy_frontier(model_results, baselines, model_name)
            path = f"outputs/{model_name}_frontier.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")

        scenarios = sorted(set(r["scenario"] for r in model_results))
        for scenario in scenarios:
            fig = plot_weekly_decisions(model_results, model_name, scenario)
            path = f"outputs/{model_name}_timeline_{scenario}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
