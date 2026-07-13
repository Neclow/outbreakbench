"""
Validate NPIs: test each option individually and in combination.
Runs baseline scenario with different NPI policies and compares outcomes.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from outbreakbench.npis import DEFAULT_POLICY, NPIManager
from outbreakbench.scenarios import create_sim


def run_with_policy(policy, label, start_week=0, seed=42):
    """Run baseline scenario applying a fixed policy from start_week onward."""
    sim = create_sim("baseline", seed=seed)
    sim.initialize()
    mgr = NPIManager(sim)

    n_weeks = sim["n_days"] // 7
    for week in range(n_weeks):
        if week >= start_week:
            mgr.apply(sim, policy)
        else:
            mgr.apply(sim, DEFAULT_POLICY)
        for _ in range(7):
            if sim.t < sim["n_days"]:
                sim.step()

    sim.finalize()
    r = sim.results
    deaths = int(r["cum_deaths"][-1])
    infections = int(r["cum_infections"][-1])
    peak_severe = int(max(r["n_severe"]))
    return {
        "label": label,
        "deaths": deaths,
        "infections": infections,
        "peak_severe": peak_severe,
        "results": r,
    }


def main():
    t0 = time.time()

    policies = {
        "No intervention": DEFAULT_POLICY,
        "Masks only": {**DEFAULT_POLICY, "masks": True},
        "Schools closed": {**DEFAULT_POLICY, "schools": "full"},
        "Schools partial": {**DEFAULT_POLICY, "schools": "partial"},
        "Workplaces closed": {**DEFAULT_POLICY, "workplaces": "full"},
        "Workplaces partial": {**DEFAULT_POLICY, "workplaces": "partial"},
        "Ban large gatherings": {**DEFAULT_POLICY, "gathering_limits": "ban_large"},
        "Ban all gatherings": {**DEFAULT_POLICY, "gathering_limits": "ban_all"},
        "Stay at home": {**DEFAULT_POLICY, "stay_at_home": True},
        "Mass testing": {**DEFAULT_POLICY, "mass_testing": True},
        "Contact tracing": {**DEFAULT_POLICY, "contact_tracing": True},
        "Test + trace": {
            **DEFAULT_POLICY,
            "mass_testing": True,
            "contact_tracing": True,
        },
        "Moderate bundle": {
            **DEFAULT_POLICY,
            "masks": True,
            "schools": "partial",
            "gathering_limits": "ban_large",
            "mass_testing": True,
            "contact_tracing": True,
        },
        "Full lockdown": {
            "schools": "full",
            "workplaces": "full",
            "masks": True,
            "mass_testing": True,
            "contact_tracing": True,
            "gathering_limits": "ban_all",
            "stay_at_home": True,
        },
    }

    runs = {}
    for name, policy in policies.items():
        print(f"Running: {name}...", end=" ", flush=True)
        runs[name] = run_with_policy(policy, name)
        print(
            f"deaths={runs[name]['deaths']}, "
            f"infections={runs[name]['infections']}, "
            f"peak_severe={runs[name]['peak_severe']}"
        )

    elapsed = time.time() - t0
    print(f"\nAll runs completed in {elapsed:.1f}s")

    plot_comparison(runs)
    plot_curves(runs)


def plot_comparison(runs):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    names = list(runs.keys())
    deaths = [runs[n]["deaths"] for n in names]
    infections = [runs[n]["infections"] for n in names]
    peak_severe = [runs[n]["peak_severe"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    colors = sns.color_palette("RdYlGn_r", len(names))

    # Sort by deaths for consistent ordering
    order = np.argsort(deaths)[::-1]
    sorted_names = [names[i] for i in order]

    ax = axes[0]
    sorted_deaths = [deaths[i] for i in order]
    ax.barh(range(len(sorted_names)), sorted_deaths, color=[colors[i] for i in order])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Cumulative Deaths")
    ax.set_title("Deaths by NPI Policy")

    ax = axes[1]
    sorted_inf = [infections[i] for i in order]
    ax.barh(range(len(sorted_names)), sorted_inf, color=[colors[i] for i in order])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Cumulative Infections")
    ax.set_title("Infections by NPI Policy")

    ax = axes[2]
    sorted_peak = [peak_severe[i] for i in order]
    ax.barh(range(len(sorted_names)), sorted_peak, color=[colors[i] for i in order])
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Peak Hospitalized")
    ax.set_title("Peak Severe Cases by NPI Policy")

    plt.tight_layout()
    plt.savefig("outputs/npi_comparison.png", dpi=150)
    print("Plot saved to outputs/npi_comparison.png")


def plot_curves(runs):
    sns.set_theme(style="whitegrid", font_scale=1.0)

    highlight = [
        "No intervention",
        "Masks only",
        "Stay at home",
        "Moderate bundle",
        "Full lockdown",
    ]
    colors = sns.color_palette("colorblind", len(highlight))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(highlight):
        r = runs[name]["results"]
        axes[0].plot(r["new_infections"], label=name, color=colors[i])
        axes[1].plot(r["new_deaths"], label=name, color=colors[i])
        axes[2].plot(r["n_severe"], label=name, color=colors[i])

    axes[0].set_title("Daily New Infections")
    axes[1].set_title("Daily Deaths")
    axes[2].set_title("Hospital Occupancy")
    for ax in axes:
        ax.set_xlabel("Day")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/npi_curves.png", dpi=150)
    print("Plot saved to outputs/npi_curves.png")


if __name__ == "__main__":
    main()
