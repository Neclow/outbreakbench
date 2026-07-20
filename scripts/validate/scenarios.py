"""
Validate scenarios: run each, print summary, save plots to outputs/.
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from outbreakbench.scenarios import SCENARIOS, create_sim


def age_stratified_deaths(sim):
    """Count deaths by age bucket from the people object."""
    people = sim.people
    dead = people.dead
    ages = people.age
    cutoffs = [0, 20, 40, 60, 80]
    labels = ["0-19", "20-39", "40-59", "60-79", "80+"]
    counts = []
    for i, lo in enumerate(cutoffs):
        hi = cutoffs[i + 1] if i + 1 < len(cutoffs) else 200
        counts.append(int(np.sum(dead & (ages >= lo) & (ages < hi))))
    return labels, counts


def run_and_summarize(key, seed=0):
    sim = create_sim(key, seed=seed)
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    r = sim.results
    print(f"\n{'=' * 60}")
    print(f"Scenario: {SCENARIOS[key]['name']} (seed={seed})")
    print(f"  Runtime:          {elapsed:.1f}s")
    print(f"  Total infections: {r['cum_infections'][-1]:.0f}")
    print(f"  Total deaths:     {r['cum_deaths'][-1]:.0f}")
    print(f"  Peak severe:      {max(r['n_severe']):.0f}")
    print(f"  Peak critical:    {max(r['n_critical']):.0f}")
    labels, deaths = age_stratified_deaths(sim)
    print(f"  Deaths by age:    {dict(zip(labels, deaths))}")
    return sim


def plot_all(sims):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    keys = list(sims.keys())
    colors = sns.color_palette("colorblind", len(keys))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. New infections
    ax = axes[0, 0]
    for i, k in enumerate(keys):
        ax.plot(
            sims[k].results["new_infections"],
            label=SCENARIOS[k]["name"],
            color=colors[i],
        )
    ax.set_title("Daily New Infections")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # 2. New deaths
    ax = axes[0, 1]
    for i, k in enumerate(keys):
        ax.plot(
            sims[k].results["new_deaths"], label=SCENARIOS[k]["name"], color=colors[i]
        )
    ax.set_title("Daily Deaths")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # 3. Hospital occupancy
    ax = axes[1, 0]
    for i, k in enumerate(keys):
        ax.plot(
            sims[k].results["n_severe"], label=SCENARIOS[k]["name"], color=colors[i]
        )
    ax.set_title("Hospital Occupancy (Severe Cases)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    for k in keys:
        beds = SCENARIOS[k]["pars"].get("n_beds_hosp")
        if beds:
            ax.axhline(beds, color="red", linestyle="--", alpha=0.4)
            break
    ax.legend(fontsize=8)

    # 4. Age-stratified deaths
    ax = axes[1, 1]
    labels = None
    x = None
    width = 0.18
    for i, k in enumerate(keys):
        l, counts = age_stratified_deaths(sims[k])
        if labels is None:
            labels = l
            x = np.arange(len(labels))
        offset = (i - len(keys) / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=SCENARIOS[k]["name"], color=colors[i])
    ax.set_title("Deaths by Age Group")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Deaths")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/scenario_validation.png", dpi=150)
    print(f"\nPlot saved to outputs/scenario_validation.png")


if __name__ == "__main__":
    sims = {}
    for key in SCENARIOS:
        sims[key] = run_and_summarize(key)
    plot_all(sims)
