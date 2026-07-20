"""
Generate weekly surveillance reports across all scenarios and plot key metrics.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from outbreakbench.scenarios import SCENARIOS, create_sim
from outbreakbench.report import (
    _economic_impact,
    _epi_indicators,
    _headline,
    _hospital_capacity,
)


def collect_weekly_metrics(scenario_key, seed=0):
    sim = create_sim(scenario_key, seed=seed)
    sim.initialize()

    report_days = list(range(7, sim["n_days"] + 1, 7))
    weeks = []

    for day in report_days:
        sim.run(until=day + 1, verbose=0)
        h = _headline(sim, day)
        epi = _epi_indicators(sim, day)
        hosp = _hospital_capacity(sim, day)
        econ = _economic_impact(sim, day)
        weeks.append(
            dict(
                day=day,
                week=day // 7,
                new_infections=h["new_infections"],
                new_deaths=h["new_deaths"],
                cum_deaths=h["cum_deaths"],
                r_eff=epi["r_eff"],
                hosp_occupied=hosp["n_hosp"],
                icu_occupied=hosp["n_icu"],
                hosp_cap=hosp["cap_hosp"],
                icu_cap=hosp["cap_icu"],
                icu_overflow_days=hosp["icu_overflow_days"],
                worker_absence_pct=econ["w_rate"],
                student_absence_pct=econ["s_rate"],
            )
        )

    return weeks


def plot_weekly_metrics(all_metrics):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    keys = list(all_metrics.keys())
    colors = sns.color_palette("colorblind", len(keys))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Weekly new infections
    ax = axes[0, 0]
    for i, k in enumerate(keys):
        weeks = all_metrics[k]
        ax.plot(
            [w["week"] for w in weeks],
            [w["new_infections"] for w in weeks],
            label=SCENARIOS[k]["name"],
            color=colors[i],
            marker=".",
            markersize=4,
        )
    ax.set_title("Weekly New Infections")
    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # 2. Weekly new deaths
    ax = axes[0, 1]
    for i, k in enumerate(keys):
        weeks = all_metrics[k]
        ax.plot(
            [w["week"] for w in weeks],
            [w["new_deaths"] for w in weeks],
            label=SCENARIOS[k]["name"],
            color=colors[i],
            marker=".",
            markersize=4,
        )
    ax.set_title("Weekly Deaths")
    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # 3. R_eff over time
    ax = axes[1, 0]
    for i, k in enumerate(keys):
        weeks = all_metrics[k]
        ax.plot(
            [w["week"] for w in weeks],
            [w["r_eff"] for w in weeks],
            label=SCENARIOS[k]["name"],
            color=colors[i],
            marker=".",
            markersize=4,
        )
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title("Estimated R_eff")
    ax.set_xlabel("Week")
    ax.set_ylabel("R_eff")
    ax.legend(fontsize=7)

    # 4. ICU occupancy vs capacity
    ax = axes[1, 1]
    for i, k in enumerate(keys):
        weeks = all_metrics[k]
        ax.plot(
            [w["week"] for w in weeks],
            [w["icu_occupied"] for w in weeks],
            label=SCENARIOS[k]["name"],
            color=colors[i],
            marker=".",
            markersize=4,
        )
    # Show both capacity lines
    caps = set()
    for k in keys:
        c = all_metrics[k][0]["icu_cap"]
        if c is not None and c not in caps:
            caps.add(c)
            ax.axhline(
                c,
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label=f"ICU capacity ({c})",
            )
    ax.set_title("ICU Occupancy")
    ax.set_xlabel("Week")
    ax.set_ylabel("Patients")
    ax.legend(fontsize=7)

    # 5. Worker absence rate
    ax = axes[2, 0]
    for i, k in enumerate(keys):
        weeks = all_metrics[k]
        ax.plot(
            [w["week"] for w in weeks],
            [w["worker_absence_pct"] for w in weeks],
            label=SCENARIOS[k]["name"],
            color=colors[i],
            marker=".",
            markersize=4,
        )
    ax.set_title("Worker Absence Rate (ages 20-64)")
    ax.set_xlabel("Week")
    ax.set_ylabel("% absent")
    ax.legend(fontsize=7)

    # 6. Cumulative deaths
    ax = axes[2, 1]
    for i, k in enumerate(keys):
        weeks = all_metrics[k]
        ax.plot(
            [w["week"] for w in weeks],
            [w["cum_deaths"] for w in weeks],
            label=SCENARIOS[k]["name"],
            color=colors[i],
            marker=".",
            markersize=4,
        )
    ax.set_title("Cumulative Deaths")
    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    plt.suptitle(
        "Weekly Surveillance Report Metrics (No Intervention)", fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig("outputs/surveillance_weekly_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved outputs/surveillance_weekly_metrics.png")


if __name__ == "__main__":
    all_metrics = {}
    for key in SCENARIOS:
        print(f"Running {SCENARIOS[key]['name']}...")
        all_metrics[key] = collect_weekly_metrics(key)
    plot_weekly_metrics(all_metrics)
