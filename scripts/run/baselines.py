"""
Generate no-intervention and full-lockdown baselines for all scenarios/seeds.
These serve as reference points on the cost-effectiveness plane.
"""

import json
import os

from outbreakbench.metrics import npi_stringency
from outbreakbench.npis import DEFAULT_POLICY, NPIManager
from outbreakbench.scenarios import SCENARIOS, create_sim

FULL_LOCKDOWN = {
    "schools": "full",
    "workplaces": "full",
    "masks": True,
    "mass_testing": True,
    "contact_tracing": True,
    "gathering_limits": "ban_all",
    "stay_at_home": True,
}

N_SEEDS = 5



def run_fixed_policy(scenario_key, policy, seed):
    sim = create_sim(scenario_key, seed=seed)
    sim.initialize()
    mgr = NPIManager(sim)
    n_weeks = sim["n_days"] // 7

    for _ in range(n_weeks):
        mgr.apply(sim, policy)
        for _ in range(7):
            if sim.t < sim["n_days"]:
                sim.step()

    sim.finalize()
    r = sim.results
    return {
        "cum_deaths": int(r["cum_deaths"][-1]),
        "cum_infections": int(r["cum_infections"][-1]),
        "peak_severe": int(max(r["n_severe"])),
        "stringency": npi_stringency(policy) * n_weeks,
    }


def main():
    baselines = []

    for scenario_key in SCENARIOS:
        for seed in range(N_SEEDS):
            for label, policy in [
                ("no_intervention", DEFAULT_POLICY),
                ("full_lockdown", FULL_LOCKDOWN),
            ]:
                print(f"  {scenario_key} seed={seed} {label}...", end=" ", flush=True)
                result = run_fixed_policy(scenario_key, policy, seed)
                result["scenario"] = scenario_key
                result["seed"] = seed
                result["policy"] = label
                baselines.append(result)
                print(
                    f"deaths={result['cum_deaths']}, infections={result['cum_infections']}"
                )

    out_dir = "outputs/runs"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nSaved {len(baselines)} baselines to {out_dir}/baselines.json")


if __name__ == "__main__":
    main()
