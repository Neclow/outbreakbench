"""
Scenario definitions for the outbreak policy benchmark.

Each scenario configures a Covasim sim that forces distinct NPI tradeoffs.
"""

import numpy as np
import covasim as cv


POP_SIZE = 50_000
N_DAYS = 180

BEDS_HOSP_PER_1K = 2.8
BEDS_ICU_PER_1K = 0.3


def _base_pars(**overrides):
    pars = dict(
        pop_size=POP_SIZE,
        pop_type="hybrid",
        n_days=N_DAYS,
        n_beds_hosp=int(POP_SIZE * BEDS_HOSP_PER_1K / 1000),
        n_beds_icu=int(POP_SIZE * BEDS_ICU_PER_1K / 1000),
    )
    pars.update(overrides)
    return pars


def _young_worker_prognoses():
    """W-curve mortality: elevated severity for 20-50 age group (1918 flu-like)."""
    prog = cv.get_prognoses()
    # age_cutoffs: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # indices:      0   1   2   3   4   5   6   7   8   9
    #
    # Boost 20-50 (indices 2,3,4) to ~1% IFR, comparable to 60-70 baseline.
    # Children and elderly stay at default COVID levels.
    prog["severe_probs"][2] = 0.15  # 20+: was 0.012
    prog["severe_probs"][3] = 0.18  # 30+: was 0.032
    prog["severe_probs"][4] = 0.15  # 40+: was 0.049
    prog["crit_probs"][2] = 0.25  # was 0.05
    prog["crit_probs"][3] = 0.30  # was 0.05
    prog["crit_probs"][4] = 0.25  # was 0.063
    prog["death_probs"][2] = 0.35  # was 0.278
    prog["death_probs"][3] = 0.40  # was 0.308
    prog["death_probs"][4] = 0.35  # was 0.454
    return prog


SCENARIOS = {
    "baseline": {
        "name": "Baseline COVID-like",
        "description": (
            "Moderate severity, age-skewed mortality (high in 60+), mixed economy. "
            "The standard reference scenario."
        ),
        "pars": _base_pars(),
    },
    "young_worker": {
        "name": "Young-worker epidemic",
        "description": (
            "W-curve mortality hitting working-age adults hardest (1918 flu-like). "
            "Workplace closures protect the most vulnerable but cost the most productivity."
        ),
        "pars": _base_pars(prognoses=_young_worker_prognoses()),
    },
    "scarce_icu": {
        "name": "Scarce ICU",
        "description": (
            "Same disease as baseline but hospital capacity at 50% and worse "
            "overflow outcomes. Forces earlier/harder NPI decisions."
        ),
        "pars": _base_pars(
            n_beds_hosp=int(POP_SIZE * BEDS_HOSP_PER_1K / 1000 / 2),
            n_beds_icu=int(POP_SIZE * BEDS_ICU_PER_1K / 1000 / 2),
            no_hosp_factor=3.0,
            no_icu_factor=5.0,
        ),
    },
    "ageing_pop": {
        "name": "Ageing population",
        "description": (
            "Japan-like age pyramid (34% aged 60+) with baseline disease. "
            "Tests whether LLMs adapt NPI strategy to demographic context."
        ),
        "pars": _base_pars(location="japan"),
    },
}


def create_sim(scenario_key, seed=0):
    """Create a ready-to-run sim for the given scenario and random seed."""
    scenario = SCENARIOS[scenario_key]
    pars = scenario["pars"].copy()
    pars["rand_seed"] = seed
    sim = cv.Sim(pars=pars, label=scenario["name"])
    return sim
