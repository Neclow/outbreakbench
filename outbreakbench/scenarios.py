"""
Scenario definitions for the outbreak policy benchmark.

Each scenario configures a Covasim sim that forces distinct NPI tradeoffs.
"""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="could not install some fonts")
    import covasim as cv

import numpy as np

from ._config import (
    BEDS_HOSP_PER_1K,
    BEDS_ICU_PER_1K,
    BURN_IN_WEEKS,
    DECISION_INTERVAL,
    N_DAYS,
    POP_SIZE,
)


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


def _mild_flu_prognoses():
    """H1N1 2009-like: high transmission, very low severity."""
    prog = cv.get_prognoses()
    prog["severe_probs"] *= 0.1
    prog["crit_probs"] *= 0.2
    prog["death_probs"] *= 0.5
    return prog


def _ebola_like_prognoses():
    """Ebola-like: ~50% CFR, flat across ages."""
    prog = cv.get_prognoses()
    n = len(prog["age_cutoffs"])
    prog["symp_probs"][:] = 0.95
    prog["severe_probs"][:] = 0.60
    prog["crit_probs"][:] = 0.70
    prog["death_probs"][:] = np.full(n, 0.75)
    return prog


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
    "mild_flu": {
        "name": "Mild flu",
        "description": (
            "H1N1 2009-like: fast-spreading but low severity. "
            "Tests whether LLMs over-protect when disease doesn't warrant aggressive NPIs."
        ),
        "pars": _base_pars(
            prognoses=_mild_flu_prognoses(),
            dur=dict(
                exp2inf=dict(dist="lognormal_int", par1=2.0, par2=1.0),
                inf2sym=dict(dist="lognormal_int", par1=0.5, par2=0.5),
                sym2sev=dict(dist="lognormal_int", par1=5.0, par2=3.0),
                sev2crit=dict(dist="lognormal_int", par1=1.5, par2=2.0),
                asym2rec=dict(dist="lognormal_int", par1=6.0, par2=2.0),
                mild2rec=dict(dist="lognormal_int", par1=6.0, par2=2.0),
                sev2rec=dict(dist="lognormal_int", par1=12.0, par2=4.0),
                crit2rec=dict(dist="lognormal_int", par1=12.0, par2=4.0),
                crit2die=dict(dist="lognormal_int", par1=8.0, par2=3.0),
            ),
        ),
    },
    "ebola_like": {
        "name": "Ebola-like",
        "description": (
            "Fast & lethal: ~50% CFR flat across ages, lower R0, "
            "household-heavy transmission. Tests rapid decisive action."
        ),
        "pars": _base_pars(
            beta=0.012,
            beta_layer=dict(h=5.0, s=0.3, w=0.3, c=0.1),
            prognoses=_ebola_like_prognoses(),
            dur=dict(
                exp2inf=dict(dist="lognormal_int", par1=8.0, par2=3.0),
                inf2sym=dict(dist="lognormal_int", par1=1.0, par2=0.5),
                sym2sev=dict(dist="lognormal_int", par1=3.0, par2=1.5),
                sev2crit=dict(dist="lognormal_int", par1=2.0, par2=1.0),
                asym2rec=dict(dist="lognormal_int", par1=10.0, par2=3.0),
                mild2rec=dict(dist="lognormal_int", par1=10.0, par2=3.0),
                sev2rec=dict(dist="lognormal_int", par1=14.0, par2=5.0),
                crit2rec=dict(dist="lognormal_int", par1=14.0, par2=5.0),
                crit2die=dict(dist="lognormal_int", par1=5.0, par2=2.0),
            ),
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
    "high_r0": {
        "name": "High-R0 variant",
        "description": (
            "Delta-like transmissibility (R0 ~5). Masks and testing alone "
            "cannot keep R_eff < 1 — forces workplace/school closures."
        ),
        "pars": _base_pars(beta=0.035),
    },
    "import_pressure": {
        "name": "Import pressure",
        "description": (
            "Baseline disease with 5 imported cases per day. Even if local "
            "transmission is suppressed, new chains keep starting. "
            "Tests sustained NPI management vs suppress-and-coast."
        ),
        "pars": _base_pars(n_imports=5),
    },
    "infrastructure_shock": {
        "name": "Infrastructure shock",
        "description": (
            "Baseline disease with mid-sim disruptions: hospital capacity "
            "halved at week 20 (staff outbreak), restored at week 30. "
            "Tests adaptive decision-making under changing constraints."
        ),
        "pars": _base_pars(),
        "shocks": [
            {"week": 20, "action": "halve_hosp"},
            {"week": 20, "action": "halve_icu"},
            {"week": 30, "action": "restore_hosp"},
            {"week": 30, "action": "restore_icu"},
        ],
    },
}


def create_sim(scenario_key, seed=0):
    """Create a ready-to-run sim for the given scenario and random seed."""
    scenario = SCENARIOS[scenario_key]
    pars = scenario["pars"].copy()
    pars["rand_seed"] = seed
    sim = cv.Sim(pars=pars, label=scenario["name"])
    return sim


