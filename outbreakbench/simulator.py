"""
Decision loop runner for the outbreak policy benchmark.

Runs a Covasim sim week-by-week, feeding surveillance reports to an LLM
and applying its NPI decisions each cycle.
"""

import json
import time

import numpy as np

from .llm import build_system_prompt, build_user_message, parse_response
from .npis import DEFAULT_POLICY, NPIManager
from .report import generate_report
from ._config import BURN_IN_WEEKS, DECISION_INTERVAL
from .scenarios import SCENARIOS, create_sim
from .disruptions import apply_shocks


def run_benchmark(
    scenario_key,
    call_llm,
    framing="neutral",
    seed=0,
    setting_attribution=False,
    age_targeting=False,
):
    """Run a single benchmark: one scenario, one LLM, one framing, one seed.

    Parameters
    ----------
    scenario_key : str
        Key from SCENARIOS dict.
    call_llm : callable
        Function with signature call_llm(system_prompt, messages) -> str.
        `messages` is a list of {"role": ..., "content": ...} dicts.
    framing : str
        One of "neutral", "public_health", "economic".
    seed : int
        Random seed for the simulation.
    setting_attribution : bool
        If True, include degraded transmission setting attribution in
        surveillance reports (requires contact tracing data from the ABM).
    age_targeting : bool
        If True, enable age-targeted NPIs (elderly shielding).

    Returns
    -------
    dict with keys: scenario, framing, seed, decisions, final_results, elapsed.
    """
    scenario = SCENARIOS[scenario_key]
    sim = create_sim(scenario_key, seed=seed)
    sim.initialize()

    pop_size = sim["pop_size"]
    n_days = sim["n_days"]
    n_weeks = n_days // 7

    system_prompt = build_system_prompt(framing, pop_size=pop_size, n_days=n_days,
                                       age_targeting=age_targeting)
    mgr = NPIManager(sim, age_targeting=age_targeting)

    shocks = scenario.get("shocks", [])
    sa_rng = np.random.default_rng(seed + 1) if setting_attribution else None

    n_decisions_expected = (
        n_weeks - BURN_IN_WEEKS + DECISION_INTERVAL - 1
    ) // DECISION_INTERVAL

    from .npis import DEFAULT_POLICY_AGE_TARGETED
    decisions = []
    messages = []
    burn_in_reports = []
    pending_alerts = []
    base_policy = DEFAULT_POLICY_AGE_TARGETED if age_targeting else DEFAULT_POLICY
    current_policy = dict(base_policy)
    prev_notes = ""

    t0 = time.time()

    for week in range(n_weeks):
        shock_msgs = apply_shocks(sim, week + 1, shocks, scenario["pars"], mgr=mgr)
        if shock_msgs:
            pending_alerts.extend(shock_msgs)

        mgr.apply(sim, current_policy)

        for _ in range(7):
            if sim.t < n_days:
                sim.step()

        day = (week + 1) * 7 - 1
        if day >= n_days:
            break

        # Burn-in: accumulate reports but don't ask the LLM
        if week < BURN_IN_WEEKS:
            report = generate_report(sim, day, active_npis=current_policy,
                                     setting_attribution_rng=sa_rng)
            if pending_alerts:
                report += "\n\n** ALERT **\n" + "\n".join(f"  - {m}" for m in pending_alerts)
                pending_alerts = []
            burn_in_reports.append((week + 1, report))
            continue

        # Only ask the LLM every DECISION_INTERVAL weeks after burn-in
        weeks_since_burn_in = week - BURN_IN_WEEKS
        if weeks_since_burn_in % DECISION_INTERVAL != 0:
            continue

        report = generate_report(sim, day, active_npis=current_policy,
                                 setting_attribution_rng=sa_rng)
        if pending_alerts:
            report += "\n\n** ALERT **\n" + "\n".join(f"  - {m}" for m in pending_alerts)
            pending_alerts = []

        # First decision: include burn-in reports as context
        if not messages and burn_in_reports:
            for bi_week, bi_report in burn_in_reports:
                bi_msg = build_user_message(bi_report, week_number=bi_week)
                messages.append({"role": "user", "content": bi_msg})
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Noted. No interventions are in place yet — I am observing the situation.",
                    }
                )
            burn_in_reports = []

        user_msg = build_user_message(
            report, week_number=week + 1, prev_notes=prev_notes
        )
        messages.append({"role": "user", "content": user_msg})

        # Sliding window: keep burn-in context + last 5 decision rounds
        n_burn_in_msgs = BURN_IN_WEEKS * 2
        max_decision_msgs = 5 * 2
        decision_msgs = messages[n_burn_in_msgs:]
        if len(decision_msgs) > max_decision_msgs + 1:
            messages = messages[:n_burn_in_msgs] + decision_msgs[-(max_decision_msgs + 1):]

        response_text = call_llm(system_prompt, messages)

        try:
            policy, justification, notes = parse_response(response_text,
                                                         age_targeting=age_targeting)
        except (ValueError, json.JSONDecodeError, AssertionError) as e:
            print(
                f"\n  PARSE FAILURE at week {week + 1}: {e}"
                f"\n  Raw response:\n{response_text[:500]}"
            )
            break

        messages.append({"role": "assistant", "content": response_text})

        decisions.append(
            {
                "week": week + 1,
                "day": day,
                "policy": policy,
                "justification": justification,
                "notes": notes,
                "shocks": shock_msgs if shock_msgs else None,
                "raw_response": response_text,
            }
        )

        dt = time.time() - t0
        print(
            f"\r  decision {len(decisions)}/{n_decisions_expected} "
            f"(week {week + 1}) {dt:.0f}s",
            end="",
            flush=True,
        )

        current_policy = policy
        prev_notes = notes

    if decisions:
        print()
    sim.finalize()
    elapsed = time.time() - t0
    r = sim.results

    return {
        "scenario": scenario_key,
        "framing": framing,
        "seed": seed,
        "decisions": decisions,
        "final_results": {
            "cum_infections": int(r["cum_infections"][-1]),
            "cum_deaths": int(r["cum_deaths"][-1]),
            "peak_severe": int(max(r["n_severe"])),
            "peak_critical": int(max(r["n_critical"])),
        },
        "time_series": {
            "n_infectious": [int(x) for x in r["n_infectious"]],
            "cum_deaths": [int(x) for x in r["cum_deaths"]],
            "n_quarantined": [int(x) for x in r["n_quarantined"]],
        },
        "elapsed": elapsed,
    }
