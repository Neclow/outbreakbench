"""
Decision loop runner for the outbreak policy benchmark.

Runs a Covasim sim week-by-week, feeding surveillance reports to an LLM
and applying its NPI decisions each cycle.
"""

import json
import time

from outbreakbench.llm import build_system_prompt, build_user_message, parse_response
from outbreakbench.npis import DEFAULT_POLICY, NPIManager
from outbreakbench.scenarios import SCENARIOS, create_sim
from outbreakbench.surveillance import generate_report


def run_benchmark(
    scenario_key,
    call_llm,
    framing="neutral",
    seed=0,
    max_retries=1,
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
    max_retries : int
        Retries on parse failure before defaulting to previous policy.

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

    system_prompt = build_system_prompt(framing, pop_size=pop_size, n_days=n_days)
    mgr = NPIManager(sim)

    decisions = []
    messages = []
    current_policy = dict(DEFAULT_POLICY)

    t0 = time.time()

    for week in range(n_weeks):
        mgr.apply(sim, current_policy)

        for _ in range(7):
            if sim.t < n_days:
                sim.step()

        day = (week + 1) * 7 - 1
        if day >= n_days:
            break

        report = generate_report(sim, day, active_npis=current_policy)
        user_msg = build_user_message(report, week_number=week + 1)
        messages.append({"role": "user", "content": user_msg})

        policy = None
        justification = ""
        parse_error = None

        for attempt in range(1 + max_retries):
            response_text = call_llm(system_prompt, messages)

            try:
                policy, justification = parse_response(response_text)
                messages.append({"role": "assistant", "content": response_text})
                break
            except (ValueError, json.JSONDecodeError, AssertionError) as e:
                parse_error = str(e)
                if attempt < max_retries:
                    retry_msg = (
                        f"Your response could not be parsed: {parse_error}\n"
                        f"Please respond with a valid JSON block containing all "
                        f"7 policy fields and a justification."
                    )
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": retry_msg})

        if policy is None:
            policy = dict(current_policy)
            justification = f"[PARSE FAILURE: {parse_error}] Defaulting to previous policy."
            messages.append({"role": "assistant", "content": justification})

        decisions.append({
            "week": week + 1,
            "day": day,
            "policy": policy,
            "justification": justification,
        })

        current_policy = policy

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
        "elapsed": elapsed,
    }
