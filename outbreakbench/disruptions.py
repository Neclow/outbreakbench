"""
Runtime shock handlers for the outbreak policy benchmark.

Applies mid-simulation disruptions (capacity changes, testing outages, border
events) defined in scenario shock lists.
"""


def apply_shocks(sim, week, shocks, scenario_pars, mgr=None):
    """Apply infrastructure shocks scheduled for this week.

    Returns a list of human-readable alert messages for any shocks that fired.
    """
    if not shocks:
        return []
    fired = []
    for shock in shocks:
        if shock["week"] != week:
            continue
        action = shock["action"]
        if action == "halve_icu":
            sim["n_beds_icu"] = max(1, sim["n_beds_icu"] // 2)
            fired.append("ICU capacity halved (staff outbreak)")
        elif action == "halve_hosp":
            sim["n_beds_hosp"] = max(1, sim["n_beds_hosp"] // 2)
            fired.append("Hospital capacity halved (staff outbreak)")
        elif action == "restore_icu":
            sim["n_beds_icu"] = scenario_pars.get("n_beds_icu", sim["n_beds_icu"] * 2)
            fired.append("ICU capacity restored")
        elif action == "restore_hosp":
            sim["n_beds_hosp"] = scenario_pars.get(
                "n_beds_hosp", sim["n_beds_hosp"] * 2
            )
            fired.append("Hospital capacity restored")
        elif action == "disable_testing":
            for intv in list(sim["interventions"]):
                if hasattr(intv, "symp_prob"):
                    sim["interventions"].remove(intv)
            if mgr is not None:
                mgr._test_prob = None
            fired.append("Testing infrastructure offline")
        elif action == "increase_imports":
            sim["n_imports"] = shock.get("value", 10)
            fired.append(
                f"Border reopening: imports increased to {sim['n_imports']}/day"
            )
        elif action == "restore_imports":
            sim["n_imports"] = scenario_pars.get("n_imports", 0)
            fired.append("Border imports restored to normal")
    return fired
