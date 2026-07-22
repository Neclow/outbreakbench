"""
Weekly surveillance report for the outbreak policy benchmark.

Generates structured text reports from a mid-run Covasim simulation,
designed as the sole input for an LLM policymaker.
"""

import numpy as np

AGE_BUCKETS = [
    ("0-19", 0, 20),
    ("20-39", 20, 40),
    ("40-59", 40, 60),
    ("60-79", 60, 80),
    ("80+", 80, 200),
]

WORKING_AGE = (20, 65)
SCHOOL_AGE = (5, 19)

_SETTING_LABELS = {"h": "Household", "s": "School", "w": "Workplace", "c": "Community"}
_LINKAGE_PROB = {"h": 0.85, "s": 0.60, "w": 0.55, "c": 0.10}
_LINKAGE_PROB_NO_TRACING = {"h": 0.50, "s": 0.05, "w": 0.05, "c": 0.02}
_TRACING_CAPACITY_PER_10K = 100

# Weekly NPI cost estimates scaled to population.
# Sources: Li & Spall (2022), Juneau et al. (2021), Molla et al. (2025).
# All figures in USD, per unit per day unless noted.
_COST_SCHOOL_PER_STUDENT_DAY = 125  # includes parental wage loss
_COST_WORK_PER_WORKER_DAY = 200  # GDP loss per absent worker-day
_COST_MASK_PER_PERSON_WEEK = 2.24  # distribution and enforcement
_COST_TEST = 36  # per test administered
_COST_TRACE_PER_CONTACT = 50  # per contact traced
_COST_GATHERING_BAN_LARGE_PER_CAPITA_WEEK = 5  # hospitality/event revenue loss
_COST_GATHERING_BAN_ALL_PER_CAPITA_WEEK = 15
_COST_SHIELDING_PER_ELDERLY_WEEK = 50  # care delivery, mental health, lost productivity


def _week_slice(day):
    return slice(max(0, day - 6), day + 1)


def _prev_week_slice(day):
    start = max(0, day - 13)
    end = max(0, day - 6)
    return slice(start, end)


def _sum_week(arr, day):
    return int(np.sum(arr[_week_slice(day)]))


def _sum_prev_week(arr, day):
    return int(np.sum(arr[_prev_week_slice(day)]))


def _pct_change(curr, prev):
    if prev == 0:
        return None
    return (curr - prev) / prev * 100


def _fmt_change(curr, prev):
    if prev is None:
        return "N/A (first report)"
    pct = _pct_change(curr, prev)
    if pct is None:
        return "N/A (no prior cases)"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% vs prior week"


def _age_masks(ages):
    masks = {}
    for label, lo, hi in AGE_BUCKETS:
        masks[label] = (ages >= lo) & (ages < hi)
    return masks


def _headline(sim, day):
    r = sim.results
    has_prev = day >= 14

    new_inf = _sum_week(r["new_infections"], day)
    new_deaths = _sum_week(r["new_deaths"], day)
    new_hosp = _sum_week(r["new_severe"], day)
    new_icu = _sum_week(r["new_critical"], day)

    prev_inf = _sum_prev_week(r["new_infections"], day) if has_prev else None
    prev_deaths = _sum_prev_week(r["new_deaths"], day) if has_prev else None
    prev_hosp = _sum_prev_week(r["new_severe"], day) if has_prev else None
    prev_icu = _sum_prev_week(r["new_critical"], day) if has_prev else None

    return dict(
        new_infections=new_inf,
        new_deaths=new_deaths,
        new_hosp=new_hosp,
        new_icu=new_icu,
        change_inf=_fmt_change(new_inf, prev_inf),
        change_deaths=_fmt_change(new_deaths, prev_deaths),
        change_hosp=_fmt_change(new_hosp, prev_hosp),
        change_icu=_fmt_change(new_icu, prev_icu),
        currently_infectious=int(r["n_infectious"][day]),
        cum_infections=int(np.sum(r["new_infections"][: day + 1])),
        cum_deaths=int(np.sum(r["new_deaths"][: day + 1])),
    )


def _epi_indicators(sim, day):
    p = sim.people
    r = sim.results

    # R_eff: mean_infectious_duration * new_infections / n_infectious, smoothed
    resolved = np.isfinite(p.date_recovered) & np.isfinite(p.date_infectious)
    if np.sum(resolved) > 10:
        durations = p.date_recovered[resolved] - p.date_infectious[resolved]
        mean_dur = float(np.mean(durations[durations > 0]))
    else:
        mean_dur = 10.0  # fallback early in epidemic

    window = min(3, day + 1)
    r_effs = []
    for d in range(day - window + 1, day + 1):
        if d < 0:
            continue
        n_inf = float(r["n_infectious"][d])
        new_inf = float(r["new_infections"][d])
        if n_inf > 0:
            r_effs.append(mean_dur * new_inf / n_inf)
    r_eff = float(np.mean(r_effs)) if r_effs else 0.0

    # Doubling time
    cum = np.cumsum(r["new_infections"][: day + 1])
    dt_window = 3
    if day >= dt_window and cum[day] > 0 and cum[day - dt_window] > 0:
        ratio = cum[day] / cum[day - dt_window]
        if ratio > 1:
            doubling_time = dt_window * np.log(2) / np.log(ratio)
            doubling_time = min(doubling_time, 999)
        else:
            doubling_time = None  # declining
    else:
        doubling_time = None

    # Prevalence and incidence
    pop_alive = int(sim["pop_size"]) - int(p.count("dead"))
    prevalence = p.count("infectious") / pop_alive * 100 if pop_alive > 0 else 0.0
    weekly_new = _sum_week(r["new_infections"], day)
    incidence = weekly_new / pop_alive * 100_000 if pop_alive > 0 else 0.0

    return dict(
        r_eff=r_eff,
        doubling_time=doubling_time,
        prevalence=prevalence,
        incidence=incidence,
    )


def _age_table(sim, day):
    p = sim.people
    ages = p.age
    masks = _age_masks(ages)
    ws = _week_slice(day)

    rows = []
    for label, mask in masks.items():
        new_inf_mask = mask & np.isfinite(p.date_exposed)
        new_inf_mask = new_inf_mask & (p.date_exposed >= ws.start)
        new_inf_mask = new_inf_mask & (p.date_exposed <= ws.stop - 1)

        rows.append(
            dict(
                label=label,
                pop=int(np.sum(mask & ~p.dead)),
                infectious=int(np.sum(mask & p.infectious)),
                hosp=int(np.sum(mask & p.severe)),
                icu=int(np.sum(mask & p.critical)),
                deaths=int(np.sum(mask & p.dead)),
                new_inf=int(np.sum(new_inf_mask)),
            )
        )
    return rows


def _hospital_capacity(sim, day):
    r = sim.results
    n_hosp = int(sim.people.count("severe"))
    n_icu = int(sim.people.count("critical"))
    cap_hosp = sim["n_beds_hosp"]
    cap_icu = sim["n_beds_icu"]

    # Count overflow days this week
    ws = _week_slice(day)
    hosp_overflow_days = 0
    icu_overflow_days = 0
    if cap_hosp is not None:
        hosp_overflow_days = int(np.sum(r["n_severe"][ws] > cap_hosp))
    if cap_icu is not None:
        icu_overflow_days = int(np.sum(r["n_critical"][ws] > cap_icu))

    return dict(
        n_hosp=n_hosp,
        n_icu=n_icu,
        cap_hosp=cap_hosp,
        cap_icu=cap_icu,
        hosp_util=n_hosp / cap_hosp * 100 if cap_hosp else None,
        icu_util=n_icu / cap_icu * 100 if cap_icu else None,
        hosp_overflow=cap_hosp is not None and n_hosp > cap_hosp,
        icu_overflow=cap_icu is not None and n_icu > cap_icu,
        hosp_overflow_days=hosp_overflow_days,
        icu_overflow_days=icu_overflow_days,
    )


def _economic_impact(sim, day):
    p = sim.people
    ages = p.age
    ill = p.symptomatic | p.severe | p.critical
    restricted = p.quarantined | p.isolated

    # Workers (20-64)
    w_mask = (ages >= WORKING_AGE[0]) & (ages < WORKING_AGE[1]) & ~p.dead
    w_total = int(np.sum(w_mask))
    w_ill = int(np.sum(w_mask & ill))
    w_restricted = int(np.sum(w_mask & restricted & ~ill))
    w_absent = w_ill + w_restricted

    # Students (5-18)
    s_mask = (ages >= SCHOOL_AGE[0]) & (ages <= SCHOOL_AGE[1]) & ~p.dead
    s_total = int(np.sum(s_mask))
    s_ill = int(np.sum(s_mask & ill))
    s_restricted = int(np.sum(s_mask & restricted & ~ill))
    s_absent = s_ill + s_restricted

    # Contact layer edge counts
    contacts = sim.people.contacts
    w_contacts = len(contacts["w"]) if "w" in contacts else None
    s_contacts = len(contacts["s"]) if "s" in contacts else None

    return dict(
        w_total=w_total,
        w_absent=w_absent,
        w_ill=w_ill,
        w_restricted=w_restricted,
        w_rate=w_absent / w_total * 100 if w_total > 0 else 0.0,
        s_total=s_total,
        s_absent=s_absent,
        s_ill=s_ill,
        s_restricted=s_restricted,
        s_rate=s_absent / s_total * 100 if s_total > 0 else 0.0,
        w_contacts=w_contacts,
        s_contacts=s_contacts,
    )


def _transmission_settings(sim, day, contact_tracing_active, rng):
    """Setting attribution from contact tracing data, degraded for realism.

    Only diagnosed cases are eligible. Of those, a fraction are
    epidemiologically linked to a transmission setting depending on the
    setting type (household >> community) and whether tracing capacity
    is overwhelmed.
    """
    p = sim.people
    ws = _week_slice(day)

    log_by_target = {}
    for e in p.infection_log:
        if e["layer"] != "seed_infection":
            log_by_target[e["target"]] = e

    diag_this_week = []
    for i in range(sim["pop_size"]):
        if np.isfinite(p.date_diagnosed[i]) and ws.start <= p.date_diagnosed[i] <= ws.stop - 1:
            diag_this_week.append(i)

    n_diagnosed = len(diag_this_week)
    if n_diagnosed == 0:
        return None

    capacity = _TRACING_CAPACITY_PER_10K * sim["pop_size"] / 10_000
    capacity_mult = min(1.0, capacity / n_diagnosed)

    linkage = _LINKAGE_PROB if contact_tracing_active else _LINKAGE_PROB_NO_TRACING

    setting_counts = {k: 0 for k in _SETTING_LABELS}
    n_unknown = 0

    for pid in diag_this_week:
        if pid not in log_by_target:
            n_unknown += 1
            continue
        layer = log_by_target[pid]["layer"]
        if rng.random() < linkage.get(layer, 0.1) * capacity_mult:
            setting_counts[layer] = setting_counts.get(layer, 0) + 1
        else:
            n_unknown += 1

    return dict(
        n_diagnosed=n_diagnosed,
        n_linked=sum(setting_counts.values()),
        n_unknown=n_unknown,
        settings=setting_counts,
    )


def _npi_costs(policy, sim):
    """Estimate weekly costs of active NPIs from policy and sim state."""
    if policy is None:
        return None
    p = sim.people
    pop_alive = int(sim["pop_size"]) - int(p.count("dead"))
    ages = p.age

    n_students = int(np.sum((ages >= SCHOOL_AGE[0]) & (ages <= SCHOOL_AGE[1]) & ~p.dead))
    n_workers = int(np.sum((ages >= WORKING_AGE[0]) & (ages < WORKING_AGE[1]) & ~p.dead))

    costs = {}

    closure_frac = {"open": 0.0, "partial": 0.5, "full": 1.0}
    if policy["schools"] != "open":
        f = closure_frac[policy["schools"]]
        costs["school_closure"] = _COST_SCHOOL_PER_STUDENT_DAY * n_students * f * 7
    if policy["workplaces"] != "open":
        f = closure_frac[policy["workplaces"]]
        costs["workplace_closure"] = _COST_WORK_PER_WORKER_DAY * n_workers * f * 7
    if policy["stay_at_home"]:
        absent = int(n_workers * 0.7)
        costs["stay_at_home"] = _COST_WORK_PER_WORKER_DAY * absent * 7
    if policy["masks"]:
        costs["masks"] = _COST_MASK_PER_PERSON_WEEK * pop_alive
    if policy["mass_testing"]:
        n_symptomatic = int(p.count("symptomatic"))
        daily_tests = n_symptomatic * 0.3 + (pop_alive - n_symptomatic) * 0.01
        costs["mass_testing"] = _COST_TEST * daily_tests * 7
    if policy["contact_tracing"]:
        n_diagnosed = int(np.sum(p.diagnosed & ~p.dead))
        avg_contacts = sum(len(sim.people.contacts[l]) for l in sim.people.layer_keys())
        avg_contacts_per_person = avg_contacts / pop_alive if pop_alive else 0
        weekly_traces = n_diagnosed * avg_contacts_per_person * 0.5
        costs["contact_tracing"] = _COST_TRACE_PER_CONTACT * weekly_traces
    if policy["gathering_limits"] == "ban_large":
        costs["gathering_limits"] = _COST_GATHERING_BAN_LARGE_PER_CAPITA_WEEK * pop_alive
    elif policy["gathering_limits"] == "ban_all":
        costs["gathering_limits"] = _COST_GATHERING_BAN_ALL_PER_CAPITA_WEEK * pop_alive
    if policy.get("shielding_elderly", False):
        n_elderly = int(np.sum((ages >= 60) & ~p.dead))
        costs["shielding_elderly"] = _COST_SHIELDING_PER_ELDERLY_WEEK * n_elderly

    return costs


def _format_report(day, headline, age_rows, hospital, epi, economic, active_npis, npi_costs=None, settings=None):
    week_num = day // 7
    lines = []
    sep = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append("WEEKLY EPIDEMIC SURVEILLANCE REPORT")
    lines.append(f"Simulation Day {day} (Week {week_num})")
    lines.append(sep)

    # Active interventions
    lines.append("")
    lines.append("ACTIVE INTERVENTIONS")
    if active_npis:
        for k, v in active_npis.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  No interventions active")

    # Headline
    lines.append("")
    lines.append(dash)
    lines.append("HEADLINE SUMMARY")
    lines.append(dash)
    h = headline
    lines.append(
        f"  New infections this week:    {h['new_infections']:>7,}  ({h['change_inf']})"
    )
    lines.append(
        f"  New deaths this week:        {h['new_deaths']:>7,}  ({h['change_deaths']})"
    )
    lines.append(
        f"  New hospitalisations:        {h['new_hosp']:>7,}  ({h['change_hosp']})"
    )
    lines.append(
        f"  New ICU admissions:          {h['new_icu']:>7,}  ({h['change_icu']})"
    )
    lines.append(f"  Currently infectious:        {h['currently_infectious']:>7,}")
    lines.append(f"  Cumulative infections:       {h['cum_infections']:>7,}")
    lines.append(f"  Cumulative deaths:           {h['cum_deaths']:>7,}")

    # Epi indicators
    lines.append("")
    lines.append(dash)
    lines.append("EPIDEMIOLOGICAL INDICATORS")
    lines.append(dash)
    lines.append(f"  Estimated R_eff (3-day avg):  {epi['r_eff']:>6.2f}")
    if epi["doubling_time"] is not None:
        lines.append(f"  Doubling time:              {epi['doubling_time']:>6.1f} days")
    else:
        lines.append("  Doubling time:               N/A (declining)")
    lines.append(f"  Prevalence (% population):    {epi['prevalence']:>5.1f}%")
    lines.append(f"  Weekly incidence (per 100k):{epi['incidence']:>8.1f}")

    # Age table
    lines.append("")
    lines.append(dash)
    lines.append("AGE-STRATIFIED BREAKDOWN")
    lines.append(dash)
    hdr = (
        f"  {'Age group':<10}| {'Pop.':>6} | {'Inf.':>5} | {'Hosp.':>5} "
        f"| {'ICU':>4} | {'Deaths':>6} | {'New inf.':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * 66)
    totals = dict(pop=0, infectious=0, hosp=0, icu=0, deaths=0, new_inf=0)
    for row in age_rows:
        lines.append(
            f"  {row['label']:<10}|{row['pop']:>7,} |{row['infectious']:>6,} "
            f"|{row['hosp']:>6,} |{row['icu']:>5,} |{row['deaths']:>7,} "
            f"|{row['new_inf']:>9,}"
        )
        for k in totals:
            totals[k] += row[k]
    lines.append("  " + "-" * 66)
    lines.append(
        f"  {'TOTAL':<10}|{totals['pop']:>7,} |{totals['infectious']:>6,} "
        f"|{totals['hosp']:>6,} |{totals['icu']:>5,} |{totals['deaths']:>7,} "
        f"|{totals['new_inf']:>9,}"
    )

    # Hospital capacity
    lines.append("")
    lines.append(dash)
    lines.append("HOSPITAL CAPACITY")
    lines.append(dash)
    hc = hospital
    if hc["cap_hosp"] is not None:
        overflow_tag = "  ** OVERFLOW **" if hc["hosp_overflow"] else ""
        lines.append(
            f"  Hospital beds:  {hc['n_hosp']:>5} / {hc['cap_hosp']:>5} occupied"
            f"  ({hc['hosp_util']:.1f}% utilisation){overflow_tag}"
        )
    else:
        lines.append(
            f"  Hospital beds:  {hc['n_hosp']:>5} occupied  (no capacity limit)"
        )
    if hc["cap_icu"] is not None:
        overflow_tag = "  ** OVERFLOW **" if hc["icu_overflow"] else ""
        lines.append(
            f"  ICU beds:       {hc['n_icu']:>5} / {hc['cap_icu']:>5} occupied"
            f"  ({hc['icu_util']:.1f}% utilisation){overflow_tag}"
        )
    else:
        lines.append(
            f"  ICU beds:       {hc['n_icu']:>5} occupied  (no capacity limit)"
        )
    if hc["hosp_overflow_days"] > 0:
        lines.append(
            f"  Hospital overflow for {hc['hosp_overflow_days']} day(s) this week."
        )
    if hc["icu_overflow_days"] > 0:
        lines.append(f"  ICU overflow for {hc['icu_overflow_days']} day(s) this week.")

    # Economic impact
    lines.append("")
    lines.append(dash)
    lines.append("ECONOMIC IMPACT (snapshot)")
    lines.append(dash)
    ec = economic
    lines.append(
        f"  Workers absent (ages 20-64):  {ec['w_absent']:>6,} / {ec['w_total']:>6,}"
        f"  ({ec['w_rate']:.1f}%)"
    )
    lines.append(f"    - symptomatic/ill:          {ec['w_ill']:>6,}")
    lines.append(f"    - quarantined/isolated:     {ec['w_restricted']:>6,}")
    lines.append(
        f"  Students absent (ages 5-18):  {ec['s_absent']:>6,} / {ec['s_total']:>6,}"
        f"  ({ec['s_rate']:.1f}%)"
    )
    lines.append(f"    - symptomatic/ill:          {ec['s_ill']:>6,}")
    lines.append(f"    - quarantined/isolated:     {ec['s_restricted']:>6,}")
    if ec["w_contacts"] is not None:
        lines.append(f"  Workplace contacts active:  {ec['w_contacts']:>8,}")
    if ec["s_contacts"] is not None:
        lines.append(f"  School contacts active:     {ec['s_contacts']:>8,}")

    # NPI costs
    if npi_costs:
        lines.append("")
        lines.append(dash)
        lines.append("ESTIMATED WEEKLY NPI COSTS (USD)")
        lines.append(dash)
        labels = {
            "school_closure": "School closure",
            "workplace_closure": "Workplace closure",
            "stay_at_home": "Stay-at-home order",
            "masks": "Mask mandate",
            "mass_testing": "Testing programme",
            "contact_tracing": "Contact tracing",
            "gathering_limits": "Gathering restrictions",
            "shielding_elderly": "Elderly shielding",
        }
        total = 0
        for key, label in labels.items():
            if key in npi_costs:
                cost = npi_costs[key]
                total += cost
                lines.append(f"  {label + ':':<28s}${cost:>12,.0f}")
        lines.append(f"  {'TOTAL:':<28s}${total:>12,.0f}")

    if settings is not None:
        lines.append("")
        lines.append(dash)
        lines.append("TRANSMISSION SETTING ATTRIBUTION (contact tracing data)")
        lines.append(dash)
        n_d = settings["n_diagnosed"]
        n_l = settings["n_linked"]
        n_u = settings["n_unknown"]
        pct = n_l / n_d * 100 if n_d > 0 else 0
        lines.append(
            f"  Of {n_d:,} cases diagnosed this week, {n_l:,} ({pct:.0f}%) had an"
        )
        lines.append(
            f"  identified transmission setting. {n_u:,} could not be linked."
        )
        lines.append("")
        lines.append("  Identified settings:")
        for lkey in ("h", "s", "w", "c"):
            count = settings["settings"].get(lkey, 0)
            lpct = count / n_l * 100 if n_l > 0 else 0
            lines.append(
                f"    {_SETTING_LABELS[lkey] + ':':<14s}{count:>5,}  ({lpct:4.0f}% of linked)"
            )
        lines.append("")
        lines.append(
            "  NOTE: Community transmission is likely underrepresented."
        )
        lines.append(
            "  Unlinked cases may include untraced community transmission."
        )

    lines.append(sep)

    return "\n".join(lines)


def generate_report(sim, day, active_npis=None, setting_attribution_rng=None):
    """Generate a weekly surveillance report from a mid-run Covasim sim.

    Parameters
    ----------
    setting_attribution_rng : numpy.random.Generator or None
        If provided, include degraded transmission setting attribution
        derived from the sim's infection log and diagnosis state.
    """
    headline = _headline(sim, day)
    epi = _epi_indicators(sim, day)
    age_rows = _age_table(sim, day)
    hospital = _hospital_capacity(sim, day)
    economic = _economic_impact(sim, day)
    costs = _npi_costs(active_npis, sim) if active_npis else None
    settings = None
    if setting_attribution_rng is not None:
        ct_active = active_npis is not None and active_npis.get("contact_tracing", False)
        settings = _transmission_settings(sim, day, ct_active, setting_attribution_rng)
    return _format_report(
        day, headline, age_rows, hospital, epi, economic, active_npis,
        npi_costs=costs, settings=settings,
    )
