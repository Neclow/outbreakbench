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


def _format_report(day, headline, age_rows, hospital, epi, economic, active_npis):
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
    lines.append(
        f"  Currently infectious:        {h['currently_infectious']:>7,}"
    )
    lines.append(
        f"  Cumulative infections:       {h['cum_infections']:>7,}"
    )
    lines.append(
        f"  Cumulative deaths:           {h['cum_deaths']:>7,}"
    )

    # Epi indicators
    lines.append("")
    lines.append(dash)
    lines.append("EPIDEMIOLOGICAL INDICATORS")
    lines.append(dash)
    lines.append(f"  Estimated R_eff (3-day avg):  {epi['r_eff']:>6.2f}")
    if epi["doubling_time"] is not None:
        lines.append(
            f"  Doubling time:              {epi['doubling_time']:>6.1f} days"
        )
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
        lines.append(f"  Hospital beds:  {hc['n_hosp']:>5} occupied  (no capacity limit)")
    if hc["cap_icu"] is not None:
        overflow_tag = "  ** OVERFLOW **" if hc["icu_overflow"] else ""
        lines.append(
            f"  ICU beds:       {hc['n_icu']:>5} / {hc['cap_icu']:>5} occupied"
            f"  ({hc['icu_util']:.1f}% utilisation){overflow_tag}"
        )
    else:
        lines.append(f"  ICU beds:       {hc['n_icu']:>5} occupied  (no capacity limit)")
    if hc["hosp_overflow_days"] > 0:
        lines.append(
            f"  Hospital overflow for {hc['hosp_overflow_days']} day(s) this week."
        )
    if hc["icu_overflow_days"] > 0:
        lines.append(
            f"  ICU overflow for {hc['icu_overflow_days']} day(s) this week."
        )

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
    lines.append(sep)

    return "\n".join(lines)


def generate_report(sim, day, active_npis=None):
    """Generate a weekly surveillance report from a mid-run Covasim sim."""
    headline = _headline(sim, day)
    epi = _epi_indicators(sim, day)
    age_rows = _age_table(sim, day)
    hospital = _hospital_capacity(sim, day)
    economic = _economic_impact(sim, day)
    return _format_report(day, headline, age_rows, hospital, epi, economic, active_npis)
