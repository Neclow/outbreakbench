# Project: Revealed Policy Preferences of LLMs Under Epidemic Uncertainty

## Goal

Build a simulation framework that places LLMs in the role of epidemic policymaker, presents them with weekly surveillance reports from a realistic agent-based model (Covasim), collects their NPI decisions, and analyses the *revealed preferences* implied by their choice patterns — not just whether they perform well, but what tradeoffs they implicitly prioritize (lives vs. livelihoods, elderly vs. young, health equity vs. efficiency).

## Key References

- **Covasim** (Kerr et al., 2021) — the ABM we're using: https://arxiv.org/abs/2104.04580
  - Docs: https://docs.covasim.org / GitHub: https://github.com/starsimhub/covasim
  - Provides: age-stratified agents, 5 contact layers (household/school/workplace/community/LTCF), built-in NPIs (clip_edges, change_beta, testing, tracing, vaccination), per-day results, custom interventions with full sim access
- **Aoki & Ghaffarzadegan (2026)** — LLM as policymaker in SEIR loop (our direct predecessor, we improve on this): https://arxiv.org/abs/2601.04245
- **Epi-LLM** (Ferencz et al., 2026) — LLM agents as individual citizens in epidemic ABM (complementary, different level of analysis): https://arxiv.org/abs/2606.02867
- **Yamin et al. (2026)** — revealed preference estimation for LLM alignment via discrete choice models (our methodological inspiration): https://arxiv.org/abs/2605.08556
- **Slama et al. (2026)** — do LLM preferences predict downstream behavior: https://arxiv.org/abs/2602.18971
- **Fed Reserve FEDS 2026-006** — revealed preferences in LLMs via economic games (shows LLMs have measurable, malleable preferences)
- **Shi et al. (2026)** — multi-agent LLM coordinated pandemic control: https://arxiv.org/abs/2601.09264

## Pipeline Overview

```
Step 1: Scenario Design
    Define 10-15 Covasim parameter sets that force distinct policy tradeoffs.
    Each scenario varies disease severity, demographics, hospital capacity,
    and economic structure to create conditions where NPI choices reveal
    implicit priorities.

Step 2: Decision Loop
    Every 7 simulated days:
    → Extract structured surveillance report from sim.people
    → Present to LLM with NPI action menu
    → Collect (action, free-text justification)
    → Apply NPI to Covasim, simulate 7 more days
    → Repeat for 180 days (~26 decision points)

Step 3: Outcome Computation
    For each trajectory, compute:
    - Health metrics: deaths by age group, peak hospitalisation, ICU overflow days
    - Productivity metrics: work-days lost (working-age quarantined/isolated/sick
      + workplace layer clipped), school-days lost
    - Equity metrics: age-stratified mortality ratios, Gini of infection burden
    - Counterfactuals: at each decision point, simulate alternative NPIs

Step 4: Revealed Preference Estimation
    Fit discrete choice model (multinomial logit) to decision trajectories.
    Features = state variables (cases by age, hospital occupancy, economic disruption).
    Choices = NPI actions. Recovered coefficients = implied utility weights.
    Compare across LLMs, prompt framings, scenarios.

Step 5: Stated vs. Revealed Analysis
    Code free-text justifications for stated priorities.
    Compare to recovered utility weights from Step 4.
```

## NPI Action Space (fixed menu, maps to Covasim)

```
1. School closure:     full / partial / open     → clip_edges(layer='s', change=0.0/0.5/1.0)
2. Workplace closure:  full / partial / open     → clip_edges(layer='w', change=0.0/0.5/1.0)
3. Mask mandate:       yes / no                  → change_beta(layers=['s','w','c'], changes=0.7/1.0)
4. Mass testing:       yes / no                  → test_prob(symp_prob=0.3/0.05)
5. Contact tracing:    yes / no                  → contact_tracing(trace_probs=dict/None)
6. Gathering limits:   ban large / ban all / none → change_beta(layer='c', changes=0.5/0.2/1.0)
7. Stay-at-home:       yes / no                  → clip_edges(layers=['w','c'], change=0.3/1.0)
```

The LLM selects a bundle each week. Each option maps deterministically to Covasim parameters.

## Design Considerations

- **Stochasticity**: Covasim is stochastic. Run each (scenario, LLM, prompt) combo with 10-20 random seeds to get stable estimates.
- **Report format**: Structured text (not images). Same template for all LLMs. Modelled on WHO/UKHSA weekly surveillance reports: headline metrics, age-stratified table, hospital capacity, economic state.
- **Prompt framing**: Test at least 2-3 framings per LLM (neutral policymaker, public health official, economic advisor) to measure preference malleability.
- **Action parsing**: LLM must return structured output (JSON) from the fixed menu. No free-text NPI selection. Parse failures = repeat prompt once, then default to no-change.
- **Baselines**: No intervention, maximum lockdown, threshold heuristic (close schools when cases > X/100k, close workplaces when ICU > 80%), and Pareto-optimal (brute-force over action space per decision point).
- **Cost model**: Assign per-unit costs from pandemic economics literature: ~$200/work-day lost, ~$50/school-day lost, ~$2000/hospitalisation-day, statistical value of life for deaths. Exact values are secondary to the tradeoff structure they create.

## Dependency Management

This project uses pixi. Add deps with:

```bash
pixi add numpy pandas matplotlib seaborn scipy
pixi add --pypi covasim
pixi add --pypi anthropic
pixi add --pypi scikit-learn statsmodels
```

## Current Task: Implement Step 1 (Scenario Design)

### What to build

A Python module `scenarios.py` that defines Covasim scenario parameter sets.

Each scenario should be a function or dict that returns a configured `cv.Sim` object. Start with 3-4 scenarios that demonstrate forced tradeoffs:

1. **Baseline COVID-like**: Moderate severity, age-skewed mortality (high in 60+), mixed economy. The "standard" scenario.
2. **Young-worker epidemic**: A pathogen that hits working-age adults hardest (e.g. 1918 flu-like W-curve mortality). Forces tradeoff: workplace closures protect the most vulnerable but cost the most productivity.
3. **Scarce ICU**: Same disease as baseline but hospital capacity at 50%. Forces earlier/harder NPI decisions.
4. **Ageing population**: Same disease but elderly-skewed demographics (e.g. Japan-like age pyramid vs. Nigeria-like). Tests whether LLMs adapt to demographic context.

### What each scenario needs

- A descriptive name and a short docstring explaining what tradeoff it forces
- Covasim `cv.Sim` parameters: population size (start with 50k for fast iteration), number of days (180), disease parameters (beta, severity by age), population demographics, hospital/ICU bed counts
- A `create_sim(seed)` function that returns a ready-to-run sim with the given random seed
- Validation: each scenario should run standalone (no LLM) and produce plausible epidemic curves

### Verify

- Each scenario runs in under 30 seconds
- Epidemic curves look qualitatively different across scenarios
- Age-stratified outcomes differ meaningfully (e.g. baseline kills mostly elderly, young-worker scenario kills mostly 20-50)
- Save basic plots to `outputs/` for visual inspection

### Do NOT

- Build the LLM interface yet (Step 2)
- Build the reporting system yet
- Add more than 4 scenarios initially
- Over-engineer the parameter space — start concrete, we'll vary later
