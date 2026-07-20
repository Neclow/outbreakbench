# Benchmark Design

## Scenarios

Each scenario configures a distinct epidemic in a population of 50,000 people over 350 days (50 weeks). The scenarios are designed so that different NPI strategies lead to meaningfully different outcomes — there is no single "correct" policy.

### 1. Baseline COVID-like (`baseline`)

A moderate-severity respiratory pathogen with age-skewed mortality (high in 60+). Standard hospital capacity. This is the reference scenario — most NPIs produce visible effects, and the LLM must balance protecting the elderly against general disruption.

- Default Covasim COVID-19 parameters
- 140 hospital beds, 15 ICU beds
- Uncontrolled: ~700 deaths, ~53,000 infections

### 2. Young-worker epidemic (`young_worker`)

A 1918 flu-like "W-curve" pathogen that hits working-age adults (20–50) hardest. Workplace closures directly protect the most vulnerable group — but also cause the greatest economic damage. Forces a sharp lives-vs-livelihoods tradeoff.

- Boosted severity for ages 20–50 (severe_probs ~0.15–0.18, crit_probs ~0.25–0.30)
- Children and elderly at default COVID severity
- Key tension: closing workplaces saves the most lives but costs the most productivity

### 3. Scarce ICU (`scarce_icu`)

Same disease as baseline, but hospital capacity halved and overflow outcomes worsened (3× mortality without hospital bed, 5× without ICU). Forces earlier and harder NPI decisions because the system saturates sooner.

- 70 hospital beds, 7 ICU beds (half of baseline)
- `no_hosp_factor=3.0`, `no_icu_factor=5.0`
- Tests whether the LLM reads capacity signals and acts pre-emptively

### 4. Mild flu (`mild_flu`)

An H1N1 2009-like pathogen: fast-spreading but very low severity. Deaths are rare even without intervention. Tests whether the LLM over-protects when the disease doesn't warrant aggressive NPIs — an economically costly type-I error.

- Severe/critical/death probabilities reduced to 10–50% of baseline
- Shorter disease durations (faster recovery)
- Uncontrolled: very few deaths despite high infection count

### 5. Ebola-like (`ebola_like`)

A fast and lethal pathogen: ~50% case fatality rate, flat across all ages, with household-heavy transmission. Lower R0 than COVID but devastating outcomes for those infected. Tests whether the LLM responds decisively to high-severity, low-frequency signals.

- `beta=0.012` with household-dominant contact weights (h=5.0, others ≤0.3)
- Uniform severity: 95% symptomatic, 60% severe, 70% critical, 75% fatal
- Longer incubation (8 days) and illness duration
- Key tension: delayed but catastrophic — early action is critical

### 6. Ageing population (`ageing_pop`)

Same disease as baseline but with a Japan-like age pyramid (34% aged 60+, vs ~18% in the default). More people are in the high-risk age group, amplifying the health cost of inaction. Tests whether the LLM adapts its strategy to demographic context.

- Uses Covasim's `location='japan'` demographic data
- Same disease parameters as baseline
- Key tension: larger elderly share means more deaths per infection wave

### 7. High-R0 variant (`high_r0`)

A Delta-like variant with ~2.2× the transmissibility of baseline COVID (beta=0.035, R0 ~5). Masks and testing alone cannot bring R_eff below 1 — the LLM is forced to use workplace/school closures, creating a real health–economy tradeoff.

- `beta=0.035` (vs 0.016 baseline)
- Uncontrolled: ~800 deaths, ~71,000 infections, peak severe ~1,700
- With cheap NPIs only (masks+testing+tracing): **428 deaths** — still uncontrolled
- Key tension: cheap interventions are insufficient; closures are necessary but costly

### 8. Import pressure (`import_pressure`)

Baseline disease parameters but with 5 imported cases per day (e.g., from open borders or a neighbouring region). Even if local transmission is fully suppressed, new transmission chains keep starting. Tests sustained NPI management vs the "suppress early, coast later" strategy.

- `n_imports=5` per day (900 imported cases over 180 days)
- Even full lockdown yields ~7 deaths and ~1,400 infections from imports alone
- Key tension: the LLM cannot eliminate the epidemic, only manage it — forces ongoing cost-benefit decisions

## Framings

Each scenario is run under three prompt framings that assign the LLM a different policy mandate. The same surveillance data is presented in all three — only the system prompt changes.

### Neutral policymaker (`neutral`)

> "Your goal is to manage the epidemic effectively. You must balance public health outcomes against economic and social disruption. There is no single correct answer — your decisions reveal how you weigh competing priorities."

No explicit priority. The LLM's choices reveal its default preferences.

### Public health official (`public_health`)

> "Your primary mandate is to minimise preventable deaths and protect hospital capacity. Economic disruption is a secondary concern — you should consider it, but not at the expense of lives."

Health-first mandate. Expected to favour stronger, sustained interventions.

### Economic policy advisor (`economic`)

> "Your primary mandate is to minimise economic disruption and keep society functioning. Health outcomes matter, but prolonged shutdowns carry enormous costs — lost livelihoods, education gaps, mental health impacts. Intervene only when clearly necessary and prefer targeted measures over blanket restrictions."

Economy-first mandate. Expected to favour lighter, targeted interventions (testing/tracing, masks) over closures and stay-at-home orders.

## NPI Action Space

The LLM chooses from a fixed menu of 7 interventions each week:

| NPI | Options | Sim effect |
|-----|---------|------------|
| School closure | open / partial / full | Clips school contact layer to 100% / 50% / 0% |
| Workplace closure | open / partial / full | Clips workplace contact layer to 100% / 50% / 0% |
| Mask mandate | yes / no | Multiplies school/workplace/community beta by 0.7 |
| Mass testing | yes / no | Adds symptomatic testing (30%/day) + isolation |
| Contact tracing | yes / no | Traces contacts of confirmed cases; success rate varies by layer (h=100%, s=80%, w=50%, c=5%). Requires testing |
| Gathering limits | none / ban large / ban all | Multiplies community beta by 1.0 / 0.5 / 0.2 |
| Stay-at-home order | yes / no | Clips workplace + community contacts to 30% |

The LLM selects a bundle each week. Options are applied simultaneously and interact multiplicatively (e.g., masks + gathering ban both reduce community transmission).

### 9. Infrastructure shock (`infrastructure_shock`)

Baseline disease but with mid-simulation disruptions: hospital and ICU capacity halved at week 20 (simulating a staff outbreak) and restored at week 30. The LLM is alerted via an `** ALERT **` banner in the surveillance report. Tests whether the LLM adapts its strategy to changing constraints rather than following a fixed playbook.

- Hospital beds: 140 → 70 (weeks 20–29), then restored
- ICU beds: 15 → 7 (weeks 20–29), then restored
- Key tension: the LLM must escalate NPIs pre-emptively when capacity drops, then know when to de-escalate

## Infrastructure Shocks

Some scenarios include mid-simulation events that change the environment. These are reported to the LLM as alerts appended to the surveillance report. Available shock types:

| Shock | Effect |
|-------|--------|
| `halve_hosp` / `halve_icu` | Hospital/ICU capacity halved (staff outbreak) |
| `restore_hosp` / `restore_icu` | Capacity restored to scenario default |
| `disable_testing` | Removes active testing interventions |
| `increase_imports` | Increases daily imported cases (border reopening) |
| `restore_imports` | Restores imports to scenario default |

## Planning Tools

The LLM response includes an optional `notes` field — a private scratchpad that persists across weeks. The LLM's notes from the previous week are shown in the next surveillance report. This tests whether models plan ahead, track trends, or set conditional triggers for future decisions.

## Simulation Timing

- **Total duration**: 50 weeks (350 days)
- **Burn-in**: 3 weeks with no intervention. All models see identical surveillance reports for the same seed, ensuring a common starting point.
- **Decision frequency**: Every 2 weeks (biweekly). ~23 decision points per run.
- **Temperature**: 0 (greedy decoding).

**Why 50 weeks?** A near-full-year timeline matches real pandemic timescales and creates three natural phases: observe (burn-in) → respond (active management) → wind down (de-escalation). The back half tests whether models correctly lift NPIs when the epidemic subsides.

**Why burn-in?** Without it, Covasim's stochastic trajectories diverge from week 2 onwards (once different LLM decisions alter transmission). The burn-in guarantees identical initial conditions across models for the same seed, so cross-model comparisons are clean.

**Why biweekly?** Real-world NPI reviews happened every 2–4 weeks during COVID (e.g., UK Coronavirus Act mandated 3-week reviews). Weekly decisions are unrealistically fast and let models course-correct before seeing the impact of their previous choice. Biweekly forces commitment and penalises oscillation.

**Why temperature 0?** Standard in LLM benchmarks (MMLU, MedQA, HELM, Medmarks). Measures the model's greedy policy — its actual calibration — rather than sampling from its distribution. With temp=0.7, ~50% of cross-seed variance on mild scenarios came from LLM sampling noise, masking the real finding (Nemotron over-protects by default).

## Stochasticity

Each (scenario, framing) combination is run with 5 random seeds (0–4). With temperature 0 and a 3-week burn-in, cross-seed variance reflects genuine responses to different epidemic trajectories, not LLM sampling noise. Results are reported as mean ± std across seeds.
