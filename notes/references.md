# NPI Modeling References

## School closure → workforce absenteeism

- **Sadique, Adams & Edmunds (2008)** — BMC Public Health 8:135. 16.1% UK workforce absent when schools close. Health sector: 31% have dependent children. Cost: ~GBP 1B/week.
- **Lempel, Epstein & Hammond (2009)** — PLoS Currents. 23% US civilian workers affected. Healthcare: 6-19% personnel reduction for 4-week closure.
- **Bayham & Fenichel (2020)** — Lancet Public Health 5:e271-e278. 15% unmet childcare need for healthcare workers. If 15% healthcare decline raises CFR from 2.00% to 2.35%, school closures increase net mortality.
- **Ferguson et al. (2006)** — Nature 442:448-452. School closure modeled with +50% household contacts and +25% community contacts for absent individuals.

## Mask compliance decay

- **Eikenberry et al. (2020)** — PMC7186508. `beta(t) = beta_min + (beta_0 - beta_min) * exp(-r*(t-t_0))`, r = 0.03-0.04/day, half-life 17-23 days.
- **Pedersen & Meneghini (2021)** — PMC8539631. `beta(t) = beta_0 * [1 - phi * (1 - f * exp(-k*mu*Q(t)))]`. f = 0.65 (35% stopped complying). Estimated 32,000 excess deaths in Italy from fatigue.
- **Teslya et al. (2022)** — PMC9675824. Vaccination-accelerated compliance loss. Baseline compliance duration 30 days; at 33% vaccination, drops to 7 days.
- Empirical: BC-Mix survey — pre-mandate ~78%, during mandate >=84%, post-mandate 38.1%.

## Economic costs of NPIs

- **Juneau et al. (2021)** — Applied Health Economics and Health Policy, PMC8192223. Systematic review of 31 studies. Testing $25-641/sample, contact tracing $41-67/contact, quarantine $41 (home) vs $1,062 (hotel)/person, masks $0.15-2.14/unit, school closure $125/student/day.
- **Li & Spall (2022)** — SN Operations Research Forum 3(4):68. Covasim wrapper with DSPSA optimization. Testing $36/test, VSL $9.3M/death, treatment $3,994 (outpatient) / $30,000 (hospitalized). 100k agents, 60-day horizon.
- **Bushaj et al. (2022) "SiRL"** — Annals of Operations Research. Covasim + DRL. 9 action types at 15-day intervals. Economic reward: `E = S + V1 + V2 + R - α*I - β*H - γ*C - D`. 500k agents, ~30k episodes.

## LLM-as-policymaker (no Covasim papers exist)

- **Aoki & Ghaffarzadegan (2026)** — arXiv:2601.04245. GPT-5 nano as "pragmatic mayor" in SEIR (1M pop). Single continuous action (0-100% shutdown). No cost function.
- **Shi et al. (2026)** — arXiv:2601.09264. Multi-agent LLMs as state-level coordinators in SEIQRD. Up to 63.7% infection reduction.
- **MechSim (Yang et al. 2026)** — arXiv:2606.04505. Claude Sonnet 4.6, DeepSeek V3.2, Qwen 3-235B on policy ranking. Precision@3: 0.82 (Claude).
- **Epi-LLM (Ferencz et al. 2026)** — arXiv:2606.02867. Starsim (Covasim fork). LLMs as individuals (not policymakers) making quarantine decisions. Tested Nemotron 120B.

## Covasim NPI papers

- **Panovska-Griffiths et al. (2020)** — Lancet Child & Adolescent Health. UK school reopening. Lockdown: 2% school, 20% workplace/community transmission. Code: `github.com/Jasminapg/Covid-19-Analysis`.
- **Panovska-Griffiths et al. (2021)** — Scientific Reports 11:8747. Mask efficacy: mean 45% (range 25-70%). Effective coverage 15-30%.
- **Stuart et al. (2021)** — BMJ Open 11(4). NSW. Masks 30% reduction (15-45%). Tracing: h=100%, s=95%, w=90%. Quarantine compliance 90%. Code: `github.com/optimamodel/covid_nsw`.
- **Abeysuriya et al. (2022)** — BMC Infectious Diseases 22:232. Victoria. Quarantine compliance 75% (tested 50-100%). Tracing capacity 250 cases/day.
- **Zhang et al. (2026)** — Scientific Reports 16:10627. Covasim + OpenAI Gym + DQN/PPO. Three continuous interventions. Multi-objective reward with dynamic weighting.
- **Cohen et al. (2020)** — "Schools are not islands." Household-mediated school-to-community coupling, but no causal absenteeism mechanism.

## Key gaps in Covasim literature

1. No compliance fatigue implementation
2. No school-closure → workforce absenteeism coupling
3. No LLM-as-policymaker integration
4. Only 2 papers with explicit economic cost functions
