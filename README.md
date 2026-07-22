# OutbreakBench

A benchmark for evaluating LLM decision-making during simulated epidemic outbreaks. Models act as public health advisors, choosing non-pharmaceutical interventions (NPIs) each week based on surveillance data. Their policies are executed in [Covasim](https://github.com/starsimhub/covasim), and outcomes are compared against reference baselines.

## Setup

Requires [Pixi](https://pixi.sh) and a running [Ollama](https://ollama.com) server (or any OpenAI-compatible endpoint).

```bash
pixi install
```

## Usage

```bash
# Run a benchmark sweep
pixi run benchmark --model nemotron-3-nano:30b

# Run all models listed in models.txt
pixi run sweep

# Generate reference baselines (no-intervention + full-lockdown)
pixi run baselines

# Analyze results
pixi run analyze   # LLM decision analysis
pixi run plot      # Simulation outcome plots
```

Use `--help` on any script for the full set of options (scenarios, framings, seeds, base URL for vLLM, etc.).

## Project structure

```
outbreakbench/       Core package
  simulator.py         Covasim wrapper
  scenarios.py         Scenario definitions
  npis.py              NPI logic
  llm.py               LLM client (OpenAI-compatible)
  metrics.py           Evaluation metrics
scripts/
  run/                 Benchmark and baseline runners
  validate/            Scenario and NPI validation checks
  analyze/             Decision analysis and outcome plotting
models.txt           Models to sweep (one Ollama tag per line)
outputs/             Benchmark results
```
