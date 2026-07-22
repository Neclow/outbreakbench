"""
Batch runner: sweep models x scenarios x framings x seeds.

Usage:
    # Ollama (default)
    pixi run benchmark \
        --model nemotron-3-nano:30b \
        --model qwen3:14b

    # vLLM
    pixi run benchmark \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --model Qwen/Qwen3.6-35B-A3B \
        --base-url http://localhost:8000/v1

    # Subset of scenarios/framings
    pixi run benchmark \
        --model nemotron-3-nano:30b \
        --scenarios baseline ebola_like \
        --framings neutral economic \
        --seeds 3

    # Dry run / smoke test only
    pixi run benchmark \
        --model nemotron-3-nano:30b \
        --dry-run

    pixi run smoke
"""

import argparse
import json
import os
import sys
import time
import traceback

from outbreakbench.llm import make_client, smoke_test
from outbreakbench.simulator import run_benchmark
from outbreakbench.scenarios import SCENARIOS

ALL_FRAMINGS = ["neutral", "public_health", "economic"]
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run outbreak policy benchmark")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model to benchmark (repeatable), e.g. nemotron-3-nano:30b",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL for all models (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS.keys()),
        choices=list(SCENARIOS.keys()),
        help="Scenarios to run (default: all)",
    )
    parser.add_argument(
        "--framings",
        nargs="+",
        default=ALL_FRAMINGS,
        choices=ALL_FRAMINGS,
        help="Framings to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds, 0..N-1 (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="outputs/runs",
        help="Output directory (default: outputs/runs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run plan without executing",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test for each model and exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if output file already exists (default: skip existing)",
    )
    return parser.parse_args()


def _sanitize(model):
    return model.replace("/", "--").replace(":", "-")


def run_name(model, scenario, framing, seed):
    return f"{scenario}_{framing}_seed{seed}"


def main():
    args = parse_args()

    runs = []
    for model in args.model:
        for scenario in args.scenarios:
            for framing in args.framings:
                for seed in range(args.seeds):
                    runs.append((model, scenario, framing, seed))

    if not args.overwrite:
        runs = [
            (model, scenario, framing, seed)
            for model, scenario, framing, seed in runs
            if not os.path.exists(
                os.path.join(
                    args.output, _sanitize(model),
                    f"{run_name(model, scenario, framing, seed)}.json",
                )
            )
        ]

    models_needed = sorted(set(m for m, _, _, _ in runs))

    if not args.smoke_test and not models_needed:
        print("All runs already exist. Nothing to do.")
        return

    clients = {}
    for model in (args.model if args.smoke_test else models_needed):
        clients[model] = make_client(
            base_url=args.base_url,
            model=model,
            temperature=args.temperature,
        )

    print("Smoke testing models...")
    all_ok = True
    for model in clients:
        print(f"  {model}... ", end="", flush=True)
        ok, msg = smoke_test(clients[model])
        if ok:
            print(f"PASS")
        else:
            print(f"FAIL — {msg}")
            all_ok = False

    if args.smoke_test:
        sys.exit(0 if all_ok else 1)

    if not all_ok:
        print("\nSome models failed smoke test. Aborting.")
        print("Check that your server is running and the model is loaded.")
        sys.exit(1)

    print()

    print(f"Benchmark plan: {len(runs)} runs")
    print(f"  Models:    {args.model}")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Framings:  {args.framings}")
    print(f"  Seeds:     0..{args.seeds - 1}")
    print()

    if args.dry_run:
        for model, scenario, framing, seed in runs:
            name = run_name(model, scenario, framing, seed)
            out = os.path.join(args.output, _sanitize(model), f"{name}.json")
            exists = " [EXISTS]" if os.path.exists(out) else ""
            print(f"  {_sanitize(model)}/{name}{exists}")
        return

    completed = 0
    failed = 0
    t0 = time.time()

    for i, (model, scenario, framing, seed) in enumerate(runs):
        name = run_name(model, scenario, framing, seed)
        out_dir = os.path.join(args.output, _sanitize(model))
        out_path = os.path.join(out_dir, f"{name}.json")

        print(
            f"[{i+1}/{len(runs)}] {model}/{name}...",
            end=" ",
            flush=True,
        )

        try:
            result = run_benchmark(
                scenario_key=scenario,
                call_llm=clients[model],
                framing=framing,
                seed=seed,
            )
            result["model"] = model
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            r = result["final_results"]
            print(
                f"deaths={r['cum_deaths']}, "
                f"infections={r['cum_infections']}, "
                f"elapsed={result['elapsed']:.1f}s"
            )
            completed += 1

        except Exception as e:
            failed += 1
            print(f"FAILED: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nDone: {completed} completed, {failed} failed")
    print(f"Total time: {elapsed:.1f}s")

    if completed > 0:
        write_summary(args.output)


def write_summary(output_dir):
    """Aggregate all run results into a summary JSON."""
    rows = []
    for model_dir in sorted(os.listdir(output_dir)):
        model_path = os.path.join(output_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(model_path, fname)) as f:
                result = json.load(f)
            rows.append({
                "model": result.get("model", model_dir),
                "scenario": result["scenario"],
                "framing": result["framing"],
                "seed": result["seed"],
                **result["final_results"],
                "elapsed": round(result["elapsed"], 1),
                "n_parse_failures": sum(
                    1 for d in result["decisions"]
                    if d["justification"].startswith("[PARSE FAILURE")
                ),
            })

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Summary written to {summary_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
