"""
Batch runner: sweep models x scenarios x framings x seeds.

Usage:
    # Ollama (default)
    pixi run python scripts/run_benchmark.py \
        --model nemotron nemotron-3-nano:30b \
        --model qwen qwen3.6:35b-a3b

    # vLLM
    pixi run python scripts/run_benchmark.py \
        --model nemotron nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
            --base-url http://localhost:8000/v1 \
        --model qwen Qwen/Qwen3.6-35B-A3B \
            --base-url http://localhost:8001/v1

    # Subset of scenarios/framings
    pixi run python scripts/run_benchmark.py \
        --model nemotron nemotron-3-nano:30b \
        --scenarios baseline ebola_like \
        --framings neutral economic \
        --seeds 3

    # Dry run / smoke test only
    pixi run python scripts/run_benchmark.py \
        --model nemotron nemotron-3-nano:30b \
        --dry-run

    pixi run python scripts/run_benchmark.py \
        --model nemotron nemotron-3-nano:30b \
        --smoke-test
"""

import argparse
import json
import os
import sys
import time
import traceback

from outbreakbench.llm import make_client, smoke_test
from outbreakbench.runner import run_benchmark
from outbreakbench.scenarios import SCENARIOS

ALL_FRAMINGS = ["neutral", "public_health", "economic"]
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run outbreak policy benchmark")
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("NAME", "MODEL_ID"),
        required=True,
        help="Model to benchmark (repeatable): name model_id",
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
        default=0.7,
        help="LLM sampling temperature (default: 0.7)",
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
        "--skip-existing",
        action="store_true",
        help="Skip runs whose output file already exists",
    )
    return parser.parse_args()


def run_name(model_name, scenario, framing, seed):
    return f"{scenario}_{framing}_seed{seed}"


def main():
    args = parse_args()

    models = {}
    for name, model_id in args.model:
        models[name] = {"base_url": args.base_url, "model_id": model_id}

    clients = {}
    for name, cfg in models.items():
        clients[name] = make_client(
            base_url=cfg["base_url"],
            model=cfg["model_id"],
            temperature=args.temperature,
        )

    # Smoke test: always run before benchmark, or standalone with --smoke-test
    print("Smoke testing models...")
    all_ok = True
    for name in models:
        print(f"  {name} ({models[name]['model_id']})... ", end="", flush=True)
        ok, msg = smoke_test(clients[name])
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

    runs = []
    for model_name in models:
        for scenario in args.scenarios:
            for framing in args.framings:
                for seed in range(args.seeds):
                    runs.append((model_name, scenario, framing, seed))

    print(f"Benchmark plan: {len(runs)} runs")
    print(f"  Models:    {list(models.keys())}")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Framings:  {args.framings}")
    print(f"  Seeds:     0..{args.seeds - 1}")
    print()

    if args.dry_run:
        for model_name, scenario, framing, seed in runs:
            name = run_name(model_name, scenario, framing, seed)
            out = os.path.join(args.output, model_name, f"{name}.json")
            exists = " [EXISTS]" if os.path.exists(out) else ""
            print(f"  {model_name}/{name}{exists}")
        return

    completed = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    for i, (model_name, scenario, framing, seed) in enumerate(runs):
        name = run_name(model_name, scenario, framing, seed)
        out_dir = os.path.join(args.output, model_name)
        out_path = os.path.join(out_dir, f"{name}.json")

        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            print(f"[{i+1}/{len(runs)}] SKIP {model_name}/{name} (exists)")
            continue

        print(
            f"[{i+1}/{len(runs)}] {model_name}/{name}...",
            end=" ",
            flush=True,
        )

        try:
            result = run_benchmark(
                scenario_key=scenario,
                call_llm=clients[model_name],
                framing=framing,
                seed=seed,
            )
            result["model"] = model_name
            result["model_id"] = models[model_name]["model_id"]

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
    print(f"\nDone: {completed} completed, {failed} failed, {skipped} skipped")
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
                "model_id": result.get("model_id", ""),
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
