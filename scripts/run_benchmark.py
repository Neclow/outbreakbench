"""
Batch runner: sweep models × scenarios × framings × seeds.

Usage:
    pixi run python scripts/run_benchmark.py \
        --model nemotron http://localhost:8000/v1 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --model qwen http://localhost:8001/v1 Qwen/Qwen3.6-35B-A3B

    # Subset of scenarios/framings
    pixi run python scripts/run_benchmark.py \
        --model nemotron http://localhost:8000/v1 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --scenarios baseline ebola_like \
        --framings neutral economic \
        --seeds 3

    # Dry run to see what would be executed
    pixi run python scripts/run_benchmark.py \
        --model nemotron http://localhost:8000/v1 nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --dry-run
"""

import argparse
import json
import os
import time
import traceback

from outbreakbench.llm import make_client
from outbreakbench.runner import run_benchmark
from outbreakbench.scenarios import SCENARIOS

ALL_FRAMINGS = ["neutral", "public_health", "economic"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run outbreak policy benchmark")
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "BASE_URL", "MODEL_ID"),
        required=True,
        help="Model to benchmark (repeatable): name base_url model_id",
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
    for name, base_url, model_id in args.model:
        models[name] = {"base_url": base_url, "model_id": model_id}

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

    clients = {}
    for name, cfg in models.items():
        clients[name] = make_client(
            base_url=cfg["base_url"],
            model=cfg["model_id"],
            temperature=args.temperature,
        )

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
