"""
I/O utilities for the outbreak policy benchmark.
"""

import json
import os


def sanitize_model_name(model):
    """Sanitize a model name for use in file/directory names."""
    return model.replace("/", "--").replace(":", "-")


def load_runs(output_dir="outputs/runs", model_filter=None):
    """Load benchmark run results from output directory.

    Each model's results are in a subdirectory of output_dir.
    """
    runs = []
    for model_dir in sorted(os.listdir(output_dir)):
        model_path = os.path.join(output_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        if model_filter and model_dir != model_filter:
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(model_path, fname)) as f:
                runs.append(json.load(f))
    return runs
