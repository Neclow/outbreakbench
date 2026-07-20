#!/usr/bin/env bash
set -euo pipefail

while IFS= read -r model || [[ -n "$model" ]]; do
    [[ -z "$model" || "$model" == \#* ]] && continue
    pixi run benchmark --model "$model"
done < models.txt
