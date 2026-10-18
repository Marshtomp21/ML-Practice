#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
mkdir -p "$log_dir"
cd "$repo_dir"

pids=()
status=0
for spec in 2:0 3:1 4:2 6:3; do
  gpu="${spec%%:*}"
  worker="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/run_m4_baseline_sensitivity.py \
    --worker-index "$worker" \
    > "$log_dir/m4_baseline_sensitivity_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "A baseline-sensitivity worker failed; inspect per-GPU logs." >&2
  exit "$status"
fi

"$python_bin" scripts/run_m4_baseline_sensitivity.py --merge \
  > "$log_dir/m4_baseline_sensitivity_merge.log" 2>&1
