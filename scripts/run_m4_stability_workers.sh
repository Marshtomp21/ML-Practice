#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
mkdir -p "$log_dir"
cd "$repo_dir"

pids=()
for spec in 2:0 3:1 4:2 6:3; do
  gpu="${spec%%:*}"
  worker="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/run_m4_stability_workers.py \
    --worker-index "$worker" \
    > "$log_dir/m4_chncxr_exp2_stability_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "At least one auxiliary worker failed; inspect per-GPU logs." >&2
  exit "$status"
fi

"$python_bin" scripts/run_m4_stability_workers.py --merge \
  > "$log_dir/m4_chncxr_exp2_stability_merge.log" 2>&1
