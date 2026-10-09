#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
mkdir -p "$log_dir"
cd "$repo_dir"

run_parallel_phase() {
  local experiment="$1"
  local phase="$2"
  local pids=()
  for spec in 2:0 3:1 4:2 6:3; do
    local gpu="${spec%%:*}"
    local worker="${spec##*:}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" scripts/run_m4_vit_workers.py \
      --experiment "$experiment" --phase "$phase" --worker-index "$worker" \
      > "$log_dir/m4_vit_${experiment}_${phase}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "A $experiment $phase worker failed; inspect per-GPU logs." >&2
    return "$status"
  fi
  "$python_bin" scripts/run_m4_vit_workers.py \
    --experiment "$experiment" --phase "$phase" --merge \
    > "$log_dir/m4_vit_${experiment}_${phase}_merge.log" 2>&1
}

CUDA_VISIBLE_DEVICES=3 "$python_bin" -m experiments.m4_vit_extension \
  --config configs/experiment/m4_vit_extension.yaml \
  --experiment partition --phase screen \
  --output results/metrics/m4_vit_partition --resume \
  > "$log_dir/m4_vit_partition_screen.log" 2>&1

run_parallel_phase partition main
run_parallel_phase partition stability
