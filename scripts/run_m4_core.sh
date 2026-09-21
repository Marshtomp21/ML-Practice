#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
mkdir -p "$log_dir"
cd "$repo_dir"

run_stage() {
  local experiment="$1"
  local phase="$2"
  local pids=()
  local names=()
  local gpu=0
  local domain
  local shard
  local name
  local status=0
  for domain in imagenet chncxr; do
    for shard in 0 1; do
      name="m4_${domain}_config_${experiment}_shard${shard}"
      CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m experiments.m4_matrix \
        --config configs/experiment/m4.yaml \
        --domain "$domain" --split configuration_selection \
        --experiment "$experiment" --phase "$phase" \
        --output "results/metrics/$name" \
        --shard-index "$shard" --shard-count 2 --resume \
        > "$log_dir/${name}_${phase}.log" 2>&1 &
      pids+=("$!")
      names+=("$name")
      gpu=$((gpu + 1))
    done
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf '%s %s complete\n' "${names[$index]}" "$phase"
    else
      printf '%s %s failed; see results/logs\n' "${names[$index]}" "$phase" >&2
      status=1
    fi
  done
  return "$status"
}

run_stage exp1 main
"$python_bin" scripts/analyze_m4_exp3.py \
  --output results/metrics/m4_exp3_config_20260917 \
  > "$log_dir/m4_exp3_config_analysis.log" 2>&1
run_stage exp2 main
run_stage exp1 stability
run_stage exp2 stability
