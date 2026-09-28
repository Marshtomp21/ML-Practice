#!/usr/bin/env bash
set -euo pipefail

mode="${1:-all}"
config="${M4_VIT_CONFIG:-configs/experiment/m4_vit_extension.yaml}"
architecture_output="${M4_VIT_ARCH_OUTPUT:-results/metrics/m4_vit_architecture}"
partition_output="${M4_VIT_PARTITION_OUTPUT:-results/metrics/m4_vit_partition}"

run_experiment() {
  local experiment="$1"
  local output="$2"
  python -m experiments.m4_vit_extension \
    --config "$config" --experiment "$experiment" --phase main \
    --output "$output" --resume
  python -m experiments.m4_vit_extension \
    --config "$config" --experiment "$experiment" --phase stability \
    --output "$output" --resume
}

case "$mode" in
  architecture)
    run_experiment architecture "$architecture_output"
    ;;
  partition)
    run_experiment partition "$partition_output"
    ;;
  all)
    run_experiment architecture "$architecture_output"
    run_experiment partition "$partition_output"
    ;;
  *)
    echo "Usage: $0 [architecture|partition|all]" >&2
    exit 2
    ;;
esac
