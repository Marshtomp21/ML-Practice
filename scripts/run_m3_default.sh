#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
mkdir -p "$log_dir"
cd "$repo_dir" || exit 1

pids=()
names=()
for shard in 0 1 2 3; do
  gpu="$shard"
  name="m3_imagenet_default_20260913_shard${shard}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m experiments.m3_default \
    --config configs/experiment/m3_imagenet_default.yaml \
    --output "results/metrics/$name" \
    --shard-index "$shard" --shard-count 4 --resume \
    > "$log_dir/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

for shard in 0 1 2 3; do
  gpu="$((shard + 4))"
  name="m3_chncxr_default_20260913_shard${shard}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m experiments.m3_default \
    --config configs/experiment/m3_chncxr_default.yaml \
    --output "results/metrics/$name" \
    --shard-index "$shard" --shard-count 4 --resume \
    > "$log_dir/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
done

cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

failed=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  status=$?
  printf '%s exit status: %s\n' "${names[$index]}" "$status"
  if (( status != 0 )); then
    failed=1
  fi
done
exit "$failed"
