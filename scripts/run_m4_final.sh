#!/usr/bin/env bash
# Run the frozen M4 final-evaluation cells on the held-out split.
#
# Usage: scripts/run_m4_final.sh main|stability [per-GPU concurrency]
#
# Each domain is split into 12 shards, three per GPU. The main phase defaults to
# one shard at a time per GPU so attribution timings are not contended; the
# stability phase may run the three shards of a GPU concurrently (its timing is
# not a reported metric). GPUs default to 2,3,4,6; override with M4_FINAL_GPUS.
set -euo pipefail

phase="${1:?usage: run_m4_final.sh main|stability [concurrency]}"
concurrency="${2:-1}"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
read -r -a gpus <<< "${M4_FINAL_GPUS:-2 3 4 6}"
shard_count=12
mkdir -p "$log_dir"
cd "$repo_dir"

run_shard() {
  local gpu="$1" domain="$2" shard="$3"
  local name="m4_${domain}_final_shard${shard}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m experiments.m4_matrix \
    --config configs/experiment/m4.yaml --domain "$domain" \
    --split final_evaluation --experiment final --phase "$phase" \
    --output "results/metrics/$name" \
    --shard-index "$shard" --shard-count "$shard_count" --resume \
    > "$log_dir/${name}_${phase}.log" 2>&1
}
export -f run_shard
export phase python_bin log_dir shard_count

pids=()
for index in "${!gpus[@]}"; do
  gpu="${gpus[$index]}"
  (
    for domain in imagenet chncxr; do
      for shard in $(seq "$index" "${#gpus[@]}" $((shard_count - 1))); do
        printf '%s %s %s\n' "$gpu" "$domain" "$shard"
      done
    done | xargs -P "$concurrency" -L 1 bash -c 'run_shard "$@"' _
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "At least one final-evaluation shard failed; see results/logs/m4_*_final_*" >&2
fi
exit "$status"
