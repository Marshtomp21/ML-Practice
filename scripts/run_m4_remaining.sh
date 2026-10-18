#!/usr/bin/env bash
# Chain the remaining M4 GPU work after the final-evaluation main phase:
# final-evaluation stability (three shards per GPU) and, in parallel, the
# OrdShap pilot -> main -> moves sequence on the first GPU of the pool.
set -uo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="/mnt/data0601/miniconda3/envs/cjy_mob/bin/python"
log_dir="$repo_dir/results/logs"
read -r -a gpus <<< "${M4_FINAL_GPUS:-2 3 4 6}"
cd "$repo_dir"

until [ "$(ls results/metrics/m4_*_final_shard*/complete_main.json 2>/dev/null | wc -l)" -ge 24 ]; do
  if ! pgrep -f "run_m4_final.sh main" >/dev/null; then
    echo "$(date '+%F %T') main orchestrator exited before all 24 shards completed" >&2
    exit 1
  fi
  sleep 300
done
echo "$(date '+%F %T') main phase complete; starting stability and OrdShap"

M4_FINAL_GPUS="${gpus[*]}" scripts/run_m4_final.sh stability 3 \
  > "$log_dir/m4_final_stability_orchestrator.log" 2>&1 &
stability_pid=$!

(
  for phase in pilot main moves; do
    CUDA_VISIBLE_DEVICES="${gpus[0]}" "$python_bin" -m experiments.m4_ordshap \
      --phase "$phase" --resume > "$log_dir/m4_ordshap_${phase}.log" 2>&1 || {
        echo "$(date '+%F %T') OrdShap $phase failed" >&2; exit 1; }
    echo "$(date '+%F %T') OrdShap $phase complete"
  done
) &
ordshap_pid=$!

status=0
wait "$stability_pid" || { echo "$(date '+%F %T') stability phase failed" >&2; status=1; }
wait "$ordshap_pid" || status=1
echo "$(date '+%F %T') remaining M4 GPU work finished with status $status"
exit "$status"
