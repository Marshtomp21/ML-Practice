"""Show M4 configuration-scan progress and the current worker log tails."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "data/splits/m4_frozen_20260917.json").read_text())


def lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def main() -> None:
    for experiment, tasks in (("exp1", 61), ("exp2", 50)):
        for domain in ("imagenet", "chncxr"):
            selected = SPLIT["domains"][domain]["configuration_selection"]
            expected = len(selected) * 2 * tasks
            for phase, filename, marker in (
                ("main", "metrics.jsonl", "complete_main.json"),
                ("stability", "stability.jsonl", "complete_stability.json"),
            ):
                got = 0
                completed = 0
                tails = []
                for shard in (0, 1):
                    name = f"m4_{domain}_config_{experiment}_shard{shard}"
                    directory = ROOT / "results/metrics" / name
                    got += lines(directory / filename)
                    completed += (directory / marker).is_file()
                    log = ROOT / "results/logs" / f"{name}_{phase}.log"
                    if log.exists():
                        text = log.read_text(encoding="utf-8", errors="replace").splitlines()
                        if text:
                            tails.append(text[-1])
                print(f"{domain:8} {experiment:4} {phase:9} "
                      f"{got:6}/{expected:<6} shards={completed}/2")
                for tail in tails:
                    if "Traceback" in tail or "Error" in tail or "failed" in tail:
                        print(f"  {tail}")


if __name__ == "__main__":
    main()
