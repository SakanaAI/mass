#!/usr/bin/env python3
"""Paired public-benchmark seeds: 3100000 + benchmark*10000 + trial*1000 + task_index."""
import json, os, sys
from pathlib import Path

BENCH_ROOT = Path(os.environ.get("MASS_BENCH_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()
BASE = 3_100_000
BENCH = {"tb2": 0, "sab": 1, "ds": 2, "mlr": 3}
PKG = Path(__file__).resolve().parent.parent
TB2_TASKS = BENCH_ROOT / "tb2" / "tasks" / "terminal-bench"


def task_names(bench: str) -> list[str]:
    if bench == "tb2":
        names = sorted(p.name for p in TB2_TASKS.iterdir() if (p / "task.toml").exists())
        assert len(names) == 89, len(names)
        return names
    if bench == "sab":
        return [f"sab_{i}" for i in range(1, 103)]
    if bench == "ds":
        names = sorted(p.name for p in (BENCH_ROOT / "ds" / "tasks").iterdir() if p.name.startswith("ds_") and (p / "task.toml").exists())
        assert len(names) == 74, len(names)
        return names
    return sorted(p.name for p in (BENCH_ROOT / "mlr" / "tasks").iterdir() if p.name.startswith("mlr_") and (p / "task.toml").exists())


def main():
    bench, trial = sys.argv[1], int(sys.argv[2])
    names = task_names(bench)
    seeds = {n: BASE + BENCH[bench] * 10_000 + trial * 1_000 + i for i, n in enumerate(names)}
    out = PKG / "configs" / f"seeds_{bench}_t{trial}.json"
    out.write_text(json.dumps(seeds, indent=1) + "\n")
    reg_p = PKG / "configs" / "seed_registry.json"
    reg = json.loads(reg_p.read_text()) if reg_p.exists() else {"formula": "3100000 + bench*10000 + trial*1000 + task_index", "bench": BENCH, "files": {}}
    reg["files"][out.name] = {"bench": bench, "trial": trial, "min": min(seeds.values()), "max": max(seeds.values()), "n": len(seeds)}
    reg_p.write_text(json.dumps(reg, indent=1) + "\n")
    print(f"{out}: {len(seeds)} seeds {min(seeds.values())}..{max(seeds.values())}")


if __name__ == "__main__":
    main()
