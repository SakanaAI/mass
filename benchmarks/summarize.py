"""Aggregate per-trial metrics; never substitute published values for outputs."""
import argparse
import json
from pathlib import Path
import statistics


def sab_trial(path, expected=102):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} scored instances, found {len(rows)} in {path}")
    ids = [r.get("instance_id") for r in rows]
    # The pinned official evaluator writes rows in dataset order without IDs.
    # If a later evaluator adds IDs, require them on every row and check uniqueness.
    if any(i is not None for i in ids) and (any(i is None for i in ids) or len(set(map(str, ids))) != expected):
        raise ValueError("Missing or duplicate ScienceAgentBench instance IDs")
    scores = [float(r["success_rate"]) for r in rows]
    if any(not 0 <= s <= 1 for s in scores):
        raise ValueError("Success rates must be between zero and one")
    return statistics.mean(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", help="Official SAB evaluator JSONL, one file per trial for one generation")
    a = p.parse_args()
    values = [sab_trial(path) for path in a.logs]
    print(json.dumps({"trials": values, "mean": statistics.mean(values),
                      "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                      "metric": "success_rate", "instances_per_trial": 102}, indent=2))


if __name__ == "__main__":
    main()
