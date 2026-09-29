"""Pair two generation rollout manifests on task and replicate, without judging."""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    def keyed(path):
        rows = json.loads(path.read_text())
        mapped = {(r["task"], r["replicate"]): r["workspace"] for r in rows}
        if len(rows) != len(mapped):
            raise ValueError(f"Duplicate task/replicate in {path}")
        return mapped
    a, b = keyed(args.a), keyed(args.b)
    if not a or set(a) != set(b):
        raise ValueError("Rollout manifests must cover the same nonempty task/replicate set")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps([{"task": task, "a": a[(task, rep)], "b": b[(task, rep)]}
                                    for task, rep in sorted(a)], indent=2) + "\n")
