#!/usr/bin/env python3
"""Build the ScienceAgentBench task images before agent execution."""
from pathlib import Path
import os
import sys
BENCH_ROOT = Path(os.environ.get("MASS_BENCH_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()

import argparse, concurrent.futures as cf, json, os, time
from pathlib import Path
import docker

TASKS = BENCH_ROOT / "sab" / "tasks"


def build(client, iid, tag):
    t0 = time.time(); name = f"sab_task_{iid}:{tag}"
    try:
        client.images.get(name); return iid, "cached", 0.0
    except docker.errors.ImageNotFound:
        pass
    img, logs = client.images.build(path=str(TASKS / f"sab_{iid}" / "environment"), tag=name, rm=True, forcerm=True)
    for _ in logs: pass
    return iid, "built", round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="20260924"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", nargs="*", type=int, default=None); a = ap.parse_args()
    ids = a.ids or sorted(int(p.name.split("_")[1]) for p in TASKS.iterdir() if p.name.startswith("sab_") and (p / "task.toml").exists())
    client = docker.from_env(timeout=3600)
    results = {}; t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(build, client, i, a.tag) for i in ids]):
            try:
                iid, st, dt = fut.result(); results[iid] = {"status": st, "seconds": dt}
                print(f"sab_{iid}: {st} {dt}s  ({len(results)}/{len(ids)}, {round(time.time() - t0)} s elapsed)", flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED:", repr(e)[:300], flush=True)
    (TASKS / f"prebuild_{a.tag}.json").write_text(json.dumps(results, indent=1) + "\n")
    print("done:", len([r for r in results.values() if r["status"] in ("built", "cached")]), "/", len(ids))


if __name__ == "__main__":
    main()
