#!/usr/bin/env python3
"""Collect predicted programs and optionally run the official ScienceAgentBench evaluator."""
from pathlib import Path
import os
import sys
BENCH_ROOT = Path(os.environ.get("MASS_BENCH_ROOT", str(Path(__file__).resolve().parents[1] / "work"))).resolve()

import argparse, json, os, subprocess, sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SAB_REPO = PKG / "vendor" / "ScienceAgentBench"
PY = Path(sys.executable)
INDEX = BENCH_ROOT / "sab" / "tasks" / "index.json"
BENCH = BENCH_ROOT / "sab" / "benchmark"


def collect(job_dir: Path, out: Path) -> dict:
    index = json.loads(INDEX.read_text()); out.mkdir(parents=True, exist_ok=True)
    trials = {}
    for t in sorted(job_dir.iterdir()):
        if not (t / "config.json").exists():
            continue
        task = t.name.split("__")[0]
        if task in index:
            trials.setdefault(task, []).append(t)  # a retried trial: keep the latest (harbor removes the failed attempt's result)
    report = {"found": 0, "missing": [], "per_task": {}}
    for task, meta in index.items():
        pred = meta["pred_program_name"]; src = None
        for t in reversed(trials.get(task, [])):
            c = t / "verifier" / "pred_programs" / pred
            if c.exists() and c.stat().st_size > 0:
                src = c; break
        if src is None:
            (out / pred).write_text("ERROR"); report["missing"].append(task); report["per_task"][task] = None
        else:
            (out / pred).write_text(src.read_text(errors="replace").replace("/workspace", "."))
            report["found"] += 1; report["per_task"][task] = str(src.parent.parent.parent.name)
    (out / "collect_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"collected {report['found']}/{len(index)} programs into {out}; missing: {len(report['missing'])}")
    return report


def evaluate(out: Path, run_id: str, max_workers: int) -> Path:
    out = out.resolve()
    log = out / f"eval_{run_id}.jsonl"
    cmd = [str(PY), "-m", "evaluation.harness.run_evaluation", "--benchmark_path", str(BENCH), "--pred_program_path", str(out),
           "--log_fname", str(log), "--run_id", run_id, "--split", "verified", "--max_workers", str(max_workers), "--cache_level", "base"]
    print(" ".join(cmd)); sys.stdout.flush()
    subprocess.run(cmd, cwd=str(SAB_REPO), env={**os.environ, "PYTHONPATH": str(SAB_REPO)}, check=True)
    return log


def summarize(log: Path) -> dict:
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    n = len(rows)
    if n != 102:
        raise ValueError(f"Expected all 102 official evaluator rows, found {n}")
    s = {"n": n, "SR": sum(r["success_rate"] for r in rows) / n, "VER": sum(r["valid_program"] for r in rows) / n,
         "CBS": sum(r["codebert_score"] for r in rows) / n}
    print(json.dumps(s)); return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("job_dir"); ap.add_argument("out_dir"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--run_id", default=None); ap.add_argument("--max_workers", type=int, default=8); a = ap.parse_args()
    collect(Path(a.job_dir), Path(a.out_dir))
    if a.eval:
        summarize(evaluate(Path(a.out_dir), a.run_id or Path(a.out_dir).name, a.max_workers))
