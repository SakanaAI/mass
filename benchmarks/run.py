"""Run one public-benchmark trial with a served L^(k) and qwen-code."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark", choices=["sab", "mlr"])
    p.add_argument("generation", choices=["L0", "L1", "L2"])
    p.add_argument("trial", type=int)
    p.add_argument("--model-id", required=True, help="Exact served-model-name in vLLM")
    p.add_argument("--ports", default="8001")
    p.add_argument("--concurrent", type=int, default=6)
    p.add_argument("--proxy-host", default="172.17.0.1")
    p.add_argument("--proxy-base", type=int, default=9100)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.trial < 1:
        p.error("trial must be positive")
    root = Path(__file__).resolve().parent
    work = Path(os.environ.get("MASS_BENCH_ROOT", root / "work")).resolve()
    scripts = root / "scripts"
    taskroot = work / a.benchmark / "tasks"
    seeds = root / f"configs/seeds_{a.benchmark}_t{a.trial}.json"
    bundle = root / "runtime/qwen-code"
    cmd = ["harbor", "run", "--path", str(taskroot), "--agent", "bench_qwen_code_agent:BenchQwenCode",
           "--model", a.generation, "--ak", f"model_dir={a.model_id}", "--ak", f"upstream_ports={a.ports}",
           "--ak", f"seed_map={seeds}", "--ak", f"proxy_python={sys.executable}",
           "--ak", f"proxy_host={a.proxy_host}", "--ak", f"proxy_port_base={a.proxy_base}",
           "--ak", f"condition={a.benchmark}_{a.generation}_t{a.trial}",
           "--mounts", json.dumps([{"type": "bind", "source": str(bundle), "target": "/opt/qwen-code", "read_only": True}]),
           "--extra-docker-compose", str(root / "configs/compose_host_gateway.yaml"),
           "--n-concurrent", str(a.concurrent), "--n-attempts", "1", "--timeout-multiplier", "1",
           "--max-retries", "2", "--retry-include", "BenchInfraApiError",
           "--jobs-dir", str(work / a.benchmark / "jobs"), "--job-name", f"{a.benchmark}_{a.generation}_t{a.trial}", "--yes"]
    if a.benchmark == "mlr":
        for name in (root / "configs/mlr_tasks.txt").read_text().splitlines():
            cmd += ["--include-task-name", f"mlr_{name}"]
    print(shlex.join(cmd))
    if a.dry_run:
        return
    if not (bundle / "bin/qwen").is_file():
        raise FileNotFoundError("Build the portable qwen-code bundle first; see benchmarks/README.md")
    names = [p.name for p in taskroot.iterdir() if (p / "task.toml").is_file()]
    required = 102 if a.benchmark == "sab" else 12
    if len(names) != required:
        raise ValueError(f"Expected {required} tasks, found {len(names)}")
    subprocess.run([sys.executable, scripts / "bench_seeds.py", a.benchmark, str(a.trial)], check=True)
    env = {**os.environ, "PYTHONPATH": str(scripts)}
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
