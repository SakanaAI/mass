"""Build the official SAB base image without executing predictions."""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    repo = root / "vendor/ScienceAgentBench"
    work = Path(os.environ.get("MASS_BENCH_ROOT", root / "work")).resolve()
    benchmark = work / "sab/benchmark"
    if not (benchmark / "datasets").is_dir():
        raise FileNotFoundError("Obtain the official verified benchmark archive; see README.md")
    code = """
import docker, sys
from datasets import load_dataset
from evaluation.harness.docker_build import build_base_images
build_base_images(docker.from_env(), list(load_dataset('osunlp/ScienceAgentBench', split='verified')),
                  sys.argv[1], sys.argv[2], False)
"""
    pred = work / "sab/base_build_predictions"
    pred.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", code, str(benchmark), str(pred)], cwd=repo,
                   env={**os.environ, "PYTHONPATH": str(repo)}, check=True)
