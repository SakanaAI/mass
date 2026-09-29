"""Fetch public benchmark code at fixed commits; no benchmark execution."""
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    for name, spec in json.loads((root / "configs/sources.json").read_text()).items():
        dest = root / "vendor" / name
        if dest.exists():
            head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
            if head != spec["commit"]:
                raise ValueError(f"Existing checkout has another revision: {dest}")
            continue
        dest.mkdir(parents=True)
        for args in [["init"], ["remote", "add", "origin", spec["url"]],
                     ["fetch", "--depth", "1", "origin", spec["commit"]], ["checkout", "--detach", "FETCH_HEAD"]]:
            subprocess.run(["git", "-C", str(dest), *args], check=True)
    # Keep CodeBERTScore import failures from zeroing executable-program scores.
    sab = root / "vendor/ScienceAgentBench"
    patch = root / "patches/sab_codebert.patch"
    applied = subprocess.run(["git", "-C", str(sab), "apply", "--reverse", "--check", str(patch)], capture_output=True).returncode == 0
    if not applied:
        subprocess.run(["git", "-C", str(sab), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(sab), "apply", str(patch)], check=True)


if __name__ == "__main__":
    main()
