"""Download the pinned public base checkpoints; does not load a model."""
import json
from pathlib import Path

if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    root = Path(__file__).resolve().parents[1]
    for name, spec in json.loads((root / "configs/models.json").read_text()).items():
        snapshot_download(**spec, local_dir=root / "models" / name)
