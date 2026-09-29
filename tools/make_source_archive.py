"""Create a ZIP from the reviewed source manifest, excluding all run outputs."""
import argparse
import json
from pathlib import Path
import zipfile

from check_source import ROOT, check


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    errors, _ = check()
    if errors:
        raise SystemExit("Source check failed:\n" + "\n".join(errors))
    if a.output.exists():
        raise FileExistsError(a.output)
    names = sorted(json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())) + ["SOURCE_MANIFEST.json"]
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in names:
            item = zipfile.ZipInfo(f"mass/{name}", (2026, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, (ROOT / name).read_bytes())
    print(f"Wrote {a.output}: {a.output.stat().st_size} bytes, {len(names)} files")
