# Development

Run commands from the repository root. The pipeline is imported directly from
the checkout, so an editable installation is not required.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests use temporary directories and synthetic fixtures. They do not start
model servers, make API calls, or launch training. Install
`requirements-core.txt` to include the mocked-backend RHI integration test.
Training dependencies and benchmark containers are only needed for their
respective experiments.

GitHub Actions runs these tests, both configuration previews, and the source
check on Python 3.12 and 3.13 with the pinned core dependencies. It does not
launch model servers, download weights, or run benchmark experiments.

## Experiment configurations

Start from `configs/paper.json` or `configs/cycle2.json` and save a local copy
such as `configs/local-demo.json`. The [single-task example](running.md#try-one-task)
shows a small search configuration.

Use a new `run_dir` when changing a configuration. The driver saves the
configuration on the first stage and rejects subsequent changes in that run
directory. This keeps reused episodes associated with their original settings.
Keep API credentials in environment variables. Local environments, generated
outputs, and `configs/local*.json` are excluded by `.gitignore`.

## Source archives

The source archive tools are optional; they are not needed to run experiments.
`SOURCE_MANIFEST.json` lists the files included in a release and their SHA-256
hashes, including the README teaser in `assets/`. Generated runs, downloaded
dependencies, model weights, Git history, and local configuration files are
not part of that manifest.

Check the listed files and build an archive:

```bash
python3 tools/check_source.py
python3 tools/make_source_archive.py --output ../mass-source.zip
```

The output path must not already exist. Files are stored under a single `mass/`
directory. The archive uses fixed timestamps and permissions.
The checker detects changed files and common personal-path, email, and
credential patterns in the listed text files. The reviewed teaser PNG is
checked by its hash and file signature; its visible content and metadata
require manual review. Other binary files are rejected. The checker does not
inspect a hosting account or Git history, and it is not an exhaustive identity
detector.

After editing source files, review the file list before updating the manifest.
Add or remove entries explicitly when adding or removing source files. Then
refresh the hashes of the listed files:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = Path("SOURCE_MANIFEST.json")
names = sorted(json.loads(manifest.read_text()))
hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
manifest.write_text(json.dumps(hashes, indent=2) + "\n")
PY
python3 tools/check_source.py
```

Refreshing hashes records the new contents; it does not review them. Only list
source and documentation intended for distribution.
