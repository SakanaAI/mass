# Contributing to MASS

For a bug report, include the command, Python version, relevant package versions,
and a small example that reproduces the problem. Include the configuration and
task ID when relevant. Remove API keys, personal paths, and private data from
logs before sharing them.

For a change to the pipeline, describe how it affects workflow search,
trajectory selection, training, or evaluation. Keep the paper's model notation:
`L0`, `L1`, and `L2` represent successive generations of the shared model.
Update the documentation when changing the protocol or output format.

Run the offline checks before submitting a pull request:

```bash
python3 -m unittest discover -s tests -v
python3 -m mass plan --config configs/paper.json
python3 -m mass plan --config configs/cycle2.json
```

Install `requirements-core.txt` to include the RHI integration test. Tests should
use fixtures or mocked model backends; they should not need API credentials,
model weights, or GPUs. For changes that need an actual model run, describe
separately what was tested and what remains unverified.

After changing release files, update the reviewed file list and hashes in
`SOURCE_MANIFEST.json`, then run `python3 tools/check_source.py`. The
[development guide](docs/development.md) describes this step and source archives.
Keep generated experiments, downloaded data, weights, and local credentials out
of source commits.
