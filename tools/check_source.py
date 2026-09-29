"""Check reviewed source integrity and common personal-path/credential leaks offline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "personal absolute path": re.compile(r"/(?:home|Users)/[\w.-]+"),
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "possible private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "possible API credential": re.compile(r"\b(?:sk|ghp)[-_][A-Za-z0-9_-]{25,}"),
}


def check(root=ROOT, deny_terms=()):
    root = Path(root)
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    problems = []
    for name, expected in manifest.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            problems.append(f"Unsafe manifest path: {name}")
            continue
        path = root / rel
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            problems.append(f"Symlink: {name}")
            continue
        if not path.is_file():
            problems.append(f"Missing file: {name}")
            continue
        data = path.read_bytes()
        if len(data) > 2_000_000:
            problems.append(f"Unexpectedly large source file: {name}")
        if hashlib.sha256(data).hexdigest() != expected:
            problems.append(f"Source changed after review: {name}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"Unexpected binary source: {name}")
            continue
        for kind, pattern in PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{kind}: {name}")
        for term in deny_terms:
            if term.casefold() in (name + "\n" + text).casefold():
                problems.append(f"Forbidden identity term: {name}")
    return problems, len(manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deny-term", action="append", default=[])
    a = p.parse_args()
    errors, n = check(deny_terms=a.deny_term)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {n} manifest files; {len(errors)} problems.")
    raise SystemExit(bool(errors))
