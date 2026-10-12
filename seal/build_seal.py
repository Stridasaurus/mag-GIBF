"""seal/build_seal.py — write seal/SEALED_FILES.sha256.

Hashes the COMMITTED content (git blobs at HEAD), not the working copy, so the
manifest is identical on every operating system regardless of line-ending
conversion. Sealed set = everything under results/ except results/post_hoc/,
plus the top-level viability-test/*.py modules that produced those results.

Run from the repo root at the commit being sealed:  python seal/build_seal.py
"""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args, text=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=text, check=True)
    return r.stdout


def sealed_paths():
    paths = git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
    out = []
    for p in paths:
        if p.startswith("results/") and not p.startswith("results/post_hoc/"):
            out.append(p)
        elif p.startswith("viability-test/") and p.count("/") == 1 and p.endswith(".py"):
            out.append(p)
    return sorted(out)


def main():
    head = git("rev-parse", "HEAD").strip()
    lines = []
    for p in sealed_paths():
        blob = subprocess.run(["git", "show", f"HEAD:{p}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        lines.append(f"{hashlib.sha256(blob).hexdigest()}  {p}")
    (ROOT / "seal" / "SEALED_FILES.sha256").write_text("\n".join(lines) + "\n")
    print(f"sealed {len(lines)} files at {head}")


if __name__ == "__main__":
    main()
