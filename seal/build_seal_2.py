"""seal/build_seal_2.py -- write seal/SEALED_FILES_2.sha256 (the second seal).

Seals the pre-registered completion (SPEC S.17): results/prereg_completion/,
results/reports/, and the two modules that produced them. Hashes committed git
blobs at HEAD. The first seal (SEALED_FILES.sha256) is not touched.

Run from the repo root at the commit being sealed:  python seal/build_seal_2.py
"""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ("results/prereg_completion/", "results/reports/")
MODULES = ("viability-test/run_prereg_completion.py", "viability-test/report_prereg.py")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def sealed_paths():
    return sorted(p for p in git("ls-tree", "-r", "--name-only", "HEAD").splitlines()
                  if p.startswith(DIRS) or p in MODULES)


def main():
    head = git("rev-parse", "HEAD").strip()
    lines = []
    for p in sealed_paths():
        blob = subprocess.run(["git", "show", f"HEAD:{p}"], cwd=ROOT, capture_output=True,
                              check=True).stdout
        lines.append(f"{hashlib.sha256(blob).hexdigest()}  {p}")
    (ROOT / "seal" / "SEALED_FILES_2.sha256").write_text("\n".join(lines) + "\n")
    print(f"sealed {len(lines)} files at {head}")


if __name__ == "__main__":
    main()
