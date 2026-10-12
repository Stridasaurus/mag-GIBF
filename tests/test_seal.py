"""tests/test_seal.py — the pre-registered results are sealed (seal/SEAL.md).

Fails if a sealed file is changed, if the working copy of one differs from HEAD,
if a file is added to a sealed directory, or if post-hoc output is not labelled.
Uses git; skipped when the tests run outside a git checkout.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEAL = ROOT / "seal" / "SEALED_FILES.sha256"
SEALED_DIRS = ("results/A_flr_coherence", "results/B_viability",
               "results/V_kernel_validation")


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


pytestmark = pytest.mark.skipif(
    not (ROOT / ".git").exists() or _git("rev-parse", "HEAD").returncode != 0,
    reason="not a git checkout")


def _sealed():
    out = {}
    for line in SEAL.read_text().splitlines():
        if line.strip():
            digest, path = line.split("  ", 1)
            out[path] = digest
    return out


def test_seal_manifest_exists_and_is_complete():
    sealed = _sealed()
    assert len(sealed) >= 185
    assert any(p.startswith("results/B_viability/powered_d2/") for p in sealed)
    assert any(p.startswith("results/B_viability/b4_b6b/") for p in sealed)
    assert "viability-test/run_experiment_B2B6_powered_d2.py" in sealed
    assert "viability-test/run_experiment_B4_B6b.py" in sealed


def test_sealed_files_are_unchanged_at_head():
    bad = []
    for path, want in _sealed().items():
        r = _git("show", f"HEAD:{path}")
        if r.returncode != 0 or hashlib.sha256(r.stdout).hexdigest() != want:
            bad.append(path)
    assert not bad, f"sealed files changed or missing: {bad[:5]} (+{max(0, len(bad) - 5)})"


def test_working_copy_of_sealed_files_matches_head():
    paths = sorted(_sealed())
    r = _git("diff", "--name-only", "HEAD", "--", *paths)
    changed = [p for p in r.stdout.decode().splitlines() if p]
    assert not changed, f"uncommitted edits to sealed files: {changed[:5]}"


def test_no_unsealed_file_in_a_sealed_directory():
    sealed = set(_sealed())
    tracked = _git("ls-files", "--", *SEALED_DIRS).stdout.decode().splitlines()
    extra = [p for p in tracked if p not in sealed and not p.startswith("results/post_hoc/")]
    assert not extra, f"files added to sealed directories: {extra[:5]}"


def test_post_hoc_output_is_labelled():
    base = ROOT / "results" / "post_hoc"
    for manifest in base.glob("*/manifest.json"):
        m = json.loads(manifest.read_text())
        assert m.get("status") == "post_hoc", f"{manifest} is not labelled post_hoc"
        assert m.get("sealed_commit"), f"{manifest} does not name the sealed commit"
