"""tests/test_ledger.py -- nothing slips through (prereg/LEDGER.md).

Every ledger row has a status; every item registered in prereg/PHFW1.md has a
ledger row; no agent-written paper prose is tracked in the public tree."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = (ROOT / "prereg" / "LEDGER.md").read_text(encoding="utf-8")
PHFW = (ROOT / "prereg" / "PHFW1.md").read_text(encoding="utf-8")


def _rows():
    return [ln for ln in LEDGER.splitlines() if re.match(r"\|\s*L\d\d\s*\|", ln)]


def test_every_ledger_row_has_a_nonempty_status():
    rows = _rows()
    assert len(rows) >= 50
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        assert len(cells) >= 3 and cells[-1] and cells[-2], ln


def test_ids_are_unique():
    ids = [re.match(r"\|\s*(L\d\d)", ln).group(1) for ln in _rows()]
    assert len(ids) == len(set(ids))


def test_every_registered_phfw_item_has_a_ledger_row():
    ids = set(re.findall(r"\((L\d\d)\)", PHFW))
    assert len(ids) >= 14
    ledger_ids = {re.match(r"\|\s*(L\d\d)", ln).group(1) for ln in _rows()}
    assert ids <= ledger_ids, ids - ledger_ids


def test_every_phfw_seed_is_distinct_and_unused_elsewhere():
    seeds = re.findall(r"\|\s*(\d{9,10})\s*\|", PHFW)
    assert len(seeds) == len(set(seeds)) == 8
    seal = (ROOT / "seal" / "SEAL.md").read_text(encoding="utf-8")
    assert not (set(seeds) & set(re.findall(r"\b\d{9,10}\b", seal)))


def test_no_agent_paper_prose_is_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return
    bad = [p for p in out.stdout.splitlines()
           if re.search(r"(paragraph|manuscript|draft)", Path(p).name, re.I)
           and not p.startswith(("docs/archive/", "secsy/", "seal/"))]
    assert not bad, f"possible agent-written paper prose tracked: {bad}"
