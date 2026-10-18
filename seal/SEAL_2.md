# SEAL 2: pre-registered completion, 2026-10-18

A second, separate seal. The first (`SEAL.md`, `SEALED_FILES.sha256`, tag `sealed-2026-10-12`) is untouched.

## What is sealed

`SEALED_FILES_2.sha256` hashes, by committed git blob at the sealing commit, the files under `results/prereg_completion/` and `results/reports/` and the two modules that produced them (`viability-test/run_prereg_completion.py`, `viability-test/report_prereg.py`). `tests/test_seal.py` fails if any of them change or if an unsealed file appears in those two directories.

## What this is

The pre-registered work that the first seal missed: the 145 B2 grid cells and 12 |rho|-endpoint cells (master seed 2250821694, new cell ids), B1 and B3 at 10,000 trials (seed 2985373835), and the owed reporting (SPEC §S.17). It was specified before any code (`1232a26`, tag `prereg-completion-spec`), executed after the confirmatory adjudication, and **adjudicates nothing**. The run began at runner commit `c4b8eaa` (62 minutes, 14 workers). The next-phase draft (`prereg/PHFW1.md`) is parked and authorizes nothing.

Scorecard of the stated expectations: ROADMAP §8, 2026-10-18.
