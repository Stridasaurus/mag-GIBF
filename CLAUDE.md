# CLAUDE.md — mag-GIBF

Guidance for Claude Code working in this repo. This file is about **how to work here
without breaking the project's disciplines**; the research *state* lives in the docs
below — read them, don't duplicate them here.

## Orientation — read in this order (do not skip)

1. **`README.md`** — repo purpose, environment, the 2026-07-01 terminology rename map
   (anything dated before then uses old names).
2. **`BUILD_BRIEF.md`** — the current onboarding brief: build state, the two invariants,
   the Gate-V/`transfer.py` adapter contract, the audit-A1 pins, what not to do.
3. **`ROADMAP.md`** — **canonical, live source of truth.** If any other doc disagrees
   with it, the roadmap wins. §0 rename map, §2 invariants, §5 decision tree, §8
   append-only experiment log, §9 current frontier.
4. **`handoff.md`** — **if a task is assigned, this is it.** It is written to be run cold
   and names its own gates and stop rule. As of 2026-07-12 it is the Experiment-B
   kickoff, triple-gated (PR #1 merged, ratification §8 entry appended, assignee named).
5. Experiment designs when you need that depth: `EXPERIMENT_CARD_A.md`,
   `viability-test/SPEC_experiment_B.md`.

## Environment & verification

```bash
conda activate mhd-env          # python 3.11; if absent, see README
pip install -e ./secsy          # secsy submodule is load-bearing, editable-installed
python -m pytest -q             # the gate — expect all green (22 passed as of 2026-07-12)
python viability-test/gateV_kernel_validation.py   # -> GATE V: PASS
```

`secsy` is a git submodule and is load-bearing (SECS field kernels). Clone with
`--recurse-submodules`; if tests fail from a cold clone, `git submodule update --init
--recursive` first. `pytest.ini` scopes the run to `tests/` so it does not recurse into
the submodule's own scripts. Never report tests green without running them in-session.

## The two invariants — never violate (ROADMAP §2)

1. **`A` is real. DF-only.** The SECS transfer matrix has no propagation phase — no
   `exp(ikr)` — because the ionosphere-ground system is quasi-static at these
   frequencies. `assert np.isrealobj(A)` everywhere except the deliberate B4
   phantom-phase ablation. Never include curl-free (CF) columns in a ground inversion:
   CF is excluded by *theorem* (Fukushima/Amm, Gate V), not by threshold. Get `A` from
   `viability-test/transfer.py`; never rebuild it ad hoc.
2. **Build the CSM from complex frequency-bin phasor snapshots, never raw real time
   samples.** Real `X` ⇒ real `S` ⇒ real eigenvectors ⇒ the premise collapses. Carry
   the eigenvalue magnitude: eigenmode `vᵢ = √λᵢ·uᵢ`, never the unit eigenvector.

## Working disciplines — how this repo stays trustworthy

- **Pre-registration is the whole method.** Every threshold and decision rule is pinned
  in `ROADMAP.md` §8 *before* results exist. **Never tune a threshold after seeing a
  number.** If a pinned value looks locally suboptimal once you are in the code, that is
  not a reason to change it — pre-registration only works if it precedes the result.
- **`ROADMAP.md` §8 is append-only.** Corrections are new dated entries referencing the
  old one — never inline edits to a past entry.
- **Commit the runner before its first real run** (the "A3 workflow"). A manifest that
  cites a commit must contain the exact script that produced it; the manifest records a
  `script_sha256`. This is how results stay reproducible and audit-able.
- **Statistics that compare across data of different physical units must be
  trace-normalized first** (× 2/trace) — the Card A Tier 2 gap-conditioning finding
  (§8, 2026-07-12). A raw cross-unit eigenvalue-gap comparison is scale-degenerate.
- **If a genuine ambiguity blocks you** (a real gap in the pinned design, not an
  implementation choice you could reasonably make either way): stop and ask Strider —
  do not improvise a pre-registered value. That is the exact semantic-drift failure the
  2026-07-07 repository audit found and fixed.

## Collaboration

Card A Tier 2 was executed by **Shane Gilbertie** on a branch+PR (`tier2-flr-coherence`).
Work assigned via `handoff.md` follows the same branch+PR pattern; Strider reviews the
PR as the ratification checkpoint. Do not merge your own PR.
