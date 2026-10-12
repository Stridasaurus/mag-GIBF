# SEAL: pre-registered Experiment B results, 2026-10-12

This directory records that the results listed in `SEALED_FILES.sha256` are final
and must not change. It is a statement of provenance, not an interpretation.

## What is sealed

185 files, hashed by their committed content (git blobs at the sealing base
commit `cc9be447af40153b2e0e3b16e69a201866501fe5`, so the manifest is the same on
every operating system):

- **`results/`, everything except `results/post_hoc/`**: Gate V, Card A Tier 1
  and Tier 2, the B1/B3/mini-pilot and d=3/d=8/d=2 pilot records, the peak-rule
  rescoring, the **confirmatory powered run** (`results/B_viability/powered_d2/`,
  81 files) and the **B4/B6b run** (`results/B_viability/b4_b6b/`, 60 files).
- **`viability-test/*.py`**: the 16 top-level modules that produced those results.

Not sealed, because they are living documents that legitimately grow:
`ROADMAP.md` (append-only), `SPEC_experiment_B.md`, `README.md`, `BUILD_BRIEF.md`,
`tests/`.

## The two runs

| | Confirmatory powered B2/B6a at d=2 | B4 phantom phase and B6b jitter |
|---|---|---|
| Spec committed before code | SPEC §S.14 (ratified 2026-09-29); off-grid row SPEC §S.14, `a7669d6` | SPEC §S.15, `fc36bb9` |
| Runner commit | `b235a16` | `485614b` |
| Runner blob | `e126fce761cb8e2057e2fea5110bbdef1b9a22a7` | `882f9015e267942821bf50566fbc33a304b74f27` |
| Seed | 2250821694 | 2411409774 |
| Seed drawn | 2026-09-23T07:38:11Z, isolated interpreter | 2026-10-12T13:11:47Z, isolated interpreter |
| Trials per cell | 10,000 | 10,000 |
| Cells | 39 | 29 |
| Wall time | 331 min, one process | 258 min, 14 workers |
| Exit | 0, empty stderr | 0, empty stderr |
| Integrity checks | 15/15 | 13/13 |
| Adjudication | applied once, as a separate step (`adjudication.json`) | none; descriptive |

`secsy` is pinned at `6f699cbdd4a0ebec8e9f6a97315ab80ede34a439`.

A first launch of the B4/B6b run was stopped before any cell completed; nothing
was written or read (ROADMAP §8, 2026-10-12). The confirmatory run had one false
start in a wrapper script before Python started; it consumed no seed. Wrapper
logs for both completed runs are in `run_logs/`.

## Reproducibility check

`verify_reproducibility.py` regenerates the first 40 trials of nine cells from
the pinned seeds and compares them with the saved arrays: five confirmatory cells
(win, null, d=1, B6a, off-grid) and four B4/B6b cells. Result at sealing:
**3,087 arrays compared, 3,087 bit-identical.** It reads the sealed files and
writes nothing.

## How it is enforced

`tests/test_seal.py` fails if any sealed file differs from its recorded hash,
if the working copy of a sealed file differs from `HEAD`, or if a file has been
added to a sealed directory. It also requires every `results/post_hoc/*/manifest.json`
to be labelled `"status": "post_hoc"` and to name the sealed commit it started from.

## What may and may not be done

- Sealed files are **never edited**. A correction is a new dated note in
  `seal/ADDENDA.md`, which does not touch the sealed file.
- The sealed modules are **not changed**. Anything that needs different code goes
  in `viability-test/post_hoc/`, as a new module.
- Post-hoc work, any analysis or run decided after the sealed results were read,
  lives in `viability-test/post_hoc/` and `results/post_hoc/` and is labelled
  post hoc (SPEC §S.16). It never alters or re-adjudicates a sealed verdict.
- If a sealed file or module must change for a real reason, build a new, dated
  seal: do not edit this one.

## Verify it yourself

```bash
python -m pytest -q tests/test_seal.py        # hashes, working tree, no extra files
python seal/verify_reproducibility.py 40       # regenerate trials from the seeds
```

Tag `sealed-2026-10-12` marks the commit that adds this directory.
Tags `prereg-b4-b6b-spec`, `prereg-b4-b6b-run`, `prereg-offgrid-spec` and
`prereg-confirmatory-d2-run` mark the specification and run commits above.
