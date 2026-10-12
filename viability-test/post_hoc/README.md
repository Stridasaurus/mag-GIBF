# post_hoc: code for work that was NOT pre-registered

Everything here was decided after the sealed results (`seal/SEAL.md`) were read.
It is exploratory. It must not be mixed into, or cited as, the pre-registered work.

Rules (SPEC §S.16):

1. **New modules only.** Do not edit any `viability-test/*.py` module that is in
   `seal/SEALED_FILES.sha256`; import it and build on it here.
2. **Outputs go to `results/post_hoc/<name>/`**, never into a sealed directory.
3. **Label and anchor.** Each run's `manifest.json` carries `"status": "post_hoc"`
   and `"sealed_commit": "<commit it started from>"` (`tests/test_seal.py` enforces it).
4. **New random draws need their own seed**, drawn in an isolated interpreter and
   recorded, distinct from every seed in `seal/SEAL.md`. Sealed arrays may be
   read, never written.
5. **State the intent before running** in a short dated note (what is being
   asked, why it was not pre-registered). It is a record of intent, not a
   pre-registration.
6. **Never re-adjudicate.** A post-hoc result cannot change a sealed verdict, and
   any text that cites one says "post hoc".
