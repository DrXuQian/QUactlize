# H800 Q4 SIMT experiments

Development measurements, not PPU admission or production defaults.

The original FP32-dot Xplane and supplied raw-GGUF reader retain their
per-weight FP16 reconstruction. The `affine*` candidates instead apply FP32
scale/min after a group dot; they must not be labelled layout-only controls.
All arms consume FP16 A, accumulate/reduce/output in FP32, and compare against
the independent official-GGUF FP64 oracle. Canonical K-pack bytes are unchanged.

`large-confirmation.json` uses one frozen candidate/config for both cache
regimes and all four larger shapes, six alternating-order rounds, 15 event
samples per round. Each sample replays a full graph/cache ring, with first
launches excluded. There is no inter-CTA Split-K or separate reducer in the
K-pack candidate. Ref's third recipe field is **intra-CTA K warps**, not an
extra reduction launch. The raw reference was screened across 60 configurations
before freezing the confirmation shortlist.

The final accepted criterion is less than five percent regression versus
**both** controls. `small-confirmation.json`, `medium-confirmation.json`, and
`large-confirmation.json` cover six shapes and two cache regimes, six rounds
each. `final-verdict.json` reports 12/12 PASS: worst +1.88545% versus Xplane,
+4.40764% versus ref. This does not imply zero regression against ref.

`random-{small,medium,large}-{93711,93719}.json` validates the same frozen
candidates on two new FP16 activation seeds. `random-fixtures.json` binds
these fixtures to the unchanged source weights. All 24 shape/cache/seed
cells pass. These one-round runs are for additional correctness checks, not
candidate selection or replacement of the confirmed performance numbers.

`final-evidence.tgz` adds generated small/medium kernels and their build logs,
frozen selections, raw confirmation and random-validation logs. The earlier
large evidence archive contains the large candidate and common control
sources/build receipts. Neither archive includes compiled binaries.

Use `dev/gemv_cuda/summarize_h800_confirmation.py` to recompute the verdict
from the JSON receipts. A bare `MEASURED` status alone is never a parity pass.
