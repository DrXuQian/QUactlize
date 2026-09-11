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

Small shapes have not yet met the no-regression requirement versus the tuned
raw reference. A `MEASURED` status is not a claim that the parity goal is closed.
