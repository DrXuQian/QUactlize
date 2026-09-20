# Paired Q4 MoE chain admission

Source baseline: 0aeffd7e977054bea05a369bc5c1a23e7380b588. Work branch:
`dev/tp2-paired-chain`, isolated from the shared develop checkout. Keep the
returned runtime46727e96 and current production policies immutable.

Priority: M1 TP2 local Q4 gate/up N512/K3072, Q5 down N3072/K512, E256/top8.
Regression: N512/K2048 gate/up and N2048/K512 down (35B). Both use BF16
activation/projection rounding and FP32 endpoints/accumulation. Test tokens1..8
for correctness; only M1 receives a new performance ranking.

Four arms per shape: selected merged projection plus production chain
SwiGLU; shipped paired implementation; historical paired fixed-S1 winner;
paired finish with the currently selected projection reader. All use the
same production router/snapshot prepare, Q5 down and weighted finish. The Q5
P4-fixed candidate from the preceding experiment is not mixed into this A/B.

Reuse production helpers and existing paired-N4 packing/SimtFinish. Ordinary
and paired layouts derive from identical GGUF gate/up matrices, explicitly
reordered before timing. Never reinterpret canonical data as paired data.
Changing IDs, graph replays, BF16-range values, output/weight aliasing, invalid
IDs and guards must pass an independent GGUF chain oracle. Packing, CPU
reference evaluation, allocations and H2D are outside the measured region.

Timing covers prepare -> projection -> optional activation -> down -> finish.
Use active-weight rings >=2.25x verified L2, complete traversals, excluded
first graph replay, six alternating rounds of15 samples and frozen existing
winners. No live model execution or production promotion in this experiment.
ACU follows confirmation and captures incumbent plus best paired chain; its
replay counters do not replace unprofiled complete-chain timings.

Derive address patterns with `dev/gemv_model/access.py` for every candidate:
paired G4/U4 ownership, per-warp low/metadata/A request footprints, native
load widths, register/shared usage and final reduction ownership. Static
compile/ISA checks are local; PPU correctness/performance needs the box.

Compile only two candidate TUs, reuse frozen execution/fusion DSOs. Do not
rebuild the TC catalog, change the caller, or add another selection layer.
Use physical cards6,7, one shape per card; all arm comparisons for a shape
stay on the same card. Preserve successful point/profile records on failure.
Publish the small prebuilt experiment through the existing LFS workflow.
Stop the local phase at a reproducible box command; do not claim a speedup
before device evidence returns.
