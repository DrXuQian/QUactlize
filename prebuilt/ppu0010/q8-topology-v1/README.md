# Q8 topology experiment

One small PPU library,26 vector reader bodies and three unchanged controls.
This is not a model runtime or a production selector update.

Run `tools/run_q8_topology_box.sh` from the development checkout. It verifies
this manifest and the immutable shipping execution library before running.
Only one idle PPU is used. No compilation or JIT occurs on box.

L2 capacity is queried through the scalar SDK attribute as well as the
device properties. If both omit it, pass independently verified bytes with
`L2_BYTES`; for the verified 64 MiB PPU, use `L2_BYTES=67108864`.
Positive query/receipt disagreements are rejected. The effective capacity
and raw queries are printed and recorded; failed children print their log tail.

The290 full-call cells cover M1 Q8 dense512x2048,2048x512 and2048x4096.
S1/S2/S4/S8 are explicit, and split timing includes the reducer. Cold weights
rotate over at least2.25 times the verified L2 size. Finals use six rounds of
fifteen samples; ACU contains shipping and the best candidate, with exact
kernel/reducer and launch-geometry checks. A passing experiment does not
mean the40%/60% MBU targets are met.

Reproduce the local build with `dev/gemv_simt/build_q8_topology.py --platform
ppu --sdk SDK --shipping IMMUTABLE_RUNTIME --output NEW_DIRECTORY`.
`resources.txt` is the SDK resource dump. The full ISA can be reconstructed
with `hgobjdump --dump-isa q8.so`; it is not necessary to transfer another
copy of it. CUDA development binaries are not included.

See `docs/MODEL_DECODE_MBU_20260916.md` for prior results, address derivation,
local numeric evidence and remaining model-helper fusion gaps.
