# Grouped decode comparison package

Experimental PPU-only binaries built from `fcda60e`. Device correctness and
performance admission are pending; this is not a production runtime bundle.

- Eleven ordinary grouped parents, including the measured Q4/Q5 incumbents
  and TM8, with runtime Split-K and ordered FP32 partial reduction.
- One five-format execution library containing the unchanged scalar GEMV
  entry and an explicitly selected SIMT pair-reader candidate.
- Sixteen isolated jobs, 260 cells; no compilation on the box.

`manifest.json` records content hashes, build authority, and static SIMT ISA
inspection. The pair reader uses fused FP16 affine arithmetic and is not
claimed bitwise equivalent to the scalar reader. Both require independent
device numerical checks. Existing runtime bundles and production selection
are unchanged.

See [the experiment guide](../../../../docs/KPACK_DECODE_SWEEP.md) for scope,
commands, timing boundaries, and failed-job resume behavior.
