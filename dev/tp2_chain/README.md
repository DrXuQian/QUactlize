# Paired gate/up + SwiGLU: matched complete-chain experiment

The experiment below is complete and reviewed in
[result-20260920.md](docs/result-20260920.md). For the selective integration,
current model/Asys command and resume protocol, use
[production-handoff.md](docs/production-handoff.md).

This is a development admission gate, not a production selector change.
The four arms use the same GGUF weights, input, router, BF16 boundaries,
prepare, Q5 down recipe and weighted finish. Only gate/up + activation differs.

| Point | Logical gate/up N × K | Down N × K | Device assignment |
| --- | --- | --- | --- |
| 122B TP2 local shard | 512 × 3072 | 3072 × 512 | 6 |
| 35B regression control | 512 × 2048 | 2048 × 512 | 7 |

Both are E=256, top-k=8. Each point uses one device; this experiment does not
include the TP2 collective or measure model TPOT. All A/B comparisons stay on
the same physical device.

## Arms

- `unfused`: current merged Q4 projection + production BF16 SwiGLU.
- `paired-shipping`: the existing paired library, retained as a control.
- `paired-0`: historical paired C4/W8/P8/S1, fixed N/K.
- `paired-1`: paired finish with the current projection's C4/W4/P4/S1 reader.

The paired artifact is existing `PAIRED_N4` (G4/U4), not a new format. FP32
accumulation, BF16 projection rounding before SwiGLU, BF16 down input, and
ordered FP32 weighted finish are retained. No FP16 clamp or activation
quantization. The newly measured Q5 P4-fixed candidate is deliberately excluded
so it cannot contaminate the gate/up fusion comparison.

## Run on box

Use the source branch `dev/tp2-paired-chain`. No model, caller build, legacy six
libraries, JIT, or compiler job is needed. The pinned Git/LFS package contains
two tiny candidate libraries plus the exact frozen execution/fusion controls.

```bash
(
    set -e
    cd /sim/eec/shared/junfu.qx/quactlize
    git fetch origin dev/tp2-paired-chain
    GIT_LFS_SKIP_SMUDGE=1 git switch dev/tp2-paired-chain
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin dev/tp2-paired-chain
    git submodule update --init third_party/actlize

    PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
    RESULT_ROOT=/sim/eec/shared/junfu.qx \
    PAIRED_CHAIN_DEVICES=6,7 \
    bash tools/run_paired_chain_box.sh
)
```

If the SDK is installed elsewhere, only change `PPU_SDK`; it must name the
native root containing `lib/libhggc_wrapper.so` and `asight/bin/acu`, not the
nested `CUDA_SDK`. Keep both devices idle for the entire run. NumPy and the
existing `gguf` Python package are required. No dependencies are installed by
the script. Runtime library hashes are recorded alongside build hashes; device
correctness is checked even when they match.

M=1…8 numerical controls precede timing: changing IDs, graph replay, router on
and off, aliased routing weights, invalid IDs, canaries, and a BF16-range input
above 65504. The reference independently decodes GGUF and models BF16 rounding
boundaries. It is not a CPU performance arm or full model accuracy proof.

Timing is M=1 only: six alternating rounds, fifteen samples, full ring
traversals with at least 32 complete chains per graph replay. First replay is
excluded. Active gate/up + down weight copies exceed 2.25× verified L2; default
64 MiB is the capacity supplied for ZW810. `L2_BYTES` overrides that receipt.
The five-stage unfused call competes with the four-stage paired call; no
sum-of-isolated-kernel proxy is used. ACU runs afterwards for unfused and the
best numerically valid paired arm, with raw exports and exact stage checks.

Progress prints during fixture creation, after each numerical arm, and every
confirmation round. Remaining time covers the confirmation phase only; it is
not a full-run guarantee. No large sweep or model reload is involved.

## Resume and return

The printed run directory is `.../paired-chain.XXXXXX`. To retry only an
incomplete point/profile under the same source, device and runtime identities:

```bash
PAIRED_CHAIN_RUN=/sim/eec/shared/junfu.qx/paired-chain.XXXXXX \
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
PAIRED_CHAIN_DEVICES=6,7 bash tools/run_paired_chain_box.sh
```

Failed attempt logs are preserved, an independent second point continues,
and passing point timings are reused when only profiling failed. Upload the
printed `*.results.tgz`; it includes numerical records, complete-call samples,
access-pattern receipts, ACU reports and raw CSVs, but no model or binary.

## Local checks / next decision

`tests/test_tp2_paired_chain.py` checks source inventory, BF16 range and rounding,
the production pointer/phase wiring, full-stage ACU parsing, timing denominators,
and pinned fetch scope. The two candidate translation units compile locally in
about 8.5 seconds; native inspection checks the exact four kernels and vector
load / fast decode / FP32 FMA instructions. PPU numerical and performance
admission remains pending.

Choose per point only after the full-chain results and ACU agree. A regression
retains its own incumbent. No automatic promotion or blanket fused default.
