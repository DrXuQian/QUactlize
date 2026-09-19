# Shape-independent decode candidates

Source branch: `dev/tp2-fastpaths`. Caller branch in the owner's llama.cpp
fork: `dev/quactlize-tp2-fastpaths`. The published TP2 runtime and its selectors
are unchanged. This is a bounded candidate package, not a replacement runtime.
`prebuilt.json` pins the LFS artifact commit, both manifest hashes and the
matching caller source. There are no llama.cpp binaries in the artifact.

## Implemented

| Area | Reusable implementation | Default-selection boundary |
|---|---|---|
| Q8 reader | Explicit hoist/non-hoist, dynamic/fixed N/K; F16/BF16 templates | Old admitted recipes remain selected |
| Q5 reader | Unsigned/fold body, parameterized fixed N/K | Old N2048/K512 admission unchanged |
| Split-K reduction | Ordered float2 `[row,split,N]`, real output stride, S2/S4/S8 | Candidate only; scalar four-byte fallback retained |
| All-SIMT prepare | Router/identity maps independent of weight N/K, tokens1..8 | Old measured token/shape scope unchanged |
| Paired fusion | Shared graph matching, caller strides/logging, library binding use actual local dimensions | A selector miss retains the unfused chain |

Paired layout remains a separate internal G4/U4 artifact. Canonical K-pack
files, public structs and precision contracts do not change. The caller keeps
dense F32 storage/F16 compute and MoE F32 storage/BF16 compute. No clipping.

The E256/top8 router topology, alignment, arithmetic and graph-liveness checks
are structural limits, not model constants. They are not removed.

## Inventory and evidence

`plan.py` contains ten actual 122B TP2 local GEMV shapes, two fused-chain
comparisons, and five old 35B regression controls: 17 points, 117 SIMT candidates
and four exact TC incumbent controls. No unbounded config product is generated.
The current per-point incumbent is retained, including its TC Split-K reducer.

Each candidate first checks tokens1..8 against independent GGUF/typed arithmetic,
changing replay, invalid IDs, guards and BF16 range controls where applicable.
Vector reducers additionally check M3, row padding and four-byte-aligned public
outputs/partials. The paired points measure gate+up+SwiGLU, not projection alone.
The ordinary points measure complete GEMV calls including reduction.

Timing is M1 only, with rotating active weights >=2.25 times verified 64 MiB L2,
three cheap screen rounds, then six alternating rounds of fifteen samples for
the incumbent and two finalists. First-use/JIT is excluded. ACU is a separate
forced-cold diagnostic. This compares to frozen K-pack implementations, not a
fresh raw-GGUF performance baseline or whole-model TPOT. Do not claim either.

## Local build

Run in this source checkout. Use a new output path and the unchanged runtime
from artifact commit `006aa757f5b995ae4d8eac772cc2106782e14e90` as `BASE_RUNTIME`.
The build checks both control DSO hashes before compiling any candidates.

```bash
python3 dev/gemv_model/build.py --cohort tp2 \
  --sdk "$PPU_SDK" --baseline "$BASE_RUNTIME" --output "$GEMV_BUILD" --jobs 8
python3 dev/gemv_model/inspect_native.py "$GEMV_BUILD" --cohort tp2
python3 dev/gemv_model/package.py "$GEMV_BUILD" "$GEMV_PACKAGE" --cohort tp2

python3 dev/moe_prepare/build.py --platform ppu --cuda "$PPU_SDK" \
  --baseline-ref 4326381 --output "$PREPARE_BUILD"
```

`--points tp2-q5-down` is a compile subset for a minimal cycle; the receipt
retains the full declared inventory and lists which points were built.
This does not rebuild a large production bundle or the whole caller. Full
caller builds must still use llama.cpp's `.aoneci/scripts/build.sh`.

Host checks need `LLAMA_CI_DIR` pointing to the matching caller and
`LLAMA_HOST_LIB_DIR` to an existing CPU `libggml-base` build:

```bash
python3 -m pytest -q tests/test_tp2_fastpaths.py tests/test_moe_prepare_gate.py \
  tests/test_gate_up_integration.py tests/test_model_gemv.py \
  tests/test_model_gemv_integration.py
```

## Device-only remainder

After moving the verified package, use an idle single PPU and its matching SDK.
These commands do not install a runtime or change production routing:

The prebuilt runner fetches only the two pinned artifact directories through
Git LFS, verifies source/payload/runtime identity, and runs the prepare gates
before independent GEMV comparisons. It requires the pinned `third_party/actlize`
submodule but does not compile anything or use a llama/NCP build directory.

```bash
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
RESULT_ROOT=/sim/eec/shared/junfu.qx CUDA_VISIBLE_DEVICES=0 \
bash tools/run_tp2_fastpaths_box.sh
```

The default includes ACU for the incumbent and selected candidate at each
point. The printed `*.results.tgz` contains summaries, logs and raw-counter CSVs;
the larger `*.acurep` files remain under `results/sweep/` on the box. There is no
whole-model Asys capture in this component gate. A failed independent point
does not stop the remaining points. Use `RESUME_RUN=/exact/printed/run` with
the same source, device and SDK to retain passed GEMV points and profiles.
`VERIFY_ONLY=1` verifies the handoff without accessing a device;
`TP2_ARTIFACT_DIR` may name an already materialized checkout at the pinned
artifact commit. Prepare timing is K3072 M1/M8 and prepare-only, not GEMM or TPOT.

The artifact branch is `artifacts/tp2-fastpaths-v1`. Materialize its LFS files
from the pinned commit in a separate checkout. `GEMV_PACKAGE` is its
`prebuilt/ppu0010/tp2-fastpaths-v1` directory and `PREPARE_BUILD` is its
`prebuilt/ppu0010/tp2-prepare-v1` directory. Verify the manifest/binary digests
against `prebuilt.json` before executing them. Do not check the artifact branch
out over an active source/build checkout.

```bash
python3 dev/gemv_model/run.py --cohort tp2 --sdk "$PPU_SDK" \
  --bundle "$GEMV_PACKAGE" --output "$RESULTS" --l2-bytes 67108864 \
  --acu "$PPU_SDK/asight/bin/acu"
"$PREPARE_BUILD/bench" --shape-check
"$PREPARE_BUILD/bench" --router-edge-check
"$PREPARE_BUILD/bench" --case 1 3072 5 1 0 1 0
```

Use `--resume` with the same inputs to retain passing points; runtime failures
start the next point in a fresh process. The prepare shape gate covers64
precision/token/dimension cases and four changing replays per arm. It does not
measure GEMM. A PP2048/TG128 model ABBA and Asys replay follow only after the
winning scopes pass device checks and are promoted into the selector/package.

Local receipts and outstanding work: `RESUME.md`, `candidates.jsonl`, and
`docs/plan.md`. No device timing is inferred from compilation or static registers.
