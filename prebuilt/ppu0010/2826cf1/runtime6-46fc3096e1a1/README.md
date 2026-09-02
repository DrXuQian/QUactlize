# PPU0010 K-pack runtime bundle with loader-safe admission

This artifact branch is a direct child of clean source commit
`2826cf12451e02ca4590f7a44682b57d2098bfb9`. The `bundle/` directory contains
exactly one manifest and the six canonical runtime libraries: the Q4
ScaleFirst default plus fully-quantized FMT0 through FMT4.

The bundle adds the loader contracts needed by a resident K-pack artifact:

- canonical arrangement, prepare and recover C entry points;
- dense and grouped any-M admission for all five K-quant formats;
- null config as shape-selected and explicit config override support;
- expert-major low/high/unit slices, with a null high plane when
  `high_bits == 0`.

Exact authorities:

```text
source commit       2826cf12451e02ca4590f7a44682b57d2098bfb9
manifest SHA-256    46fc3096e1a14b712ad5d7a50de096d2a973ad5826aa3ffe6a6764d1fc12180d
verifier SHA-256    8b552c33a8b9b34e184f6410720ae7150d562b57f47c8e604719b82cb324ec47
verifier Git blob   22516983f4659e712fc04a0278e9e63bd1ef3b14
config oracle SHA   5bb69a089074d9c89541ebca7d6106688b7b4c3c56da5dcda387afd4b881ec44
config oracle blob  8774480440d67d31dca41175ba0667701989d381
SDK release         2.1.1-a5c56e
SDK archive SHA     63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd
```

Fetch, hydrate and verify every payload before loading it:

```bash
BRANCH=artifacts/ppu0010/2826cf1-runtime6-46fc3096e1a1
ARTIFACT=prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1

git fetch origin \
  "refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
git switch --detach "origin/${BRANCH}"
git lfs pull --include="${ARTIFACT}/bundle/*.so" --exclude=""

BUNDLE="${PWD}/${ARTIFACT}/bundle"
test "$(sha256sum "${BUNDLE}/manifest.json" | awk '{print $1}')" = \
  46fc3096e1a14b712ad5d7a50de096d2a973ad5826aa3ffe6a6764d1fc12180d
python3 "${ARTIFACT}/verify_bundle.py" "${BUNDLE}" \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}"
```

Do not use `--manifest-only` for admission. It omits the exported-symbol and
embedded-image checks. The host selected-config oracle passed exact measured
dense selection, unmeasured compiled default, explicit override, stale-name
fail-closed, grouped compiled default, every-format any-M admission, and
default-library rejection.

Device admission is a separate, compile-free step. Use source-compatible
runner commit `1d579bc`; its tracked runner has SHA-256
`09110ba66b0d455ed91d44c3b2c0c648c84c923dacd38acbdf0b060412bf8297`
(Git blob `56264dcc327ae5e20d2c5cd49e3f1592e92b929d`). The gate requires NumPy,
official `gguf==0.19.0`, exactly one visible PPU, and a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/run_prebuilt_ppu_box_gate.py "${BUNDLE}" \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}" \
  --q4-correctness-repeats 8192 \
  --output /workspace/quactlize-a01-device-result-2826cf1
```

Admission requires `status=PASS`, six libraries, all five dense and grouped
formats, empty-expert rows `[2,0,3,1]`, raw-bit-stable Q4 dense and grouped
outputs across all 8192 launches, planted numerical RED controls, zero device
builds, and zero host compilations. Until that result exists, this publication
is host/ELF admitted but not PPU-device admitted.
