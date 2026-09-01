# PPU0010 measured-policy K-pack runtime bundle

This artifact branch is based directly on clean source commit
`8f9fa07de9694901a5db91d546d6c994720f86b1`. The `bundle/` directory contains
exactly one manifest and the six canonical runtime libraries (`default`,
`fmt0` through `fmt4`). It includes the host-visible selected-config ABI and
the measured dense K-pack policy from that source commit.

Bundle manifest SHA-256:

```text
b3eb070bc65f42d5443626aa82baac468863657d5f479021caccba2d36f75097
```

The build used PPU SDK `2.1.1-a5c56e` for `ppu0010`; the admitted SDK archive
SHA-256 is:

```text
63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd
```

The manifest records the exact compile definitions, submodule commits,
compiler identity, file sizes, and SHA-256 for every DSO. The bundled strict
verifier is SHA-256
`43266a59b0676f19a34740d46fecbbdb2fd1ab80d88ee3911765f4f2ca5a21e7`
(Git blob `be9006407cd37ac21a861cdb9fc658f597a5188d`). It is copied byte-for-byte
from the binary source commit.

Fetch, hydrate, and verify every payload, exported ABI, and embedded PPU image
before use:

```bash
BRANCH=artifacts/ppu0010/8f9fa07-runtime6-b3eb070bc65f
ARTIFACT=prebuilt/ppu0010/8f9fa07/runtime6-b3eb070bc65f

git fetch origin \
  "refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
git switch --detach "origin/${BRANCH}"
git lfs pull \
  --include="${ARTIFACT}/bundle/*.so" \
  --exclude=""

BUNDLE="${PWD}/${ARTIFACT}/bundle"
test "$(sha256sum "${BUNDLE}/manifest.json" | awk '{print $1}')" = \
  b3eb070bc65f42d5443626aa82baac468863657d5f479021caccba2d36f75097
python3 "${ARTIFACT}/verify_bundle.py" "${BUNDLE}" \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}"
```

Do not use the verifier's `--manifest-only` mode for admission. It omits the
exported-symbol and embedded-image checks required for this bundle. Setting
`QUACTLIZE_PPU_BUNDLE` only selects a directory and does not verify it.

The host-only selected-config oracle passed these boundaries on the exact
files: measured dense, unmeasured compiled default, explicit override,
stale-name fail-closed, and grouped compiled default. Its source SHA-256 is
`43d5c0fb1ce07020489ff9e85e0528116e5a5cc920aef54ce388330bed209eea`
(Git blob `a0d355d351a579a46959ad03cdce52fb22df15e7`). The complete loader-facing
host ABI suite also passed all 26 cases for Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K.

Consumers must pin these public headers from source commit
`8f9fa07de9694901a5db91d546d6c994720f86b1` together with the libraries:

- `quactlize/include/quactlize_ppu_config.h`
- `quactlize/include/quactlize_ppu_device.h`
- `quactlize/include/quactlize_ppu_packed.h`

Device admission is a separate step and must execute this prebuilt bundle on
one PPU. Use a clean develop checkout whose runtime source is unchanged from
the manifest and whose tracked runner has SHA-256
`01ee28df26d8cc6c5cfe66ec94b99a9198794f390fb8e032cdfff5014c53ac0f`:

```bash
CUDA_VISIBLE_DEVICES=0 \
  python3 tools/run_prebuilt_ppu_box_gate.py "${BUNDLE}" \
    --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}" \
    --output /workspace/quactlize-prebuilt-gate-result
```

The gate requires NumPy and official `gguf==0.19.0`. It compiles nothing on
the box and writes non-overwriting `bundle.json` and `result.json` evidence.
Until that exact command passes, this publication is host/ELF admitted but not
PPU device admitted.
