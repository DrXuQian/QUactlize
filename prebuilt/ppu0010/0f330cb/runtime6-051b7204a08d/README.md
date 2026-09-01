# PPU0010 six-library runtime bundle

This artifact branch is based directly on source commit
`0f330cbed40cff57e88679081b5ed00676301471`. The verified bundle root is
`bundle/`; it contains exactly one manifest and the six canonical runtime
libraries (`default`, `fmt0` through `fmt4`).

Bundle manifest SHA-256:

```text
051b7204a08d490007aea11be8d13dbb588748d50c24cf133862afce08a0577e
```

The build used PPU SDK `2.1.1-a5c56e` for `ppu0010`:

```bash
PPU_SDK=/root/ppu-sdk/2.1.1 \
PPU_SDK_ARCHIVE=/root/ppu-sdk-cache/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz \
JOBS=2 PPU_BUNDLE_JOBS=6 \
  bash tools/build_ppu_runtime_bundle.sh \
  /root/autodl-tmp/quactlize-runtime-0f330cb/bundle
```

The bundled strict verifier is SHA-256
`8ef252f2e984a306c99e645ccc27de6d680938306dc17d82c8c3deb71daaa3ef`
(Git blob `cfb9b355722a8fcc5e9d9a335fce8c9492611eb6`). It is the identical verifier
from develop commits `5c61cf6356ea83cd58c8e8ce5055a4ff10137bb1` and
`574058f8c1d3b1411097d9af86ab0f8c424bd706`.

Fetch, hydrate, and verify every payload, exported ABI, and embedded PPU image
before use:

```bash
BRANCH=artifacts/ppu0010/0f330cb-runtime6-051b7204a08d
ARTIFACT=prebuilt/ppu0010/0f330cb/runtime6-051b7204a08d

git fetch origin \
  "refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
git switch --detach "origin/${BRANCH}"
git lfs pull \
  --include="${ARTIFACT}/bundle/*.so" \
  --exclude=""

BUNDLE="${PWD}/${ARTIFACT}/bundle"
test "$(sha256sum "${BUNDLE}/manifest.json" | awk '{print $1}')" = \
  051b7204a08d490007aea11be8d13dbb588748d50c24cf133862afce08a0577e
python3 "${ARTIFACT}/verify_bundle.py" "${BUNDLE}" \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}"
export QUACTLIZE_PPU_BUNDLE="${BUNDLE}"
```

Do not use the verifier's `--manifest-only` mode for admission. It omits the
exported-symbol and embedded-image checks required for this bundle.

The host-only loader ABI suite passed all 26 cases in
`tests/test_kpack_host_abi.py`. The test authority was
`574058f8c1d3b1411097d9af86ab0f8c424bd706`; that change only corrected the
test's K-quant domain, and used the same runtime-bundle verifier blob as the
build review. Running this host test locally required an Ubuntu 24 loader, a
preloaded floor `libstdc++`, and a task-local compatibility filename overlay
from the expected hggcrt12 name to the installed SDK 13 runtime. That overlay
is test infrastructure only: it is not part of this bundle and is not PPU
device-execution or device-admission evidence.

Consumers must pin the public C ABI headers from source commit
`0f330cbed40cff57e88679081b5ed00676301471` together with the libraries:

- `quactlize/include/quactlize_ppu_config.h`
- `quactlize/include/quactlize_ppu_device.h`
- `quactlize/include/quactlize_ppu_packed.h`

All three headers are included by both supported Python package manifests in
that source commit.

`QUACTLIZE_PPU_BUNDLE` only selects a library directory; setting it does not
automatically verify the manifest, payload hashes, exported ABI, or embedded
PPU image. Run the strict verifier first. Absence or mismatch is an error; do
not rebuild or choose a different format library implicitly.
