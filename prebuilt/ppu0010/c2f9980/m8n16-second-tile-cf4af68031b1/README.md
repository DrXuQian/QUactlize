# PPU0010 m8n16 second-tile prebuilt probes

This development artifact is based directly on source commit
`c2f9980d52c6b32c955ed75b9bebba302f619aec`.  It contains the two already
compiled PPU0010 executables needed to classify the grouped M=8-boundary
failure.  The box runner performs verification and execution only; it never
configures or builds the repository.

The prerequisite executable proves the positive G0/G1/G2 device path.  It does
not include the separate PPU0015 compile-negative portion of gate 111, so this
bundle does not claim that part of the build-time proof.  The second executable
runs the focused G3/G4 probe over M=9/15/16/17 and distinguishes mainloop A
delivery from ptr-array epilogue placement in both nonpersistent and persistent
grouped kernels.

Authorities:

```text
source             c2f9980d52c6b32c955ed75b9bebba302f619aec
manifest SHA-256   cf4af68031b1829c1a52784ca2ea23c98deccab632d23e1a06439f9b324547da
gates SHA-256      465ddbac3b6d86af6b9eaa09e06bae1d8ea825d3ec086eeccd04494b3a789a34
collective SHA-256 9906e92fa2403d598f537db7625d0b34a64bae1ae13ccc0bbc70d5c78afa9d85
actlize            8d46b758c8931807df840a6ed87d272d74a8fdf4
cutlass            f94ec46f4f63f96003d6cfdf2014731e7672c281
SDK                 2.1.1-a5c56e
architecture        ppu0010 / PPU 1.0
```

The binaries were host-linked with `-Wl,--allow-shlib-undefined` because the
build host is older than the official Ubuntu 24.04 SDK runtime.  PPU code was
compiled by hgcc 2.1.1-a5c56e for PPU0010; the runner binds and live-inspects
the official SDK supplied on the box before either executable is launched.

Use a separate worktree so the current development checkout is untouched:

```bash
ART_BRANCH=artifacts/ppu0010/c2f9980-m8n16-second-tile-cf4af68031b1
WT=/workspace/quactlize-m8n16-c2f9980-prebuilt
BUNDLE=prebuilt/ppu0010/c2f9980/m8n16-second-tile-cf4af68031b1
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK
OUT=/workspace/quactlize-m8n16-second-tile-c2f9980-result

git fetch origin "$ART_BRANCH"
git worktree add --detach "$WT" FETCH_HEAD
git -C "$WT" lfs pull --include="$BUNDLE/bin/*" --exclude='' origin

CUDA_VISIBLE_DEVICES=0 \
  bash "$WT/$BUNDLE/run-prebuilt.sh" \
    --ppu-sdk "$PPU_SDK" \
    --output "$OUT"
```

Do not source `run-prebuilt.sh`; invoke it with `bash` as shown, so a failed
probe returns only from the child process and cannot close the interactive
shell.  A fully admitted run ends with:

```text
M8N16_SECOND_TILE_PREBUILT_DEVICE PASS ...
```
