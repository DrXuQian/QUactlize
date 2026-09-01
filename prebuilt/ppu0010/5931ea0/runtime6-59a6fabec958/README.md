# PPU0010 six-library runtime bundle

This artifact commit is a direct child of source commit
`5931ea04e5bbc94cd769f83314f713316fed2cf8`. The verified bundle root is
`bundle/`; it contains exactly one manifest and the six canonical runtime
libraries (`default`, `fmt0` through `fmt4`).

Bundle manifest SHA-256:

```text
59a6fabec95826015798797c324f29b18b4c86459ee8e348d8b35599c24041cc
```

The build used PPU SDK `2.1.1-a5c56e` for `ppu0010`:

```bash
PPU_SDK=/root/ppu-sdk/2.1.1 \
PPU_SDK_ARCHIVE=/root/ppu-sdk-cache/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz \
JOBS=4 PPU_BUNDLE_JOBS=6 \
  bash tools/build_ppu_runtime_bundle.sh /tmp/quactlize-ppu-runtime-5931ea0
```

Verify every payload, exported ABI and embedded PPU image before use:

```bash
python3 quactlize/ppu_bundle.py \
  prebuilt/ppu0010/5931ea0/runtime6-59a6fabec958/bundle \
  --ppu-sdk "${PPU_SDK:?set PPU_SDK to the admitted SDK root}"
```

Consumers should point `QUACTLIZE_PPU_BUNDLE` at the verified `bundle/`
directory. Absence or manifest mismatch is an error; do not rebuild or choose a
different format library implicitly.
