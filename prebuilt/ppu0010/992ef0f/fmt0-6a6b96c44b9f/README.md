# PPU0010 Q4_K FMT0 prebuilt

This bundle contains the `quactlize_ppu` loadable library built from source
`992ef0f28b061ad2110784252bf0406a7b91647a` for PPU0010. Its admitted build
definitions are:

```text
PPU_PACKED_SCALE=1
PPU_PACKED_FORMAT=0
QUACTLIZE_DENSE_ONLY=12
```

The carrier is an x86-64 shared object with embedded PPU 1.0 SIMD images;
`hgobjdump --list-elf` found the embedded kernels. The SDK is the Ubuntu 24.04
2.1.1-a5c56e package named in `manifest.json`. At runtime, make that SDK's
`lib` directory available through `LD_LIBRARY_PATH`. Its wrapper requires a
host with glibc 2.38 or newer and `GLIBCXX_3.4.32` or newer.

After cloning this artifact ref, download and verify the LFS payload before
loading it:

```bash
git lfs pull

BUNDLE=prebuilt/ppu0010/992ef0f/fmt0-6a6b96c44b9f
LIB="$(
  python3 "$BUNDLE/verify_prebuilt_ppu_bundle.py" \
    "$BUNDLE/manifest.json" \
    --role fully-quantized \
    --qtype 12 \
    --target quactlize_ppu \
    --expect-source 992ef0f28b061ad2110784252bf0406a7b91647a \
    --expect-submodule third_party/actlize=8d46b758c8931807df840a6ed87d272d74a8fdf4 \
    --expect-submodule third_party/cutlass=f94ec46f4f63f96003d6cfdf2014731e7672c281 \
    --expect-sdk 'PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e sha256=63ca196b152f2fec667fce8b18c04f1d6d0fa9e7bc7f72e18f017c96d11731dd' \
    --expect-compiler 'hgcc Release version 2.1.1-a5c56e built=2026-07-25T09:15:42 sha256=fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e' \
    --expect-arch ppu0010 \
    --expect-ppu-def PPU_PACKED_SCALE=1 \
    --expect-ppu-def PPU_PACKED_FORMAT=0 \
    --expect-ppu-def QUACTLIZE_DENSE_ONLY=12
)"

export QUACTLIZE_PPU_LIB_FMT0="$LIB"
```

The verifier checks every manifest payload and prints only the selected
absolute path on success. This artifact is a loadable device library, not a
standalone test executable; use it with the matching quactlize host extension
and runtime test or benchmark.
