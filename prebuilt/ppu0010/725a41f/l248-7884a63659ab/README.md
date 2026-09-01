# Q4 N16xK64 raw-bit PPU handoff

This bundle contains the runnable layout-3 delivery probe built from source
`725a41fcea8f487af31d416d918fe94701cb9b0c`. The local image audit found one
kernel, four AIU-plain b32 loads, sixteen `tsm.ld.b32x4` loads, and the required
commit, wait, and CTA barrier.

Verify the complete manifest and select the binary before executing it:

```bash
BUNDLE=prebuilt/ppu0010/725a41f/l248-7884a63659ab
BIN="$(python3 tools/verify_prebuilt_ppu_bundle.py \
  "$BUNDLE/manifest.json" \
  --role raw-bit-probe --qtype 12 \
  --target test_q4_n16k64_delivery_rawbit \
  --expect-source 725a41fcea8f487af31d416d918fe94701cb9b0c \
  --expect-submodule third_party/actlize=8d46b758c8931807df840a6ed87d272d74a8fdf4 \
  --expect-submodule third_party/cutlass=f94ec46f4f63f96003d6cfdf2014731e7672c281 \
  --expect-arch ppu0010 \
  --expect-ppu-def PPU_PACKED_SCALE=1 \
  --expect-ppu-def PPU_PACKED_FORMAT=0 \
  --expect-ppu-def QUACTLIZE_DENSE_ONLY=12)" || exit 1
"$BIN"
if "$BIN" --plant-wrong-oracle; then
  echo 'wrong-oracle control unexpectedly passed' >&2
  exit 1
fi
```

The positive run is admitted only with `raw_bad=0`, `sentinel=0`, and four
zero launch-status fields. The wrong-oracle control must return nonzero and
report a positive `raw_bad` count.
