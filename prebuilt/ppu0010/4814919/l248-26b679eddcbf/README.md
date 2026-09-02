# PPU0010 L248 AIU-plain + UniversalCopy raw-bit probe

This development artifact is based directly on source commit
`481491936a61e54f3c340fb60fed4953ac26f4b9`. It is an experimental layout-3
device probe, not a product runtime library and not evidence of a device pass.

The payload contains one PPU kernel. Static image inspection proved four
AIU-plain 32-bit writes, sixteen `tsm.ld.b32x4` UniversalCopy reads, one
commit/wait/barrier sequence, and no swizzle load. The exact authorities are:

```text
archive SHA-256   8fed0187617bb0328f8a70325576186e94b78b294225f0cb6b8c157ba31497ea
manifest SHA-256  26b679eddcbf150899dd16c3680dd07b52998a22802d7b11e2660068d74af451
binary SHA-256    6f4575c80d22ca3ef6f120e48aa170ce3238994de796b7fa86a9c0e4ee348f9e
PPU ISA SHA-256   ae0c86e87b81b81318640a55255b8c341d4c05dca1cc5bf22733e8d2761a9fc7
```

Hydrate and unpack the LFS object before execution:

```bash
ARTIFACT=prebuilt/ppu0010/4814919/l248-26b679eddcbf
git lfs pull --include="${ARTIFACT}/*.tar.gz" --exclude=""

echo '8fed0187617bb0328f8a70325576186e94b78b294225f0cb6b8c157ba31497ea  '"${ARTIFACT}"'/l248-q4-n16k64-prebuilt-481491936a61.tar.gz' |
  sha256sum -c -
mkdir /workspace/a03-l248-prebuilt-481491936a61
tar -xzf "${ARTIFACT}/l248-q4-n16k64-prebuilt-481491936a61.tar.gz" \
  -C /workspace/a03-l248-prebuilt-481491936a61
```

Run the manifest-owned runner from the extracted payload. It verifies the
source objects, submodule trees, SDK release/runtime, binary, ISA, and payload
file set before requiring exactly one visible device. A successful device
admission ends with `L248_Q4_N16K64_PREBUILT_DEVICE PASS` and reports
`raw_bad=0 sentinel=0 launch=0/0/0/0 reds=1`.
