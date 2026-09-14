# Typed decode device query

The first typed endpoint gate stopped at `loaded module device differs from
policy`, before numerical execution. The typed entry read SM count from
`hggcDeviceProp.multiProcessorCount`; the established entry and subsequent
typed resource query use `hggcDeviceGetAttribute` through
`cutlass::KernelHardwareInfo::query_device_multiprocessor_count`.

The typed entry now uses that same attribute query. It does not change kernel
arithmetic, layout, split factor, grid policy or the required 72-SM device.
Attribute errors still fail closed. A host regression makes the property value
1 and the attribute value 72; the real host entry returns 72, and rejects an
attribute-query failure. This is not PPU numerical admission.

The gate now probes one ordinary control and one typed module before creating
fixtures. It records name, ordinal, return code, reported SM count and explicit
attribute count. A common mismatch produces one infrastructure failure with
zero numerical cases started, rather than repeating the same failure 104 times.

For a read-only inspection of either old or repaired payloads:

```bash
python3 tools/probe_kpack_decode_device.py --sdk "$PPU_SDK" --bundle "$BUNDLE"
```

The repaired artifact has now passed the original 104 requests with both
F32/BF16 storage variants and the MoE graph controls. See the
[PPU result review](KPACK_DECODE_IO_RESULTS_20260914.md). The ordinary and
typed queries both report 72 CUs, agreeing with the explicit attribute.
This closes the identity failure for the tested artifact, not model speed.

The subsequent `undefined symbol: quactlize_ppu_kpack_canonical_arrangement_v1`
is a separate fixture dependency error. The decode runner incorrectly selected
the old single-source `kpack-pack-v1` producer for the MoE chain gate. It now
fetches the existing paired producer from `kpack-fusion-v1`, checks all four
required exports plus canonical/size queries up front, and records its path
and SHA256. No compute module or kernel body changes for this correction.
