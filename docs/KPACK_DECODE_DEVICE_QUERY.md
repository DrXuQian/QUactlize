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

The repaired artifact must rerun the original 104 requests, both F32/BF16
storage variants and the MoE graph controls. Numerical and performance verdicts
remain pending until the new PPU results return.
