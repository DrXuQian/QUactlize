# Native SDK 2.1.1 L2 query audit

Scope: local static inspection of the supplied 2.1.1 SDK, following the
Q4 comparison's `DeviceProperties.l2CacheSize == 0` box failure. No hardware
capacity or kernel-performance measurement is claimed here.

## Findings

The native scalar L2 query is implemented:

1. `include/driver_types.h:963` defines `hggcDevAttrL2CacheSize = 38`.
   `include/hggc.h:573` calls the matching driver attribute
   `HG_DEVICE_ATTRIBUTE_LLC_CACHE_SIZE = 38`.
2. In `libhggcrt.13.0.so`, `hggcDeviceGetAttribute` calls
   `hggcapiDeviceGetAttribute` at `0x34460`. Its call at `0x344d4` goes
   through the `hgDeviceGetAttribute` relocation, preserving the attribute
   number and returning the driver result.
3. In `libhggc.so`, `hgapiDeviceGetAttribute` calls
   `_HGdevice::getAttribute(int*, HGdevice_attribute_enum)` at `0x101a00`.
   The jump table at `0x1c98b0`, entry 38 at `0x1c9948`, has signed displacement
   `0xfff383c8`, selecting `0x101c78`:

   ```asm
   mov 0x540(%rdi), %eax
   mov %eax, (%rsi)
   jmp 0x101a5e          # eax=0; return success
   ```

   It reads a device-information field, rather than returning a hardcoded zero
   or “not supported”.
4. `_HGdevice::_initDeviceInfo()` at `0x102b20` obtains lower-level device
   information via a virtual call. At `0x102d5d` it copies the returned LLC-size
   field (input offset `0x308`) into device offset `0x540`. Actual capacity
   therefore comes from the board/driver environment; this local binary audit
   does not establish that it is 64 MiB.

The previous probe only consumed the properties structure. Its zero is real
evidence for that field on that run, **not evidence that all SDK L2 queries are
unimplemented**. Whether the scalar query returns a positive value on the box
still needs one device-side runtime query, not a kernel rebuild or a sweep.

The Python runner now tries attribute 38 when the properties field is zero.
It records unsupported/unwritten responses and does not suppress device or
unexpected runtime errors. Explicit `L2_BYTES` remains labelled an override;
an unverified value is never promoted to a measured device property. The
comparison kernels, launch recipes, canonical format, and timing method are
unchanged; all three prebuilt binary hashes remain unchanged.

## Inspected binary identities

| Library | SHA256 |
|---|---|
| libhggcrt.13.0.so | f4765821e374712a5d9a21cb7276101067ff7da56f9e3c55366ffe88c1997e9f |
| libhggc.so | 4acb6f71da458fbef346db163e5c04a1bdc341c8c560158412ec2c9618c1525a |
| libhggc_wrapper.so | 71c32cb41191458503234324360fcd3f1fa890dd5a082d465bb07328630c775e |

Addresses above apply only to these exact files. The wrapper exports the
query, but symbol presence alone was not used as evidence of implementation;
the runtime/driver bodies and attribute jump-table entry were inspected.
