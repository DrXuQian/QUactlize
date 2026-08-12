# Dense TileM=8 shipping census

`ppu_dense_configs.inc` now ships the complete ShortWide-compatible family
`8x128:8x32:s{2,3,4,6,8,12}`. This is six tactic rows, not one selected result. `l147_dense_shipping_tm8.cpp`
crosses those rows with all five registered k-quant formats and both metadata modes to audit the broad tactic-space
model. Final legality belongs to `l148_dense_shipping_tm8_compiled.cu`: it reads the same `DenseKernelTypes` used by
the production launcher and therefore sees the exact `GemmKernel::SharedStorageSize`.

`l148` is a local nvcc/stub **type-layout witness**, not a PPU execution result. The shipping backend instantiates the
same trait and repeats the exact size guard under its real compiler; the numbers below do not claim device correctness
or performance before that build/run.

## Result

The exhaustive product has **60 cells: 51 legal, 9 illegal**. Every exact rejection has one reason: compiled
`GemmKernel::SharedStorageSize` exceeds the 256 KiB block limit. The broad host model reports 52/8; the equal-looking
old count was not enough evidence because it omitted Q6 fully-quantized raw packed staging. Production does not use
that arithmetic as a final guard.

| Format | Mode | Tactic/Artifact TileK | Legal stages | Illegal stages |
|---|---|---:|---|---|
| Q2_K | scale-first | 128 | 2, 3, 4, 6, 8, 12 | — |
| Q2_K | fully quantized | 256 | 2, 3, 4, 6, 8 | 12 |
| Q3_K | scale-first | 256 | 2, 3, 4, 6, 8 | 12 |
| Q3_K | fully quantized | 256 | 2, 3, 4, 6, 8 | 12 |
| Q4_K | scale-first | 64 | 2, 3, 4, 6, 8, 12 | — |
| Q4_K | fully quantized | 256 | 2, 3, 4, 6, 8 | 12 |
| Q5_K | scale-first | 256 | 2, 3, 4, 6 | 8, 12 |
| Q5_K | fully quantized | 256 | 2, 3, 4, 6 | 8, 12 |
| Q6_K | scale-first | 128 | 2, 3, 4, 6, 8, 12 | — |
| Q6_K | fully quantized | 128 | 2, 3, 4, 6, 8 | 12 |

The family remains one shared inventory. `list_valid_*` exposes only the exact-type legal cells for the concrete
format/mode. `launch_dense_tactic` retains only non-smem kernel/producer exclusions before instantiation; the shared
compiled trait then applies the exact block limit. This matters at two boundaries: Q5 scale-first s8 is 262,160 B,
not 262,144 B, because an aligned zero-length packed member still costs 16 B; Q6 fully-quantized s12 is 301,056 B
because raw packed metadata staging is real storage.

## Default and resource meaning

An empty/null config selects `8x128:8x32:s3` for `M<8`; `M>=8` and stale explicit-name fallback retain the previous
`64x64:32x32:s3` default. Explicit compiled names are unchanged. Validity queries and all host/device launch ABIs
consume the same `default_config_for_m(m)` policy.

Logical TileM=8 still pays for **16 physical A rows** in every one of the 60 cells. The expected savings are the
smaller FP32 accumulator fragment, the m8 MMA, and the smaller epilogue—not half of A shared memory. Adding TM8 also
changes the shared workspace upper bound from a minimum TileM of 16 to 8; the workspace query derives that bound from
the same inventory.

Reproduce locally:

```sh
c++ -std=c++17 -Iquactlize/include \
  dev/fold_derivation/l147_dense_shipping_tm8.cpp -o /tmp/l147_dense_shipping_tm8
/tmp/l147_dense_shipping_tm8
python3 ci/check_dense_shipping_tm8.py
bash dev/fold_derivation/run_l148_dense_shipping_tm8_compiled.sh
```
