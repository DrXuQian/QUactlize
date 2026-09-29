# Frozen Q4 TP local fixture

`tp2_q4_fixture.tar.gz` contains six synthetic local shards and their receipt.
It contains no model data, libraries or executables. The box runner needs no
PyTorch, NumPy or llama build to consume it.

Archive SHA256:
`dd939d422f342f570cc663ab818a77efdd262c1bf7586624dbdd7810e370fa97`.
Each file's SHA256 and each field's FNV-1a64 are also in `fixture.json`.
The runner checks the full member inventory, shapes, sizes and hashes before
uploading anything to the device.

## Independent source and oracle

These are the deterministic inputs used by llama.cpp scheduler test
`--tp2-q4-local`, caller source `f9a0dbe2fb8b1e0dd50b0aee128617530c53a76f`.
Global weights have shape E4/N512/K1024. For flattened weight index `i`,
the F32 source is `((17*i + floor(i/29)) % 97 - 48) / 128`.
The actual llama host `ggml_quantize_chunk(Q4_K)` with unit importance weights
produces the raw GGUF blocks. Its `dequantize_row_q4_K` supplies the decoded
weights for the dot oracle, independently of the K-pack producer and reader.

For replay `r=0,1,2`, the activation has two K1024 rows, with flattened value
`((13*i + floor(i/37) + 5*r) % 61 - 30) / 128` and IDs `[r, (r+1)%4]`.
Rank0 takes K0..511; rank1 takes K512..1023. The local golden is the F64 sum
of decoded GGUF weights times local activations, rounded once to F32. It is
computed before and without using the packed planes. Every activation is
exactly representable in F16 and BF16; this fixture cannot conflate overflow
with a reader/packing defect.

The production host-only K-pack converter generates the low/unit planes.
Independent decoding of those planes recovers all 2,097,152 local weights
bit-exactly. Raw-shard and replay0 golden hashes reproduce the uploaded PPU
`kpack-tp2.Zu8e1e` results; this is not a new randomly selected passing case.
See `docs/KPACK_TP2.md` for the device evidence and its admission limits.

## File layout

All values are little-endian, without headers or padding:

| Field | Bytes | Meaning |
| --- | ---: | --- |
| raw | 589824 | Four local Q4_K experts, row-major GGUF blocks |
| low | 524288 | Canonical Q4 K-pack4 codes |
| units | 65536 | Canonical packed scale/min units |
| A | 4096 | Two F32 rows of K512 |
| IDs | 8 | Two I32 expert IDs |
| golden | 4096 | Two F32 output rows of N512 |

The diagnostic freezes generic SIMT variant0/columns4/warps4/values4/S1. It
tests host versus GPU production and F16 versus BF16 compute, with F32 storage.
A separate scalar reader uses the same canonical readers but no warp/shared
reduction; it is a localization control, not an independent numeric oracle.
Swapping expert IDs while leaving the golden unchanged must turn red.
