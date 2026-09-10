# K-pack fusion candidate

PPU SDK 2.1.1, small JIT-only delivery. **Device admission pending.**

- `libquactlize_ppu_pack.so`: Q2/3/4/5/6_K plus Q8_0 K-pack2, optional paired raw gate/up producer.
- `dispatch/`: C++ selector and unchanged execution helper. No compiled GEMM pool; missing selected parents JIT into a trusted local cache. No online tuning/sweep.
- `manifest.json`: producer hashes; `dispatch/manifest.json`: selector, execution and JIT source contract.

Use `bash tools/run_kpack_fusion_box.sh --help`. The runner gates exact GPU
pair bytes, Q8 W8A16, changing-route graph replay and real selected MoE chains
before model timing. The old six-library bundle is still required for the
K-quant intake/capability ABI and explicit fallback; it is not rebuilt here.

Q8 has initial, not measured-optimal, recipes. Fused gate/up uses doubled N;
when no exact family exists, a single same-qtype/K/E N/2-family transfer is
marked predicted. Resources and recipe use real doubled N. This is not a
retuned/measurement-backed winner. The selector correction leaves the GEMM
JIT source contract and cached device modules unchanged.
Small MoE fusion supports <=32 routed rows and compatible closed SwiGLU
graphs. Larger contexts, incompatible activation and concurrent-stream
graphs keep their original path. Two-source pairing is opt-in through
`QUACTLIZE_KPACK_PAIR_WEIGHTS=1`, enabled by this runner. It requires equal
qtype/shape, no per-projection bias/scales, and no separate-weight LoRA.
Merged weights use llama's conversion-time `--fuse-gate-up-exps` ordering.

Runtime cache v2 records both real source spans, even if nonadjacent. Verified
offline bundle v3 is unchanged; old runtime cache v1 remains readable.
Background D2H/writes retain two bounded pinned slots and never add a CPU wait
to inference kernel launch. The writer is joined before resident weights die.

Model timings use the uploaded batched-bench PP/TG axes, NPL=1. Every process
excludes the first whole pass for each PP; cold and hot cache use separate
processes. Timing is not full-model accuracy admission. Optional Asys captures
a warmed request separately. Tensor-parallel K-pack is explicitly NOT_TESTED.
