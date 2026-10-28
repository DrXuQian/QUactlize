"""Historical format/path evidence retained for legacy schemes introspection.

The referenced harnesses live on experiments/ppu-tuning. This metadata is not
a product CI admission; current executable test ownership is ci/suites.py."""
GEMV_P, NATIVE_P, FP16_P, DENSE_P = "gemv", "fused_native_scale", "fused_fp16_scale", "dequant_then_dense"


SCALE_GEMV_P = "scale_first_gemv"


SCALE_GEMV_MOE_P = "scale_first_gemv_moe"


NATIVE_GEMV_P = "native_gemv"


NATIVE_GEMV_MOE_P = "native_gemv_moe"


SCALE_DENSE_P = "scale_first_dense"


SCALE_ARTIFACT_P = "scale_first_artifact"


SCALE_DENSE_ARTIFACT_P = "scale_first_dense_artifact"


HARNESS_PATHS = {
    "test_moe_grouped_verify": [FP16_P], "test_moe_grouped_real": [FP16_P], "test_lowbit_grouped": [FP16_P],
    "test_moe_grouped_streamk": [FP16_P],
    "test_q3_concat_real": [FP16_P], "test_q3_bconcat_real": [FP16_P], "test_q65_bconcat_real": [FP16_P],
    "test_w2a16_real": [FP16_P], "test_q4k_packed_gemm": [NATIVE_P, FP16_P], "test_q4k_native_scale": ["scale_decode"],
    "test_ppu_f16x2_probe": ["probe"], "test_w1a16_grouped": [FP16_P], "test_w2a16_grouped": [FP16_P],
    "test_w1a16_diag": [FP16_P], "test_w2a16_diag": [FP16_P], "test_w4a16_diag": [FP16_P],
    "test_fpA_kquant_dense": [SCALE_DENSE_P],
    "test_gemv_lowbit": [GEMV_P, SCALE_GEMV_P, SCALE_GEMV_MOE_P],
    "test_ppu_m8n16_collective": [FP16_P],
    # THE FIRST EVIDENCE FOR dequant_then_dense. Until this existed DENSE_P appeared in the path vocabulary and in
    # no harness at all, which is the honest state of a route whose host side was never wired -- and check_against_
    # formats() said nothing, because formats.DEQUANT_THEN_DENSE was empty too. Two empty sets agree.
    # NATIVE_P joined when the packed dense and grouped GEMMs gained oracles in this same file --
    # test_fully_quantized_dense/grouped_matches_dequant_first_and_rejects_fault, each comparing against
    # matmul_dequant_first through official gguf semantics, with a planted SCALE-UNIT fault required to fail
    # first. Declaring it is what lets nine newly VALIDATED cells rest on something rather than on nothing.
    "test_gguf_routes": [DENSE_P, SCALE_GEMV_P, SCALE_GEMV_MOE_P, NATIVE_GEMV_P, NATIVE_GEMV_MOE_P,
                         SCALE_ARTIFACT_P, SCALE_DENSE_ARTIFACT_P, NATIVE_P],
}


HARNESSES = {
    "test_moe_grouped_verify":  (["int4"],            "self",      None,
        "grouped vs the same kernel at L=1; plumbing only, blind to a wrong dequant constant"),
    "test_moe_grouped_real":    (["gptq-int4-sym"],   "real",      None,
        "real GPTQ weights through the grouped path; reads the fixture path from argv"),
    "test_lowbit_grouped":      (["int4", "int2", "int1"], "self", None,
        "L>1 grouped output vs per-expert L=1 runs of the same kernel; per-expert addressing only"),
    "test_moe_grouped_streamk": (["int4"],            "synthetic", None,
        "isolated phase-2 min2 grouped Stream-K: per-expert W/scale/zero/A and nonzero C against an independent CPU "
        "golden, plus global-q lock/fixup and ragged expert decoding; not a production dispatch route"),
    "test_q3_concat_real":      (["gguf-q3k"],        "real",      "real_q3k_concat.bin",
        "A-concat: two GEMMs summed, same golden as the B-concat but 2x the mma"),
    "test_q3_bconcat_real":     (["gguf-q3k"],        "real",      "real_q3k_concat.bin",
        "ONE GEMM with two B bit planes, vs the native Q3_K golden"),
    "test_q65_bconcat_real":    (["gguf-q5k", "gguf-q6k"], "synthetic", None,
        "SYNTHETIC despite the name -- full code range with a CPU golden, no importer exists for Q5_K/Q6_K"),
    "test_w2a16_real":          (["gguf-q2k"],        "real",      "real_q2k_ffn_gate_L0.bin",
        "int2 on a real 3B ffn_gate slice at gs=16"),
    "test_q4k_packed_gemm":     (["gguf-q4k"],        "real",      "q4k_packed.bin",
        "rowA/rowB run Q4_K on fp16 scale planes and rowC on the native 16 B unit, so it is evidence for BOTH paths; "
        "rowC is the only Scale_TileK==8 row and the only one that has been intermittent"),
    "test_q4k_native_scale":    (["gguf-q4k"],        "real",      "q4k_packed.bin",
        "device decode vs a host reference, no GEMM"),
    "test_ppu_f16x2_probe":     (["gguf-q4k"],        "synthetic", None,
        "the two f16x2 asm ops against the scalar ops they replace"),
    "test_w1a16_grouped":       (["int1"],            "synthetic", None, ""),
    "test_w2a16_grouped":       (["int2"],            "synthetic", None, ""),
    "test_w1a16_diag":          (["int1"],            "synthetic", None, "diagonal probe"),
    "test_w2a16_diag":          (["int2"],            "synthetic", None, "diagonal probe"),
    "test_w4a16_diag":          (["int4"],            "synthetic", None, "diagonal probe"),
    "test_fpA_intB_ppu":        ([],                  "none",      None,
        "superseded perf-only sweep; uninitialised synthetic weights and no output comparison"),
    "test_fpA_kquant_dense":    (["gguf-q2k", "gguf-q3k", "gguf-q4k", "gguf-q5k", "gguf-q6k"], "self", None,
        "two fpA dense configurations over each resident k-quant artifact; catches launcher plumbing but shares constants"),
    # Also declares gptq-int4-sym, and the reason is representational rather than a second harness: this sweep
    # instantiates a scale-only quant op with fp16 per-group scales over packed int4, which IS the GPTQ symmetric
    # representation -- same packing, same scale dtype, zero folded into the code range. What it does NOT cover is a
    # real GPTQ checkpoint reaching the GEMV, so the oracle stays synthetic.
    "test_gemv_lowbit":         (["int4", "int2", "int1", "gptq-int4-sym"], "synthetic", None, "CUDA-core GEMV"),
    "test_moe_gemm_ppu":        ([],                  "none",      None,
        "retired uniform-M timing probe; L>1 fails closed because the legacy vendor collective has unsafe "
        "sub-byte expert pitches; contributes no correctness evidence"),
    "test_moe_grouped_ppu":     ([],                  "none",      None,
        "superseded perf-only grouped sweep; no output comparison"),
    "test_ppu_m8n16_collective":(["int4"],            "synthetic", None,
        "ppu001 m8 collective gate: composed results at M=1/2/3/7/8/9/15/16/17 against an independent CPU "
        "golden and exact m16 control; M=9/15/16/17 additionally carry absolute-row A and ptr-array epilogue "
        "tags plus a persistent/nonpersistent control; also covers the E=256 non-contiguous ragged route with "
        "expert/row-offset negatives"),
    # Component probes only: these establish the ppu001 m8 atom ABI and the
    # AIU/x4-swzl delivery contract.  Empty format lists are deliberate; a
    # register/layout probe is not evidence for any quantized launch path.
    "test_ppu_m8n16_atom":      ([],                  "synthetic", None,
        "component probe: raw ppu001 m8n16 atom vs independent CPU golden; no quant format path"),
    "test_ppu_m8n16_aiu":       ([],                  "synthetic", None,
        "component probe: 16-row AIU x4-swzl delivery plus same-payload historical x2-index replay; no quant format path"),
    # SYNTHETIC, and the reason is a property of the oracle rather than a shortcut: the official gguf package has no
    # k-quant QUANTISER, only dequantize(), so there is no way to ask it for the bytes of a given weight. The bytes
    # are synthesised and the official dequantiser defines what they mean -- an independent implementation of the
    # spec over generated inputs, which is exactly this file's definition of synthetic. Real-checkpoint bytes reach
    # only Q4_K, through test_q4k_packed_gemm, and that is a different path.
    "test_gguf_routes":         (["gguf-q2k", "gguf-q3k", "gguf-q4k", "gguf-q5k", "gguf-q6k"], "synthetic", None,
        "official gguf oracle over raw-block dense fallback, production native/scale-first GEMV dense/MoE, and "
        "dequant-all/dequant-scale inverses for both derived scale-first formats; asymmetric n/k, ragged experts, "
        "and a path-specific planted fault before every positive comparison"),
}


FORMAT_TO_QUANT_TYPE = {
    "gguf-q2k": "Q2_K", "gguf-q3k": "Q3_K", "gguf-q4k": "Q4_K", "gguf-q5k": "Q5_K", "gguf-q6k": "Q6_K",
    "gptq-int4-sym": "GPTQ_INT4_SYM", "awq-int4": "AWQ_INT4",
}


def coverage_by_path():
    """(format, path) -> the strongest oracle any harness gives that pair."""
    rank = {"none": -1, "self": 0, "synthetic": 1, "real": 2}
    best = {}
    for name, (fmts, oracle, _fx, _n) in HARNESSES.items():
        for path in HARNESS_PATHS.get(name, []):
            for f in fmts:
                key = (f, path)
                if rank[oracle] > best.get(key, (None, -1))[1]:
                    best[key] = (f"{name} ({oracle})", rank[oracle])
    return {k: v[0] for k, v in best.items()}
