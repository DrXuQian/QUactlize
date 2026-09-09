#!/usr/bin/env python3
"""Compile exact llama.cpp dot-product bodies, without the server or model."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REMOVAL = "c3ea58aca"


def between(source, begin, end):
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("llama.cpp source boundaries changed")
    first, last = source.index(begin), source.index(end)
    if first >= last:
        raise ValueError("reversed source boundaries")
    return source[first:last]


def source(llama):
    cuda = llama / "ggml/src/ggml-cuda"
    mmvq = (cuda / "mmvq.cu").read_text()
    quantize = (cuda / "quantize.cu").read_text()
    revision = subprocess.check_output(
        ["git", "-C", str(llama), "rev-parse", REMOVAL + "^"], text=True
    ).strip()
    historical = subprocess.check_output(
        ["git", "-C", str(llama), "show", revision + ":ggml/src/ggml-cuda/dmmv.cu"],
        text=True,
    )
    pieces = {
        "mmvq_lookup": between(
            mmvq,
            "typedef float (*vec_dot_q_cuda_t)",
            "static __host__ mmvq_parameter_table_id get_device_table_id(int cc)",
        ),
        "mmvq_topology": between(
            mmvq,
            "static constexpr __host__ __device__ int calc_nwarps(",
            "template <ggml_type type, int ncols_dst, bool has_fusion",
        ),
        "mmvq_dot": between(
            mmvq,
            "template <ggml_type type, int ncols_dst, bool has_fusion",
            "// Dedicated MoE multi-token kernel.",
        ),
        "quantize_q8_1": between(
            quantize,
            "__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)",
            "__device__ __forceinline__ uint8_t compute_e8m0_scale",
        ),
        "dmmv_q4_q5": between(
            historical,
            "static __global__ void dequantize_mul_mat_vec_q4_k(",
            "static __global__ void dequantize_mul_mat_vec_q6_k(",
        ),
    }
    # Only entry qualifiers change so a thin indexed wrapper can reuse the
    # original bodies. No GGUF decode, arithmetic or reduction is rewritten.
    historical_device = pieces["dmmv_q4_q5"].replace(
        "static __global__ void", "static __device__ __forceinline__ void"
    )
    text = '#include "mmvq.cuh"\n#include "quantize.cuh"\n#include "unary.cuh"\n#include "vecdotq.cuh"\n'
    text += "\n".join(
        pieces[k] for k in ("mmvq_lookup", "mmvq_topology", "mmvq_dot", "quantize_q8_1")
    )
    text += (
        "\n#define K_QUANTS_PER_ITERATION 2\nnamespace dmmv_legacy {\n"
        + historical_device
        + "\n}\n"
    )
    text += (HERE / "llama_reference.cuh").read_text()
    authority = dict(
        current_revision=subprocess.check_output(
            ["git", "-C", str(llama), "rev-parse", "HEAD"], text=True
        ).strip(),
        historical_revision=revision,
        fragments={
            name: hashlib.sha256(body.encode()).hexdigest()
            for name, body in pieces.items()
        },
        wrapper_sha256=hashlib.sha256(
            (HERE / "llama_reference.cuh").read_bytes()
        ).hexdigest(),
        historical_adapter="GPU_INDEXED_POINTER_REBASE_BODY_UNCHANGED",
        current_adapter="SINGLE_TOKEN_UNFUSED_GENERIC_SMALL_K_RULE",
    )
    return text, authority


def build(llama, cuda, output):
    text, authority = source(llama)
    output.mkdir(parents=True, exist_ok=False)
    tu = output / "llama_reference.cu"
    tu.write_text(text)
    nvcc = cuda / "bin/nvcc"
    version = subprocess.check_output([nvcc, "--version"], text=True)
    if "NVIDIA" not in version or "HGG" in version:
        raise ValueError("this diagnostic requires NVIDIA CUDA")
    includes = [
        llama / "ggml/include",
        llama / "ggml/src",
        llama / "ggml/src/ggml-cuda",
    ]
    hashes = {
        str(p.relative_to(llama)): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in includes[:2]
        for p in d.rglob("*")
        if p.is_file() and p.suffix in (".h", ".cuh")
    }
    command = [
        str(nvcc),
        "-std=c++17",
        "-arch=sm_120",
        "-O3",
        "-lineinfo",
        "-shared",
        "--cudart=shared",
        "-Xcompiler=-fPIC",
        "-Xlinker=-Bsymbolic",
        "-Xptxas=-v",
        *[f"-I{p}" for p in includes],
        str(tu),
        "-o",
        str(output / "libllama_reference.so"),
    ]
    with (output / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"reference build failed: {output/'build.log'}")
    authority.update(
        compiler=version,
        command=command,
        headers=hashes,
        library_sha256=hashlib.sha256(
            (output / "libllama_reference.so").read_bytes()
        ).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(authority, indent=2) + "\n")
    print("CUDA_LLAMA_REFERENCE_BUILD PASS output=" + str(output), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama", type=Path, required=True)
    p.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-12.8"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.llama.resolve(), a.cuda.resolve(), a.output.resolve())
