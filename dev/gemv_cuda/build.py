#!/usr/bin/env python3
"""Development-only NVIDIA compile of the unchanged SIMT reader/kernel."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("production schedule seam changed: " + before[:80])
    return source.replace(before, after)


def grid_schedule(source):
    source = replace_once(
        source,
        """    int const tiles = c.n / Columns;
    int const tile = int(blockIdx.x) % tiles;
    int const outer = int(blockIdx.x) / tiles;
    int const partition = outer % split, row = outer / split;""",
        """    int const tile = int(blockIdx.x);
    int const partition = int(blockIdx.y), row = int(blockIdx.z);""",
    )
    source = replace_once(
        source,
        """    unsigned const grid = unsigned(uint64_t(c.rows) * split * (c.n / Columns));""",
        """    if (c.rows > 65535) return QKG_INVALID;
    dim3 const grid(c.n / Columns, split, c.rows);""",
    )
    source = replace_once(
        source,
        """    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < int64_t(c.rows) * c.n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t const row = i / c.n, col = i % c.n;""",
        """    int64_t const row = blockIdx.y;
    int const col = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < c.n) {""",
    )
    return replace_once(
        source,
        """        uint64_t const count = (uint64_t(c.rows)*c.n+255)/256;
        kpack_gemv_reduce<<<unsigned(count < 65535 ? count : 65535),256,0,stream>>>(c,split);""",
        """        dim3 const reduce_grid((c.n + 127) / 128, c.rows);
        kpack_gemv_reduce<<<reduce_grid,128,0,stream>>>(c,split);""",
    )


def affine_source(original):
    start = "template<class Reader>\n__device__ __forceinline__ float pair_dot("
    end = "template<int Columns, int Warps, bool Pair = false>\n__global__ void kpack_gemv("
    if original.count(start) != 1 or original.count(end) != 1:
        raise ValueError("production dot boundaries changed")
    first, last = original.index(start), original.index(end)
    if first >= last:
        raise ValueError("reversed dot boundaries")
    return (
        original[:first]
        + (HERE / "affine_dot.cuh").read_text()
        + "\n"
        + original[last:]
    )


def build(cuda, output, jobs, reader="production", schedule="production"):
    if reader == "cuda-half2" and schedule != "production":
        raise ValueError("the half2 include wrapper requires the production schedule")
    nvcc = cuda / "bin/nvcc"
    version = subprocess.check_output([nvcc, "--version"], text=True)
    if "NVIDIA" not in version or "HGG" in version:
        raise ValueError("requires a complete NVIDIA CUDA SDK")
    output.mkdir(parents=True, exist_ok=False)
    commands = {}

    def run(name, command):
        commands[name] = [str(x) for x in command]
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {output / f'{name}.log'}")
        print(f"CUDA_GEMV_BUILD phase={name} status=PASS", flush=True)

    started = time.monotonic()
    run(
        "probe-build",
        [nvcc, "-arch=sm_120", "-O2", HERE / "probe.cu", "-o", output / "probe"],
    )
    run("probe-run", [output / "probe"])
    includes = [
        HERE / "compat",
        ROOT / "quactlize/execution",
        ROOT / "quactlize/include",
        ROOT / "third_party/actlize/include",
        ROOT / "third_party/actlize/tools/util/include",
        ROOT / "third_party/actlize/examples/common",
    ]
    flags = [
        "-std=c++17",
        "-arch=sm_120",
        "-O3",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "-Xptxas=-v",
        "-DCUTLASS_USE_PACKED_TUPLE=1",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-include",
        str(HERE / "compat/compiler_bridge.h"),
        *[f"-I{x}" for x in includes],
    ]
    sources = sorted(
        {
            p
            for d in includes + [HERE, ROOT / "quactlize/runtime"]
            for p in d.rglob("*")
            if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".cu", ".cpp", ".inc")
        }
    )
    sources += [ROOT / "tests/kpack_grouped_postops_layout.hpp"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    gemv_source = (
        ROOT / "quactlize/execution/gemv.cu"
        if reader == "production"
        else HERE / "gemv_half2.cu"
    )
    original = gemv_source.read_text()
    if reader == "cuda-affine":
        original = affine_source((ROOT / "quactlize/execution/gemv.cu").read_text())
    if schedule == "cuda-grid":
        original = grid_schedule(original)
    if reader == "cuda-affine" or schedule != "production":
        gemv_source = output / "gemv_experiment.cu"
        gemv_source.write_text(original)
    entries = [("q8", ROOT / "quactlize/execution/gemv.cu", ["-DQKG_QTYPE=8"])] + [
        (f"q{q}", gemv_source, [f"-DQKG_QTYPE={q}"]) for q in range(10, 15)] + [
        ("dispatch", ROOT / "quactlize/execution/dispatch.cpp", []),
        ("reducer", HERE / "reducer.cu", []),
        (
            "direct",
            HERE / "direct.cu",
            ["-DPPU_PACKED_SCALE=1", "-DPPU_PACKED_FORMAT=0"],
        ),
    ]

    def compile_one(item):
        name, source, defs = item
        run(
            name,
            [
                nvcc,
                *flags,
                *defs,
                "-x",
                "cu",
                "-c",
                source,
                "-o",
                output / f"{name}.o",
            ],
        )

    with ThreadPoolExecutor(max_workers=min(jobs, len(entries))) as pool:
        list(pool.map(compile_one, entries))
    library = output / "libkpack_gemv_cuda.so"
    run(
        "link",
        [
            nvcc,
            "-shared",
            "--cudart=shared",
            "-Xlinker=-Bsymbolic",
            *[output / f"{name}.o" for name, _, _ in entries],
            "-o",
            library,
        ],
    )
    if any(digest(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("source changed during compilation")
    manifest = dict(
        schema="quactlize.dev-gemv-cuda.v1",
        compiler=version,
        compiler_sha256=digest(nvcc),
        source_hashes=hashes,
        builder_sha256=digest(Path(__file__)),
        generated_source_sha256=digest(gemv_source),
        commands=commands,
        library=library.name,
        library_sha256=digest(library),
        seconds=time.monotonic() - started,
        production_kernel="quactlize/execution/gemv.cu",
        pair_affine={
            "production": "CUDA_SCALAR_FALLBACK_NOT_PPU_F16X2_ASM",
            "cuda-half2": "CUDA_HFMA2_EXPERIMENT",
            "cuda-affine": "FP32_GROUP_AFFINE_NO_METADATA_WEIGHT_OR_A_FP16_ROUNDING",
        }[reader],
        reader=reader,
        schedule=schedule,
        ppu_admission=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"CUDA_GEMV_BUILD status=COMPLETE seconds={manifest['seconds']:.3f} library={library}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--reader",
        choices=("production", "cuda-half2", "cuda-affine"),
        default="production",
    )
    parser.add_argument(
        "--schedule", choices=("production", "cuda-grid"), default="production"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(
        args.cuda.resolve(),
        args.output.resolve(),
        args.jobs,
        args.reader,
        args.schedule,
    )
