#!/usr/bin/env python3
"""Compile the small K-pack GEMV and metadata library, not a GEMM sweep."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha


def build(sdk, output, jobs, variant="production"):
    if variant not in ("production", "fp32-affine"):
        raise ValueError("unknown GEMV variant")
    sdk, output = sdk.resolve(), output.resolve()
    if jobs < 1:
        raise ValueError("jobs must be positive")
    for path in [sdk / "bin/hgcc", *[sdk / "lib" / f"lib{x}.so" for x in LIBRARIES]]:
        if not path.is_file():
            raise ValueError(f"missing SDK input: {path}")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "quactlize/execution"
    gemv_source = source / "gemv.cu"
    extra_sources = []
    if variant == "fp32-affine":
        # Reuse only the device-independent arithmetic/source transformation
        # that was tested on CUDA. No CUDA runtime/compiler compatibility
        # header enters this hgcc build, and production gemv.cu is unchanged.
        from dev.gemv_cuda.build import affine_source, grid_schedule

        generated = grid_schedule(affine_source(gemv_source.read_text()))
        tag = "QKG_CONCAT(kpack_q,QKG_QTYPE)"
        if generated.count(tag) != 3:
            raise ValueError("GEMV namespace seams changed")
        generated = generated.replace(tag, "QKG_CONCAT(kpack_affine_q,QKG_QTYPE)")
        gemv_source = output / "gemv_affine.cu"
        gemv_source.write_text(generated)
        extra_sources = [ROOT / "dev/gemv_fq_sf/adapters.cu"]
    includes = [
        source,
        ROOT / "quactlize/include",
        ROOT / "third_party/actlize/include",
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    inputs = sorted(
        {
            p
            for d in includes
            for p in d.rglob("*")
            if p.is_file() and p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh", ".inc")
        }
    )
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    if variant == "fp32-affine":
        for p in [
            ROOT / "dev/gemv_cuda/build.py",
            ROOT / "dev/gemv_cuda/affine_dot.cuh",
            *extra_sources,
        ]:
            hashes[str(p.relative_to(ROOT))] = sha(p)
    commands = []
    for name, filename, defines in (
        [(f"q{q}", gemv_source, [f"-DQKG_QTYPE={q}"]) for q in range(10, 15)]
        + [
            ("metadata", source / "metadata.cu", []),
            ("dispatch", source / "dispatch.cpp", []),
        ]
        + [("adapters", p, []) for p in extra_sources]
    ):
        command = [
            str(sdk / "bin/hgcc"),
            *FLAGS,
            *defines,
            *[f"-I{d}" for d in includes],
            "-c",
            str(filename),
            "-o",
            str(output / f"{name}.o"),
        ]
        commands.append((name, command))

    def run(item):
        name, command = item
        print(f"KPACK_EXECUTION_BUILD phase={name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        if result.returncode:
            raise RuntimeError(
                f"{name} failed rc={result.returncode}; log={output / f'{name}.log'}"
            )

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(jobs, len(commands))) as executor:
        list(executor.map(run, commands))
    library = output / "libquactlize_ppu_execution.so"
    run(
        (
            "link",
            [
                "g++",
                "-shared",
                "-Wl,-Bsymbolic",
                *(c[-1] for _, c in commands),
                "-o",
                str(library),
                f"-L{sdk / 'lib'}",
                *[f"-l{x}" for x in LIBRARIES],
            ],
        )
    )
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("execution inputs changed during compilation")
    manifest = dict(
        schema="quactlize.kpack-execution-build.v1",
        source_hashes=hashes,
        compiler_sha256=sha(sdk / "bin/hgcc"),
        flags=FLAGS,
        runtime={f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES},
        library=library.name,
        sha256=sha(library),
        compile_seconds=time.monotonic() - start,
        formats=list(range(10, 15)),
        gemv_configs=[
            dict(columns=c, warps=w, split=s)
            for c in (16, 32)
            for w in (4, 8)
            for s in (1, 4)
        ],
        device_validated=False,
        gemv_pair_configs=[
            dict(columns=c, warps=w, split=s)
            for c in (16, 32)
            for w in (2, 4, 8)
            for s in (1, 2, 4, 8)
        ],
        gemv_pair_affine="FP16_FMA_ONE_ROUNDING",
        variant=variant,
        generated_source_sha256=sha(gemv_source),
        comparison_adapters=bool(extra_sources),
    )
    if variant == "fp32-affine":
        manifest["gemv_pair_affine"] = "FP32_GROUP_AFFINE_NO_INTERMEDIATE_FP16_ROUNDING"
        manifest["grid"] = "GRID_X_N_TILE_Y_SPLIT_Z_ROW_ROWS_LE65535"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"KPACK_EXECUTION_BUILD status=COMPILED device_validated=0 seconds={manifest['compile_seconds']:.3f} library={library}"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--variant", choices=("production", "fp32-affine"), default="production"
    )
    args = parser.parse_args()
    build(args.sdk, args.output, args.jobs, args.variant)
