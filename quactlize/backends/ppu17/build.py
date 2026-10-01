"""Build an isolated PPU1.7 library; never changes the existing build graph."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from dependency import BACKEND, REVISION, require_cutlass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QTYPES = (8, 10, 11, 12, 13, 14)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log, *, env):
    with log.open("w") as f:
        f.write(json.dumps(command) + "\n")
        f.flush()
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def build(args):
    compiler = Path(args.compiler).resolve(strict=True)
    cutlass = Path(args.cutlass).resolve(strict=True)
    dependency = require_cutlass(cutlass)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == ROOT or ROOT in output.parents:
        raise ValueError("use a separate build root outside the checkout")
    shared = ROOT / "quactlize/include"
    common = [str(compiler), "-std=c++17", "-O3", "--expt-relaxed-constexpr",
              "--extended-lambda", "-gencode=arch=compute_90a,code=sm_90a",
              "-Xcompiler=-fPIC", "-lineinfo", "-I" + str(HERE),
              "-I" + str(shared), "-I" + str(cutlass / "include")]
    if args.target == "native":
        common += ["-DQ17_NATIVE=1", "-DACOMPUTE_VERSION=10700"]
    if args.enable_pdl:
        common += ["-DQ17_ENABLE_PDL=1", "-DCUTLASS_ENABLE_GDC_FOR_SM90=1"]
    identity = {"schema": 1, "backend": BACKEND, "target": args.target,
                "compiler": str(compiler), "compiler_sha256": sha(compiler),
                "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True),
                "cutlass_revision": REVISION, "cutlass": dependency, "flags": common[1:],
                "dependency_validator_sha256": sha(HERE / "dependency.py"),
                "source_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "pdl_compiled": args.enable_pdl,
                "device_admission": "PENDING", "timing_admission": "PENDING"}
    compiler_root = compiler.parent.parent
    identity["toolchain_files"] = {str(p): sha(p) for p in (
        compiler_root / "bin/ptxas", compiler_root / "bin/fatbinary", compiler_root / "nvvm/bin/cicc",
        compiler_root / "include/cuda_runtime.h", compiler_root / "include/cuda_runtime_api.h",
        compiler_root / "include/cuda_fp16.h", compiler_root / "include/cuda_bf16.h") if p.is_file()}
    # Includes only this backend and the read-only format owners. Legacy
    # actlize and the old dispatch tables cannot affect or enter this build.
    owners = [shared / p for p in ("quactlize_ppu_config.h", "ppu_format_config.inc", "ppu_format_config.hpp",
              "ppu_placed_arrangement.hpp", "q4_kpack4_offline.hpp", "q4_n16k64_direct_offline.hpp",
              "kquant_kpack_offline.hpp", "q8_kpack2.hpp", "gguf_scale_layout.hpp")]
    identity["sources"] = {str(p.relative_to(ROOT)): sha(p)
                           for p in sorted(list(HERE.glob("*.h*")) + list(HERE.glob("*.cuh")) +
                                           list(HERE.glob("*.cu")) + list(HERE.glob("*.cpp")) +
                                           [HERE / "exports.map"] + owners)}
    identity["builder_sha256"] = sha(__file__)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = output / key[:20]
    root.mkdir(exist_ok=True)
    existing = root / "manifest.json"
    if existing.is_file():
        receipt = json.loads(existing.read_text())
        library = root / "libquactlize_ppu17.so"
        if receipt.get("build_key") != key or not library.is_file() or sha(library) != receipt.get("library_sha256"):
            raise ValueError("existing bundle is damaged; preserve it and choose a fresh output root")
        print(f"PPU17_BUILD status=CACHE_HIT target={args.target} device_admission=PENDING bundle={root}", flush=True)
        return root
    temporary = root / "compiler-tmp"
    temporary.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(temporary)}
    started = time.monotonic()
    run(common + ["--ptx", str(HERE / "probe.cu"), "-o", str(root / "target.ptx")],
        root / "target.log", env=env)
    if ".target sm_90a" not in (root / "target.ptx").read_text():
        raise ValueError("compiler did not emit an SM90a target")

    def compile_one(name, source, definitions=()):
        obj = root / (name + ".o")
        command = common + list(definitions) + ["-c", str(source), "-o", str(obj)]
        run(command, root / (name + ".log"), env=env)
        print(f"PPU17_BUILD object={name} status=COMPILED elapsed_s={time.monotonic()-started:.1f}", flush=True)
        return obj

    tasks = [(f"q{q}", HERE / "module.cu", (f"-DQ17_QTYPE={q}",)) for q in QTYPES]
    tasks += [("host", HERE / "host.cpp", ()), ("runtime", HERE / "runtime.cu", ())]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        objects = list(pool.map(lambda task: compile_one(*task), tasks))
    library = root / "libquactlize_ppu17.so"
    run(common + ["-shared", "-Xlinker=--version-script=" + str(HERE / "exports.map")]
        + [str(p) for p in objects] + ["-o", str(library)], root / "link.log", env=env)
    if any(sha(ROOT / name) != digest for name, digest in identity["sources"].items()):
        raise ValueError("sources changed during compilation; this library is not published")
    require_cutlass(cutlass)
    if sha(HERE / "dependency.py") != identity["dependency_validator_sha256"]:
        raise ValueError("dependency validator changed during compilation")
    identity.update(library=library.name, library_sha256=sha(library), build_key=key,
                    build_seconds=time.monotonic()-started, compile_admission="PASS")
    (root / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(f"PPU17_BUILD status=COMPILED target={args.target} device_admission=PENDING bundle={root}", flush=True)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutlass", required=True)
    p.add_argument("--compiler", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target", choices=("cuda-source", "native"), default="cuda-source")
    p.add_argument("--enable-pdl", action="store_true", help="build opt-in PDL specializations; old entrypoints remain ordinary")
    p.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    a = p.parse_args()
    if a.jobs < 1:
        p.error("jobs must be positive")
    build(a)


if __name__ == "__main__":
    main()
