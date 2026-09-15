#!/usr/bin/env python3
"""Build isolated all-format SIMT candidates. CUDA is a development adapter only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt import spec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(a):
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sdk = a.sdk.resolve(strict=True)
    started = time.monotonic()
    includes = [ROOT, ROOT / "quactlize/execution", ROOT / "quactlize/include",
                ROOT / "third_party/actlize/include", ROOT / "third_party/actlize/tools/util/include"]
    env = dict(os.environ)
    if a.platform == "ppu":
        from quactlize.runtime.compiler import FLAGS, LIBRARIES
        compiler = sdk / "bin/hgcc"
        flags = list(FLAGS)
        linker = ["g++", "-shared", "-Wl,-Bsymbolic"]
        libs = [f"-L{sdk}/lib", *[f"-l{x}" for x in LIBRARIES]]
        env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    else:
        compiler = sdk / "bin/nvcc"
        version = subprocess.check_output([compiler, "--version"], text=True)
        if "NVIDIA" not in version or "HGG" in version:
            raise ValueError("CUDA adapter requires NVIDIA nvcc")
        includes.insert(0, ROOT / "dev/gemv_cuda/compat")
        flags = ["-std=c++17", "-O3", "-lineinfo", f"-arch={a.arch}",
                 "--expt-relaxed-constexpr", "-Xcompiler=-fPIC", "-Xptxas=-v",
                 "-DCUTLASS_USE_PACKED_TUPLE=1", "-DCUTE_USE_PACKED_TUPLE=1",
                 "-include", str(ROOT / "dev/gemv_cuda/compat/compiler_bridge.h")]
        linker = [str(compiler), "-shared", "--cudart=shared", "-Xlinker=-Bsymbolic"]
        libs = []
    sources = {str(p.relative_to(ROOT)): sha(p)
               for directory in includes[1:] if directory != ROOT
               for p in directory.rglob("*")
               if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".inc")}
    for directory in (ROOT / "quactlize/execution", Path(__file__).parent):
        for p in directory.glob("*"):
            if p.is_file() and p.suffix in (".cu", ".cpp", ".py", ".h", ".hpp", ".cuh"):
                sources[str(p.relative_to(ROOT))] = sha(p)
    commands = {}
    def run(label, command):
        commands[label] = list(map(str, command))
        with (output / f"{label}.log").open("w") as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        if rc:
            raise RuntimeError(f"{label}: rc={rc}; {output / (label + '.log')}")
        print(f"SIMT_BUILD stage={label} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    def compile_q(q):
        source = output / f"q{q}.cu"
        source.write_text(spec.source(q, a.profile))
        obj = output / f"q{q}.o"
        run(f"q{q}", [compiler, *flags, *[f"-I{p}" for p in includes], "-c", source, "-o", obj])
        return obj
    print(f"SIMT_BUILD_START platform={a.platform} formats=6 "
          f"bodies={2*sum(len(spec.inventory(q,a.profile)) for q in spec.QTYPES)} "
          f"profile={a.profile} jobs={a.jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=min(a.jobs, 6)) as pool:
        objects = list(pool.map(compile_q, spec.QTYPES))
    probe = output / "probe.cu"
    probe_source = (ROOT / "dev/gemv_simt/probe.cu").read_text()
    if a.platform == "ppu":
        probe_source = re.sub(r"\bcuda(?=[A-Z_])", "hggc", probe_source)
    probe.write_text(probe_source)
    run("probe", [compiler, *flags, *[f"-I{p}" for p in includes], "-c", probe, "-o", output / "probe.o"])
    objects.append(output / "probe.o")
    baseline = output / "libquactlize_simt_baseline.so"
    def compile_old(q):
        obj = output / f"old-q{q}.o"
        run(f"old-q{q}", [compiler, *flags, *[f"-I{p}" for p in includes], f"-DQKG_QTYPE={q}",
                          "-c", ROOT / "quactlize/execution/gemv.cu", "-o", obj])
        return obj
    with ThreadPoolExecutor(max_workers=min(a.jobs, 6)) as pool:
        old_objects = list(pool.map(compile_old, spec.QTYPES))
    old_dispatch = output / "old-dispatch.o"
    run("old-dispatch", ["g++", "-std=c++17", "-O3", "-fPIC", *[f"-I{p}" for p in includes],
                         "-c", ROOT / "quactlize/execution/dispatch.cpp", "-o", old_dispatch])
    run("old-link", [*linker, *old_objects, old_dispatch, *libs, "-o", baseline])
    dispatch = output / "dispatch.o"
    run("dispatch", ["g++", "-std=c++17", "-O3", "-fPIC", *[f"-I{p}" for p in includes],
                     "-c", ROOT / "quactlize/execution/simt.cpp", "-o", dispatch])
    target = output / "libquactlize_simt_candidates.so"
    run("link", [*linker, *objects, dispatch, *libs, "-o", target])
    if any(sha(ROOT / p) != s for p, s in sources.items()):
        raise ValueError("source changed during build; output has no admission receipt")
    manifest = spec.plan(a.profile) | dict(platform=a.platform, library=target.name,
        library_sha256=sha(target), source_hashes=sources, commands=commands,
        compiler_sha256=sha(compiler), build_seconds=time.monotonic()-started)
    manifest["baseline"] = dict(library=baseline.name, sha256=sha(baseline),
        scope="UNCHANGED_SCALAR_AND_PAIR_READER_NOT_Q4_OPTIMIZED_OR_TC")
    if a.platform == "ppu":
        run("isa", [sdk / "bin/hgobjdump", "--dump-isa", target])
        manifest["isa_sha256"] = sha(output / "isa.log")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SIMT_BUILD_COMPLETE output={output} device_validated=0", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("ppu", "cuda"), required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("full", "smoke"), default="full")
    p.add_argument("--arch", default="sm_120")
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    if a.jobs < 1:
        p.error("jobs must be positive")
    build(a)
