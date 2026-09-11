#!/usr/bin/env python3
"""Small native PPU add-on: matched SIMT B readers and raw GGUF FP32 reference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.bload_source import REFERENCE_SHA256, RECIPES, reference_fp32, reference_wrapper
from dev.gemv_ppu.build import ppu_api
from dev.gemv_ppu.run import verify_bundle
from quactlize.runtime.compiler import FLAGS, LIBRARIES

SCHEMA = "quactlize.q4-bload-ppu.v1"
PAYLOAD = "libq4_ppu_bload.so"


def verify(output, baseline, *, sources=True):
    control = verify_bundle(baseline, sources=sources)
    m = json.loads((output / "manifest.json").read_text())
    if (m.get("schema") != SCHEMA or m.get("recipes") != [list(r) for r in RECIPES]
            or m.get("baseline_manifest_sha256") != digest(baseline / "manifest.json")
            or m.get("reference_original_sha256") != REFERENCE_SHA256
            or m.get("compiler_sha256") != control["compiler_sha256"]):
        raise ValueError("B-load/reference experiment identity differs")
    if m["payload"]["file"] != PAYLOAD or digest(output / PAYLOAD) != m["payload"]["sha256"]:
        raise ValueError("B-load payload missing/changed/LFS pointer")
    with (output / PAYLOAD).open("rb") as f:
        if f.read(4) != b"\x7fELF": raise ValueError("not a native ELF")
    for name, expected in (m["source_hashes"].items() if sources else []):
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("B-load/reference source differs: " + name)
    return m


def build(sdk, output, baseline, jobs):
    sdk, baseline = sdk.resolve(strict=True), baseline.resolve(strict=True)
    control = verify_bundle(baseline)
    if digest(sdk / "bin/hgcc") != control["compiler_sha256"]:
        raise ValueError("use the original control compiler")
    raw = ROOT / "dev/gemv_ppu/reference/gemv_ref.cuh"
    if digest(raw) != REFERENCE_SHA256: raise ValueError("uploaded reference changed")
    if jobs < 1: raise ValueError("jobs must be positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = set(control["source_hashes"])
    sources.update(str(p.relative_to(ROOT)) for p in [raw, Path(__file__), *[
        ROOT / "dev/gemv_ppu" / name for name in
        ("bload_source.py", "bload_contract.hpp", "bload.cu", "bload_layout.cu")]])
    hashes = {name: digest(ROOT / name) for name in sorted(sources)}
    for name in ("q4_native.cuh", "q4_aligned.cuh"):
        (output / name).write_text(ppu_api((ROOT / "dev/gemv_cuda" / name).read_text()))
    (output / "gemv_ref_fp32.cuh").write_text(reference_fp32(raw.read_text()))
    (output / "raw_reference.cu").write_text(reference_wrapper())
    env = dict(os.environ)
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    includes = [output, ROOT / "dev/gemv_ppu", ROOT / "quactlize/include",
                ROOT / "third_party/actlize/include", ROOT / "third_party/actlize/tools/util/include"]
    commands = []
    def run(label, command):
        command = list(map(str, command))
        commands.append(dict(label=label, argv=command))
        with (output / (label + ".log")).open("w") as log:
            rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc: raise ValueError(f"{label}: rc={rc}; log={output / (label + '.log')}")
        print(f"Q4_BLOAD_BUILD phase={label} status=PASS", flush=True)
    started = time.monotonic()
    common = [sdk / "bin/hgcc", *FLAGS, *[f"-I{p}" for p in includes]]
    units = [(ROOT / "dev/gemv_ppu/bload.cu", output / "bload.o"),
             (output / "raw_reference.cu", output / "raw_reference.o"),
             (ROOT / "dev/gemv_ppu/probe.cu", output / "probe.o")]
    def compile_one(unit):
        src, obj = unit
        run(src.stem, [*common, "-c", src, "-o", obj])
    with ThreadPoolExecutor(max_workers=min(jobs, len(units))) as pool:
        list(pool.map(compile_one, units))
    libs = [f"-L{sdk / 'lib'}", *[f"-l{name}" for name in LIBRARIES]]
    # Pure host CuTe composition: use native SDK headers without linking its
    # device runtime, whose glibc floor may exceed the local compiler host.
    run("layout-build", ["g++", "-x", "c++", "-std=c++17", "-O2",
        "-DCUTLASS_USE_PACKED_TUPLE=1", "-DCUTE_USE_PACKED_TUPLE=1", f"-I{sdk / 'include'}",
        *[f"-I{p}" for p in includes], ROOT / "dev/gemv_ppu/bload_layout.cu", "-o", output / "layout"])
    run("layout-run", [output / "layout"])
    expected = "Q4_BLOAD_LAYOUT PASS words=256 codes=1024 bad=0 wrong_reg=256 wrong_nibble=1024"
    if expected not in (output / "layout-run.log").read_text(): raise ValueError("CuTe layout proof incomplete")
    run("link", ["g++", "-shared", "-Wl,-Bsymbolic", *[o for _, o in units], "-o", output / PAYLOAD, *libs])
    run("isa", [sdk / "bin/hgobjdump", "--dump-isa", output / PAYLOAD])
    isa = (output / "isa.log").read_text()
    for symbol in ("q4_ppu_marker", "q4_bload", "transport_gate", "q4k_gemv_fp32"):
        if symbol not in isa: raise ValueError("missing native device body: " + symbol)
    if any(digest(ROOT / name) != expected for name, expected in hashes.items()):
        raise ValueError("source changed during compilation")
    m = dict(schema=SCHEMA, device_validated=False, production_changed=False,
        recipes=RECIPES, reference_original_sha256=REFERENCE_SHA256,
        reference_generated_sha256=digest(output / "gemv_ref_fp32.cuh"),
        baseline_manifest_sha256=digest(baseline / "manifest.json"), source_hashes=hashes,
        compiler_sha256=digest(sdk / "bin/hgcc"), inspector_sha256=digest(sdk / "bin/hgobjdump"),
        runtime={f"lib{name}.so": digest(sdk / "lib" / f"lib{name}.so") for name in LIBRARIES},
        payload=dict(file=PAYLOAD, sha256=digest(output / PAYLOAD), isa_sha256=digest(output / "isa.log")),
        layout_proof_sha256=digest(output / "layout-run.log"), commands=commands,
        seconds=time.monotonic()-started)
    (output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    verify(output, baseline)
    print(f"Q4_BLOAD_BUILD status=COMPILED device_validated=0 seconds={m['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args()
    build(a.sdk, a.output, a.baseline, a.jobs)
