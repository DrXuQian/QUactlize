#!/usr/bin/env python3
"""Compile a small PPU A-staging comparison add-on; never rebuild its controls."""
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
from dev.gemv_ppu.astage_source import CASES, source, dispatch, validation
from dev.gemv_ppu.build import ppu_api
from dev.gemv_ppu.run import verify_bundle
from quactlize.runtime.compiler import FLAGS, LIBRARIES

SCHEMA = "quactlize.q4-astage-ppu.v1"


def verify(output, baseline, *, sources=True):
    control = verify_bundle(baseline, sources=sources)
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("cases") != json.loads(json.dumps(CASES))
            or manifest.get("baseline_manifest_sha256") != digest(baseline / "manifest.json")):
        raise ValueError("A-stage experiment/control identity differs")
    payload = manifest["payload"]
    if payload["file"] != "libq4_ppu_astage.so" or digest(output / payload["file"]) != payload["sha256"]:
        raise ValueError("A-stage payload missing, modified, or an LFS pointer")
    with (output / payload["file"]).open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("A-stage payload is not an ELF")
    for name, expected in (manifest["source_hashes"].items() if sources else []):
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("A-stage source changed: " + name)
    if manifest["compiler_sha256"] != control["compiler_sha256"]:
        raise ValueError("A/B compiler differs")
    return manifest


def build(sdk, output, baseline, jobs):
    sdk, baseline = sdk.resolve(strict=True), baseline.resolve(strict=True)
    control = verify_bundle(baseline)
    if digest(sdk / "bin/hgcc") != control["compiler_sha256"]:
        raise ValueError("use the baseline compiler for this one-seam experiment")
    compile_command = next(c["argv"] for c in control["commands"] if c["label"] == "new-gemv")
    if compile_command[1:len(FLAGS)+1] != FLAGS:
        raise ValueError("A/B compiler flags differ")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = set(control["source_hashes"])
    paths.update(str(p.relative_to(ROOT)) for p in (
        Path(__file__), ROOT / "dev/gemv_ppu/astage_source.py"))
    hashes = {name: digest(ROOT / name) for name in sorted(paths)}
    body = source((ROOT / "quactlize/execution/gemv.cu").read_text())
    for name in ("q4_native.cuh", "q4_aligned.cuh"):
        (output / name).write_text(ppu_api((ROOT / "dev/gemv_cuda" / name).read_text()))
        body = body.replace(str(ROOT / "dev/gemv_cuda" / name), str(output / name))
    (output / "gemv.cu").write_text(body)
    (output / "dispatch.cpp").write_text(dispatch((ROOT / "quactlize/execution/dispatch.cpp").read_text()))
    (output / "validation.hpp").write_text(validation((ROOT / "quactlize/execution/validation.hpp").read_text()))
    env = dict(os.environ)
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    includes = [output, ROOT / "quactlize/execution", ROOT / "quactlize/include", ROOT / "benchmarks",
                ROOT / "third_party/actlize/include", ROOT / "third_party/actlize/tools/util/include"]
    units = [(output / "gemv.cu", output / "gemv.o"), (output / "dispatch.cpp", output / "dispatch.o"),
             (ROOT / "dev/gemv_ppu/probe.cu", output / "probe.o")]
    commands = []
    def run(label, command):
        command = list(map(str, command))
        commands.append(dict(label=label, argv=command))
        with (output / (label + ".log")).open("w") as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise ValueError(f"{label}: rc={proc.returncode}; log={output / (label + '.log')}")
        print(f"Q4_ASTAGE_BUILD phase={label} status=PASS", flush=True)
    def compile_one(unit):
        src, obj = unit
        run(src.stem, [sdk / "bin/hgcc", *FLAGS, "-DQKG_QTYPE=12", *[f"-I{p}" for p in includes],
                       "-c", src, "-o", obj])
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(jobs, len(units))) as pool:
        list(pool.map(compile_one, units))
    library = output / "libq4_ppu_astage.so"
    run("link", ["g++", "-shared", "-Wl,-Bsymbolic", *[obj for _, obj in units], "-o", library,
                 f"-L{sdk / 'lib'}", *[f"-l{name}" for name in LIBRARIES]])
    run("isa", [sdk / "bin/hgobjdump", "--dump-isa", library])
    isa = (output / "isa.log").read_text()
    if "kpack_q4_large_static_astageILi4ELi8ELi5120ELi8192" not in isa or "q4_ppu_marker" not in isa:
        raise ValueError("expected PPU experiment/marker missing")
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("source changed during build")
    manifest = dict(schema=SCHEMA, device_validated=False, production_changed=False, cases=CASES,
        scope="A_GLOBAL_TO_SHARED_ONLY_FP16_A_FP32_DOT_S1_FIXED_PPU_WINNERS",
        baseline_manifest_sha256=digest(baseline / "manifest.json"), source_hashes=hashes,
        generated_hashes={p.name: digest(p) for p in output.iterdir() if p.suffix in (".cu", ".cpp", ".hpp", ".cuh")},
        compiler_sha256=digest(sdk / "bin/hgcc"), inspector_sha256=digest(sdk / "bin/hgobjdump"),
        runtime={f"lib{name}.so": digest(sdk / "lib" / f"lib{name}.so") for name in LIBRARIES},
        payload=dict(file=library.name, sha256=digest(library), isa_sha256=digest(output / "isa.log")),
        commands=commands, seconds=time.monotonic() - started)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(output, baseline)
    print(f"Q4_ASTAGE_BUILD status=COMPILED device_validated=0 seconds={manifest['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    build(args.sdk, args.output, args.baseline, args.jobs)
