#!/usr/bin/env python3
"""Compile the three frozen Q4 readers plus the 60-recipe raw FP32 control."""
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
from dev.gemv_ppu.bload_source import reference_fp32, REFERENCE_SHA256
from dev.gemv_ppu.h800_port import SCHEMA, IMPLEMENTATIONS, REFERENCE_RECIPES, candidate_source, reference_source, verify
from dev.gemv_ppu.run import verify_bundle
from quactlize.runtime.compiler import FLAGS, LIBRARIES


def build(sdk, output, baseline, jobs):
    sdk, baseline = sdk.resolve(strict=True), baseline.resolve(strict=True)
    control = verify_bundle(baseline)
    if jobs < 1 or digest(sdk / "bin/hgcc") != control["compiler_sha256"]:
        raise ValueError("invalid jobs or control compiler differs")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = set(control["source_hashes"])
    inputs.update(str(p.relative_to(ROOT)) for p in (ROOT / "dev/gemv_cuda").iterdir()
                  if p.suffix in (".py", ".cuh", ".hpp", ".h"))
    inputs.update("dev/gemv_ppu/" + name for name in
                  ("build_h800_port.py", "h800_port.py", "bload_source.py", "bload_contract.hpp", "reference/gemv_ref.cuh"))
    hashes = {name: digest(ROOT / name) for name in sorted(inputs)}
    raw = ROOT / "dev/gemv_ppu/reference/gemv_ref.cuh"
    if digest(raw) != REFERENCE_SHA256:
        raise ValueError("supplied reference changed")
    for arm in IMPLEMENTATIONS:
        (output / f"{arm}.cu").write_text(candidate_source(arm))
    (output / "gemv_ref_fp32.cuh").write_text(reference_fp32(raw.read_text(), ppu=True))
    (output / "reference.cu").write_text(reference_source())
    env = dict(os.environ)
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    includes = [output, ROOT / "dev/gemv_ppu", ROOT / "quactlize/execution", ROOT / "quactlize/include",
                ROOT / "benchmarks", ROOT / "third_party/actlize/include", ROOT / "third_party/actlize/tools/util/include"]
    commands = []
    def run(label, argv):
        argv = list(map(str, argv))
        commands.append(dict(label=label, argv=argv))
        with (output / f"{label}.log").open("w") as log:
            rc = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise ValueError(f"{label} rc={rc}; log={output / (label + '.log')}")
        print(f"Q4_H800_PORT_BUILD phase={label} status=PASS", flush=True)
    units = [(arm, output / f"{arm}.cu") for arm in (*IMPLEMENTATIONS, "reference")]
    units.append(("probe", ROOT / "dev/gemv_ppu/probe.cu"))
    started = time.monotonic()
    def compile_one(unit):
        name, src = unit
        run(name, [sdk / "bin/hgcc", *FLAGS, "-DQKG_QTYPE=12", *[f"-I{p}" for p in includes],
                   "-c", src, "-o", output / f"{name}.o"])
    with ThreadPoolExecutor(max_workers=min(jobs, len(units))) as pool:
        list(pool.map(compile_one, units))
    payloads = {}
    for arm in (*IMPLEMENTATIONS, "reference"):
        library = output / f"libq4_ppu_port_{arm}.so"
        run("link-" + arm, ["g++", "-shared", "-Wl,-Bsymbolic", output / f"{arm}.o", output / "probe.o",
                            "-o", library, f"-L{sdk / 'lib'}", *[f"-l{name}" for name in LIBRARIES]])
        run("isa-" + arm, [sdk / "bin/hgobjdump", "--dump-isa", library])
        isa = (output / f"isa-{arm}.log").read_text()
        kernel = "q4_cooperative_metadata" if arm == "small" else "q4k_gemv_fp32" if arm == "reference" else "q4_group_affine"
        if "q4_ppu_marker" not in isa or kernel not in isa:
            raise ValueError("native device bodies missing: " + arm)
        payloads[arm] = dict(file=library.name, sha256=digest(library), isa_sha256=digest(output / f"isa-{arm}.log"))
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("source changed while compiling")
    data = dict(schema=SCHEMA, device_validated=False, production_changed=False,
                implementations=IMPLEMENTATIONS, reference_recipes=REFERENCE_RECIPES,
                baseline_manifest_sha256=digest(baseline / "manifest.json"), source_hashes=hashes,
                compiler_sha256=digest(sdk / "bin/hgcc"), inspector_sha256=digest(sdk / "bin/hgobjdump"),
                runtime={f"lib{name}.so": digest(sdk / "lib" / f"lib{name}.so") for name in LIBRARIES},
                payloads=payloads, commands=commands,
                generated_hashes={p.name: digest(p) for p in output.iterdir() if p.suffix in (".cu", ".cuh")},
                reference_original_sha256=REFERENCE_SHA256, seconds=time.monotonic() - started)
    (output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    verify(output, baseline)
    print(f"Q4_H800_PORT_BUILD status=COMPILED device_validated=0 seconds={data['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "prebuilt/ppu0010/q4-simt-ab-v1")
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    build(args.sdk, args.output, args.baseline, args.jobs)
