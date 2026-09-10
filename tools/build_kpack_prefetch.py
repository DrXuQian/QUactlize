#!/usr/bin/env python3
"""Compile only the small cache probe; reuse the measured Q4/Q5 GEMM images."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha
from quactlize.runtime.native import sdk_identity

SCHEMA = "quactlize.kpack-prefetch.v1"
BASE = Path("prebuilt/ppu0010/kpack-grouped-postops-v1")


def subjects():
    baseline = json.loads((ROOT / BASE / "manifest.json").read_text())
    modules = {m["key"]: m for m in baseline["modules"]}
    result = []
    for q, split, n, k in ((12, 4, 512, 2048), (13, 1, 2048, 512)):
        group = next(
            g for g in baseline["groups"] if g["job"] == f"fq-q{q}-tm8-ordinary"
        )
        record = modules[group["candidate"]]
        if sha(ROOT / BASE / record["path"]) != record["sha256"]:
            raise ValueError("measured parent payload differs")
        result.append(
            dict(
                case=f"q{q}",
                q=q,
                split=split,
                n=n,
                k=k,
                module=record | {"path": str(BASE / record["path"])},
            )
        )
    return result


def build(sdk, output):
    sdk, output = sdk.resolve(strict=True), output.resolve()
    rows = subjects()
    identity = sdk_identity(sdk)
    if any(r["module"]["identity"]["sdk"] != identity for r in rows):
        raise ValueError("probe and measured parents require the same SDK")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "dev/l2_prefetch/probe.cu"
    headers = ROOT / "third_party/actlize/include"
    inputs = [source, headers / "cute/arch/copy.hpp", headers / "cute/config.hpp"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    obj, library = output / "probe.o", output / "libkpack_prefetch.so"
    commands = [
        [
            str(sdk / "bin/hgcc"),
            *FLAGS,
            f"-I{headers}",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        [
            "g++",
            "-shared",
            "-Wl,-Bsymbolic",
            str(obj),
            "-o",
            str(library),
            f"-L{sdk / 'lib'}",
            *[f"-l{x}" for x in LIBRARIES],
        ],
    ]
    start = time.monotonic()
    with (output / "build.log").open("x") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(
                command,
                check=True,
                env=dict(os.environ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    with (output / "isa.txt").open("x") as log:
        subprocess.run(
            [str(sdk / "bin/hgobjdump"), "--dump-isa", str(obj)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    isa = (output / "isa.txt").read_text()
    if "vmem.prefetch" not in isa or "vmem.ld.b32" not in isa:
        raise ValueError("expected device prefetch/load instructions absent")
    if any(sha(ROOT / p) != value for p, value in hashes.items()):
        raise ValueError("probe source changed during compilation")
    manifest = dict(
        schema=SCHEMA,
        library=library.name,
        sha256=sha(library),
        sdk=identity,
        source_hashes=hashes,
        subjects=rows,
        compile_seconds=time.monotonic() - start,
        device_validated=False,
        production_selection_changed=False,
        source=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"KPACK_PREFETCH_BUILD COMPILED seconds={manifest['compile_seconds']:.1f} library={library}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.sdk, args.output)
