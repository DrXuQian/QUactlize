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
from quactlize.runtime.tuning import digest

SCHEMA = "quactlize.kpack-prefetch.v1"
BASE = Path("prebuilt/ppu0010/kpack-grouped-postops-v1")
SDK_TOOLS = ("bin/hgcc", "bin/hgobjdump")
SDK_RUNTIME = tuple(f"lib/lib{name}.so" for name in LIBRARIES)
SDK_FILES = SDK_TOOLS + SDK_RUNTIME


def sdk_files(sdk, *, optional_tools=False):
    files = {}
    for relative in SDK_FILES:
        try:
            files[relative] = sha(Path(sdk) / relative)
        except FileNotFoundError:
            if not optional_tools or relative not in SDK_TOOLS:
                raise
            files[relative] = None
    return files


def valid_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_sdk_files(manifest):
    files = manifest.get("sdk_files", {})
    if set(files) != set(SDK_FILES) or any(not valid_hash(v) for v in files.values()):
        raise ValueError(
            "missing/invalid per-file build SDK receipt; pull updated manifest"
        )
    if digest([(p, files[p]) for p in SDK_FILES]) != manifest["sdk"]:
        raise ValueError(
            "per-file build SDK receipt differs from original combined identity"
        )
    return files


def runtime_sdk_report(manifest, actual, *, allow_unverified=False):
    expected = validate_sdk_files(manifest)
    if set(actual) != set(SDK_FILES) or any(
        not valid_hash(actual[p]) and not (p in SDK_TOOLS and actual[p] is None)
        for p in SDK_FILES
    ):
        raise ValueError(
            "missing/invalid runtime SDK files; an override cannot supply libraries"
        )
    differences = [
        dict(path=p, build_sha256=expected[p], actual_sha256=actual[p])
        for p in SDK_FILES
        if expected[p] != actual[p]
    ]
    runtime_changed = any(row["path"] in SDK_RUNTIME for row in differences)
    allowed = not runtime_changed or allow_unverified
    if runtime_changed:
        status = "UNVERIFIED_RUNTIME" if allowed else "REJECTED_RUNTIME_MISMATCH"
    else:
        status = "RUNTIME_MATCH_TOOLS_DIFFER" if differences else "EXACT_SDK"
    return dict(
        status=status,
        allowed=allowed,
        override_requested=allow_unverified,
        build_sdk=manifest["sdk"],
        actual_sdk=digest([(p, actual[p]) for p in SDK_FILES]),
        build_files=expected,
        actual_files=actual,
        differences=differences,
        runtime_matches=not runtime_changed,
        device_validation="REQUIRED_NOT_IMPLIED_BY_HASHES",
    )


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
    files = sdk_files(sdk)
    validate_sdk_files(dict(sdk=identity, sdk_files=files))
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
    if sdk_files(sdk) != files:
        raise ValueError("SDK changed during compilation")
    manifest = dict(
        schema=SCHEMA,
        library=library.name,
        sha256=sha(library),
        sdk=identity,
        sdk_files=files,
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
