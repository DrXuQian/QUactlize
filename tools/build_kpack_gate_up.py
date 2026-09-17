#!/usr/bin/env python3
"""Build the explicit paired-N4 TC/SIMT library; no production selector changes."""

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

FORMATS = {8: 0, 10: 2, 11: 3, 12: 0, 13: 1, 14: 4}


def commands(sdk, output):
    includes = [
        ROOT / "quactlize/include",
        ROOT / "quactlize/runtime",
        ROOT / "quactlize/integrations",
        ROOT / "third_party/actlize/include",
        ROOT / "third_party/actlize/tools/util/include",
        ROOT / "third_party/actlize/examples/common",
    ]
    sources = [
        ("dispatch", "quactlize/fusion/dispatch.cu", []),
        ("pack", "quactlize/packing/ppu_pack.cu", []),
        ("sizes", "quactlize/packing/sizes.cpp", []),
    ]
    for q, fmt in FORMATS.items():
        for backend in ("simt", "tc"):
            sources.append(
                (
                    f"{backend}-q{q}",
                    f"quactlize/fusion/{backend}.cu",
                    [
                        f"-DQGU_QTYPE={q}",
                        f"-DPPU_PACKED_FORMAT={fmt}",
                        f"-DPPU_PACKED_SCALE={int(q!=8)}",
                    ],
                )
            )
    return [
        (
            name,
            [
                str(sdk / "bin/hgcc"),
                *FLAGS,
                *defs,
                *[f"-I{p}" for p in includes],
                "-c",
                str(ROOT / src),
                "-o",
                str(output / (name + ".o")),
            ],
        )
        for name, src, defs in sources
    ]


def build(sdk, output, jobs, only=None):
    sdk, output = sdk.resolve(), output.resolve()
    if jobs < 1:
        raise ValueError("jobs must be positive")
    for p in [sdk / "bin/hgcc", *[sdk / "lib" / f"lib{x}.so" for x in LIBRARIES]]:
        if not p.is_file():
            raise ValueError(f"missing SDK input: {p}")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    inputs = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in [
            "quactlize/fusion",
            "quactlize/execution",
            "quactlize/packing",
            "quactlize/include",
            "quactlize/runtime",
            "quactlize/decode",
            "third_party/actlize/include",
        ]
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in (".cu", ".cuh", ".cpp", ".h", ".hpp", ".inc")
    }
    cmds = commands(sdk, output)
    if only:
        requested = set(only.split(","))
        if requested - {n for n, _ in cmds}:
            raise ValueError("unknown compile-only unit")
        cmds = [x for x in cmds if x[0] in requested]

    def run(item):
        name, cmd = item
        print(f"GATE_UP_BUILD unit={name} status=START", flush=True)
        with (output / (name + ".log")).open("w") as log:
            rc = subprocess.run(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        print(f"GATE_UP_BUILD unit={name} rc={rc}", flush=True)
        return rc

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(jobs, len(cmds))) as pool:
        codes = list(pool.map(run, cmds))
    (output / "commands.json").write_text(json.dumps(cmds, indent=2) + "\n")
    if any(codes):
        raise RuntimeError(f"compilation failed; logs: {output}")
    if only:
        return
    lib = output / "libquactlize_ppu_gate_up.so"
    if run(
        (
            "link",
            [
                "g++",
                "-shared",
                "-Wl,-Bsymbolic",
                *[cmd[-1] for _, cmd in cmds],
                "-o",
                str(lib),
                f'-L{sdk/"lib"}',
                *[f"-l{x}" for x in LIBRARIES],
            ],
        )
    ):
        raise RuntimeError("gate/up link failed")
    if any(sha(ROOT / p) != h for p, h in inputs.items()):
        raise ValueError("build sources changed")
    manifest = dict(
        schema="quactlize.gate-up-paired-n4.v1",
        layout_id="0x47554e3400000001",
        source_hashes=inputs,
        compiler_sha256=sha(sdk / "bin/hgcc"),
        flags=FLAGS,
        runtime={f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES},
        library=lib.name,
        sha256=sha(lib),
        compile_seconds=time.monotonic() - start,
        formats=list(FORMATS),
        device_validated=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"GATE_UP_BUILD status=COMPILED device_validated=0 library={lib}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--compile-only", help="comma-separated units, no linked package")
    a = p.parse_args()
    build(a.sdk, a.output, a.jobs, a.compile_only)
