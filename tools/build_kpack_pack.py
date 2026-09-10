#!/usr/bin/env python3
"""Compile the GPU GGUF-to-Kpack producer without rebuilding GEMM modules."""

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


def build(sdk, output):
    sdk, output = sdk.resolve(), output.resolve()
    for path in [sdk / "bin/hgcc", *[sdk / "lib" / f"lib{x}.so" for x in LIBRARIES]]:
        if not path.is_file():
            raise ValueError(f"missing SDK input: {path}")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "quactlize/packing"
    includes = [source, ROOT / "quactlize/include", ROOT / "third_party/actlize/include"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    inputs = sorted({p for d in includes for p in d.rglob("*")
                     if p.is_file() and p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh", ".inc")})
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    commands = []
    for name in ("sizes.cpp", "ppu_pack.cu"):
        commands.append([str(sdk / "bin/hgcc"), *FLAGS,
                         *[f"-I{d}" for d in includes], "-c", str(source / name),
                         "-o", str(output / f"{name}.o")])

    def run(item):
        name, command = item
        print(f"KPACK_PACK_BUILD phase={name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"{name} failed rc={result.returncode}; log={output / f'{name}.log'}")

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, zip(("sizes", "device"), commands)))
    library = output / "libquactlize_ppu_pack.so"
    link = ["g++", "-shared", "-Wl,-Bsymbolic", *(c[-1] for c in commands),
            "-o", str(library), f"-L{sdk / 'lib'}",
            *[f"-l{x}" for x in LIBRARIES]]
    run(("link", link))
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("packing inputs changed during compilation")
    manifest = dict(
        schema="quactlize.kpack-device-pack-build.v1", source_hashes=hashes,
        compiler_sha256=sha(sdk / "bin/hgcc"), flags=FLAGS,
        runtime={f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES},
        library=library.name, sha256=sha(library), compile_seconds=time.monotonic() - start,
        formats=[8, 10, 11, 12, 13, 14], device_validated=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"KPACK_PACK_BUILD status=COMPILED device_validated=0 seconds={manifest['compile_seconds']:.3f} library={library}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    build(args.sdk, args.output)
