#!/usr/bin/env python3
"""Build a bounded decode endpoint gate and its selected-parent package, without a GPU."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.decode.compiler import DecodeCompiler
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES, sha, source_contract
from tools.build_kpack_dispatch import catalog, plan
from tools.verify_kpack_dispatch import verify
from tools.run_kpack_moe_gate import chain_requests


def native_simt_source(source):
    source = source.replace('<cuda_runtime.h>', '<hggc_runtime.h>')
    return re.sub(r'\bcuda(?=[A-Z_])', 'hggc', source)


def build(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = verify(args.base)
    compiler = DecodeCompiler(args.sdk, out / "modules", args.jobs)
    ordinary = Compiler(args.sdk, out / "modules", args.jobs)
    identity = dict(source=source_contract(ordinary.identity), typed=compiler.identity,
                    base_execution=base["execution_sha256"], builder=sha(__file__))
    stamp = out / "build-inputs.json"
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        if any(previous.get(k) != identity[k] for k in ("source", "typed", "base_execution")):
            raise ValueError("decode module inputs changed; use a new output directory")
    stamp.write_text(json.dumps(identity, indent=2) + "\n")
    inputs = [(q, route, m, 1024, 5120, 1, m)
              for q in (8, 10, 11, 12, 13, 14)
              for route in ((1,) if q == 8 else (0, 1)) for m in range(1, 9)]
    inputs += [(12, 0, m, n, k, 1, m)
               for n, k in ((512, 2048), (4096, 4096)) for m in (1, 8)]
    parents, selected = plan(out, inputs)
    decode_inputs = [r for r in inputs if r[0] == 12 and r[1] == 0]
    decode_parents, measured = plan(out, decode_inputs, decode=True)
    for row in selected:
        row["decode_policy"] = 0
    for row in measured:
        row["decode_policy"] = 1
    selected += measured
    parents = list({p["symbol"]: p for p in parents + decode_parents}.values())
    grouped_inputs = sorted({tuple(r[k] for k in ("q", "route", "m", "n", "k", "experts", "max_rows"))
                            for merged in (False, True) for t in (1, 8)
                            for r in chain_requests(merged, t)})
    controls, grouped = plan(out, grouped_inputs + [(12, 0, 1, 1024, 5120, 1, 1)])
    print(f"DECODE_IO_BUILD_PLAN typed_parents={len(parents)} controls={len(controls)} requests={len(selected)}", flush=True)
    started = time.monotonic()
    def progress(done, total):
        print(f"DECODE_IO_BUILD completed={done}/{total} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    records = compiler.compile_only(parents, progress)
    records += ordinary.compile_only(controls, progress)
    for record in records:
        record["path"] = str(Path(record["path"]).relative_to(out))
    bins = []
    for name in ("kpack_indexed_cuda", "kpack_moe_chain_cuda"):
        output = out / name
        log = out / (name + ".build.log")
        source = ROOT / "tests" / (name + ".cu")
        generated = out / (name + ".cu")
        generated.write_text(native_simt_source(source.read_text()))
        command = [str(args.sdk / "bin/hgcc"), *FLAGS, f"-I{ROOT}",
                   *[f"-I{p}" for p in ordinary.includes], str(generated),
                   "-o", str(output), "-Wl,--allow-shlib-undefined",
                   f"-L{args.sdk/'lib'}", *[f"-l{x}" for x in LIBRARIES]]
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        bins.append(dict(path=name, sha256=sha(output), source_sha256=sha(source),
                         generated_sha256=sha(generated)))
        print(f"DECODE_IO_SIMT_COMPILED {name}", flush=True)
    (out / "catalog.inc").write_text(catalog(records, source_contract(ordinary.identity)))
    host = out / "libquactlize_kpack_dispatch.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-pthread", "-Wl,-Bsymbolic",
                    f"-I{out}", str(ROOT / "quactlize/dispatch/binding.cpp"), "-ldl", "-o", str(host)], check=True)
    shutil.copy2(args.base / "libquactlize_ppu_execution.so", out / "libquactlize_ppu_execution.so")
    manifest = dict(schema="quactlize.kpack-native-dispatch.v1", modules=records,
                    dispatch_sha256=sha(host), execution_sha256=base["execution_sha256"],
                    execution_receipt=base["execution_receipt"], jit_required=True,
                    jit_source_contract=source_contract(ordinary.identity),
                    jit_source_identity={k: ordinary.identity[k] for k in ("kernel", "flags", "generator")},
                    device_validated=False, heuristic_admitted=False,
                    decode_io_gate=dict(requests=selected, grouped=grouped, simt_binaries=bins,
                                        source="EXISTING_PRODUCTION_SELECTORS", endpoints=["F32", "BF16"]))
    if "decode_policy" in base:
        manifest["decode_policy"] = base["decode_policy"]
        path = base["decode_policy"]["path"]
        shutil.copy2(args.base / path, out / path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(out, sdk=args.sdk)
    print(f"DECODE_IO_BUILD COMPLETE output={out} device_admission=PENDING", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--base", type=Path, default=ROOT / "prebuilt/ppu0010/q4-decode-policy-v1")
    build(p.parse_args())
