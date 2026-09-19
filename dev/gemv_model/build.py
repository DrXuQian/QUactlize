#!/usr/bin/env python3
"""Local narrow compile; immutable model controls, no production rebuild."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_model.plan import POINTS, SCHEMA, BASE_SOURCE, BASE_ARTIFACT, BASE_EXECUTION, BASE_FUSION, candidates, inventory
from dev.gemv_model.plan import cohort
from dev.gemv_model.source import source
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha
from quactlize.runtime.native import sdk_identity
from quactlize.decode.compiler import DecodeCompiler


def build(args):
    plan = cohort(args.cohort)
    points = [p for p in plan.POINTS if not args.points or p.name in args.points.split(',')]
    if not points or (args.points and set(args.points.split(',')) - {p.name for p in plan.POINTS}):
        raise ValueError('unknown/empty compile point set')
    sdk, out, base = (p.resolve(strict=True) for p in (args.sdk, args.output.parent, args.baseline))
    out = out / args.output.name
    out.mkdir(exist_ok=False)
    for name, digest in [("execution", plan.BASE_EXECUTION), ("gate_up", plan.BASE_FUSION)]:
        path = base / f"libquactlize_ppu_{name}.so"
        if sha(path) != digest:
            raise ValueError("immutable model control differs: " + name)
        shutil.copy2(path, out / path.name)
    include = [ROOT, ROOT / "quactlize/include", ROOT / "third_party/actlize/include",
               ROOT / "third_party/actlize/tools/util/include"]
    paths = [p for folder in (ROOT / "quactlize", ROOT / "third_party/actlize/include")
             for p in folder.rglob("*") if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".inc")]
    paths += [Path(__file__), ROOT / "dev/gemv_model/source.py", ROOT / "dev/gemv_model/plan.py"]
    paths += [Path(plan.__file__)]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    env = dict(os.environ)
    env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    compiler = DecodeCompiler(sdk, out / "tc-cache", jobs=1, compute_type="f16")
    start = time.monotonic()

    def one(p):
        src, obj, lib = [out / (p.name + suffix) for suffix in (".cu", ".o", ".so")]
        configs = plan.candidates(p)
        src.write_text(source(p, configs))
        commands = [[str(sdk / "bin/hgcc"), *FLAGS, *[f"-I{x}" for x in include], "-c", str(src), "-o", str(obj)],
                    ["g++", "-shared", "-Wl,-Bsymbolic", "-Wl,-z,defs", str(obj), f"-L{sdk}/lib",
                     *[f"-l{x}" for x in LIBRARIES], "-o", str(lib)]]
        print(f"MODEL_GEMV_BUILD point={p.name} candidates={len(configs)} status=START", flush=True)
        with (out / (p.name + ".build.log")).open("x") as log:
            for cmd in commands:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for suffix, flags in [("resources.txt", ["--dump-resource-usage=all"]), ("isa.txt", ["--dump-isa"])]:
            with (out / (p.name + "." + suffix)).open("x") as log:
                subprocess.run([sdk / "bin/hgobjdump", *flags, lib], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        record = dict(point=asdict(p), candidates=[asdict(c) for c in configs],
                      library=lib.name, sha256=sha(lib), source_sha256=sha(src), commands=commands)
        if p.tc:
            # Recompile exactly the frozen parent from unchanged source. The
            # different module key records this provenance; it is not a retune.
            tc = compiler.build(p.parent)
            dest = out / (p.name + ".tc.so")
            shutil.copy2(tc["path"], dest)
            record["tc"] = tc | dict(path=dest.name)
        print(f"MODEL_GEMV_BUILD point={p.name} status=PASS elapsed_s={time.monotonic()-start:.1f}", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(points))) as pool:
        records = list(pool.map(one, points))
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("compile input changed")
    payloads = {p.name: sha(p) for p in out.iterdir() if p.suffix in (".so", ".cu", ".txt")}
    result = dict(schema=plan.SCHEMA, cohort=args.cohort, inventory=plan.inventory(),
                  compiled_points=[p.name for p in points], records=records, payloads=payloads,
                  source_hashes=hashes, sdk=sdk_identity(sdk),
                  runtime={f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES},
                  baseline_source=plan.BASE_SOURCE, baseline_artifact=plan.BASE_ARTIFACT,
                  build_seconds=time.monotonic()-start, selector_changed=False, device_admission="PENDING")
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"MODEL_GEMV_BUILD PASS seconds={result['build_seconds']:.1f} output={out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "output", "baseline"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument('--cohort', choices=('model','tp2'), default='model')
    parser.add_argument('--points', help='bounded comma-separated subset; full inventory remains in the receipt')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("positive jobs required")
    build(args)
