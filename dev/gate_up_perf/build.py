#!/usr/bin/env python3
"""Reuse immutable fusion/execution images; compile only two TC controls + post-op."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gate_up_perf.plan import points, incumbent, parent
from quactlize.decode.compiler import DecodeCompiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha
from tools.run_kpack_gate_up import verify


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    lib, fusion = verify(args.fusion, args.sdk)
    base = json.loads((args.baseline / "manifest.json").read_text())
    pin = json.loads((ROOT / "tools/kpack_q4_model_artifact.json").read_text())
    if sha(args.baseline / "manifest.json") != pin["manifest_sha256"]:
        raise ValueError("baseline is not the pinned production package")
    runtime = {f"lib{x}.so": sha(args.sdk / "lib" / f"lib{x}.so") for x in LIBRARIES}
    if base["execution_receipt"]["runtime"] != runtime or runtime != fusion["runtime"]:
        raise ValueError("baseline and fusion runtime differ")
    execution = args.baseline / "libquactlize_ppu_execution.so"
    if sha(execution) != base["execution_sha256"]:
        raise ValueError("baseline execution image differs")
    for key in ("smallm_matched_policy", "q8_vector_policy"):
        r = base[key]
        if sha(args.baseline / r["path"]) != r["sha256"]:
            raise ValueError("pinned baseline policy file differs: " + key)
    policies = {
        key: json.loads((args.baseline / name).read_text())
        for key, name in (
            ("matched", "smallm-matched-policy.json"),
            ("vector", "q8-vector-policy.json"),
        )
    }
    cohort = [p | dict(incumbent=incumbent(p, **policies)) for p in points()]
    tc = {
        p["incumbent"]["config"]["symbol"]: p
        for p in cohort
        if p["incumbent"]["config"]["kind"] == "tc"
    }
    env = dict(os.environ)
    env["PATH"] = str(args.sdk / "bin") + ":" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(args.sdk / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    os.environ.update({k: env[k] for k in ("PATH", "LD_LIBRARY_PATH")})
    started = time.monotonic()

    def compile_tc(p):
        compiler = (GroupedComputeCompiler if p["mode"] else DecodeCompiler)(
            args.sdk,
            args.output / "modules",
            compute_type="bf16" if p["compute"] else "f16",
        )
        record = compiler.build(parent(p["incumbent"]))
        record["path"] = str(Path(record["path"]).relative_to(args.output))
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        modules = list(pool.map(compile_tc, tc.values()))
    output = args.output / "libgate_up_perf_postop.so"
    cmd = [
        str(args.sdk / "bin/hgcc"),
        *FLAGS,
        f'-I{ROOT/"third_party/actlize/include"}',
        "-shared",
        str(ROOT / "dev/gate_up_perf/postop.cu"),
        "-o",
        str(output),
        f'-L{args.sdk/"lib"}',
        *[f"-l{x}" for x in LIBRARIES],
    ]
    with (args.output / "postop.log").open("w") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(lib, args.output / lib.name)
    shutil.copy2(execution, args.output / execution.name)
    payloads = {
        p.name: sha(p)
        for p in (output, args.output / lib.name, args.output / execution.name)
    }
    manifest = dict(
        schema="quactlize.gate-up-perf.v1",
        source_commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        payloads=payloads,
        runtime=runtime,
        fusion_manifest=fusion,
        baseline_pin=pin,
        policy_hashes={
            name: sha(args.baseline / name)
            for name in ("smallm-matched-policy.json", "q8-vector-policy.json")
        },
        points=cohort,
        modules=modules,
        source_hashes={
            str(p.relative_to(ROOT)): sha(p)
            for p in (ROOT / "dev/gate_up_perf").glob("*")
            if p.suffix in (".py", ".cu")
        },
        compile_seconds=time.monotonic() - started,
        device_validated=False,
        baseline_postop="MINIMAL_STANDALONE_NOT_APPLICATION_GLUE",
        tc_scope="CURRENT_SELECTED_PARENT_REBUILT_TYPED_ENDPOINT_INCLUDING_INDEXED_ADAPTERS",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "GATE_UP_PERF_BUILD COMPLETE modules="
        + str(len(modules))
        + " seconds="
        + str(manifest["compile_seconds"]),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "fusion", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    build(p.parse_args())
