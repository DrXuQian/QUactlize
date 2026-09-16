"""Build only the explicit BF16 capability gate's bounded parent inventory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.bf16_compute.plan import modules, serializable
from quactlize.decode.compiler import DecodeCompiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha
from dev.bf16_compute.matched import parents as matched_parents


def run(command, log):
    with log.open("w") as out:
        out.write(json.dumps(list(map(str, command))) + "\n")
        out.flush()
        subprocess.run(command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)


def execution_receipt(directory):
    directory = directory.resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "quactlize.kpack-execution-build.v1":
        raise ValueError("execution reuse requires its original build receipt")
    path = (directory / manifest["library"]).resolve(strict=True)
    if path.parent != directory or sha(path) != manifest["sha256"]:
        raise ValueError("execution reuse payload differs")
    if any(sha(ROOT / name) != expected for name, expected in manifest["source_hashes"].items()):
        raise ValueError("execution reuse source differs from checkout")
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(path)], text=True)
    names = ("simt_query_v2", "simt_run_v2", "moe_simt_query_v1", "moe_simt_bind_v1",
             "moe_mixed_stage_v2", "moe_weighted_finish_v2")
    if any("quactlize_kpack_" + name not in symbols for name in names):
        raise ValueError("execution reuse lacks explicit BF16 endpoints")
    for q in (8, 10, 11, 12, 13, 14):
        wanted = dict(variant=1 if q == 8 else 3, columns=4, warps=4, values=4)
        if not any(all(config.get(name) == value for name, value in wanted.items())
                   for config in manifest.get("simt_configs", {}).get(str(q), [])):
            raise ValueError(f"execution reuse lacks the gate's SIMT recipe for q={q}")
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--execution", type=Path, help="reuse a source-matching execution build directory")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("bounded gate jobs must be in 1..8")
    reuse = execution_receipt(args.execution) if args.execution else None
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "modules").mkdir()
    source = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "dev/bf16_compute").glob("*.py")}
    started = time.monotonic()
    def build(parent):
        cls = GroupedComputeCompiler if parent.route.endswith("grouped") else DecodeCompiler
        record = cls(args.sdk, args.cache, compute_type=parent.compute).build(parent.record())
        destination = out / "modules" / (parent.key + ".so")
        shutil.copy2(record["path"], destination)
        record["path"] = str(destination.relative_to(out))
        record["spec"] = asdict(parent)
        print(f"BF16_GATE_BUILD parent={parent.key} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        return parent.key, record
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = dict(pool.map(build, modules()))
    def build_matched(item):
        key, compute, parent = item
        record = GroupedComputeCompiler(args.sdk,args.cache,compute_type=compute).build(parent)
        destination = out / "modules" / (key + ".so")
        shutil.copy2(record["path"],destination)
        record["path"] = str(destination.relative_to(out))
        record["spec"] = dict(parent,compute=compute)
        print(f"BF16_GATE_BUILD parent={key} elapsed_s={time.monotonic()-started:.1f}",flush=True)
        return key,record
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        matched = dict(pool.map(build_matched,matched_parents()))
    reused_execution = None
    if reuse:
        source_library, reused_execution = reuse
        helper = simt = out / "libquactlize_ppu_execution.so"
        shutil.copy2(source_library, helper)
    else:
        run([sys.executable, ROOT / "dev/gemv_simt/build.py", "--platform", "ppu",
             "--sdk", args.sdk, "--output", out / "simt", "--profile", "smoke", "--jobs", str(args.jobs)],
            out / "simt-build.log")
        includes = [ROOT, ROOT / "quactlize/include", ROOT / "third_party/actlize/include",
                    ROOT / "third_party/actlize/tools/util/include"]
        obj = out / "moe.o"
        helper = out / "libbf16_moe_helpers.so"
        run([args.sdk / "bin/hgcc", *FLAGS, *[f"-I{p}" for p in includes], "-c",
             ROOT / "quactlize/execution/moe.cu", "-o", obj], out / "moe-build.log")
        run(["g++", "-shared", "-Wl,-Bsymbolic", obj, f"-L{args.sdk / 'lib'}",
             *[f"-l{name}" for name in LIBRARIES], "-o", helper], out / "moe-link.log")
        simt = out / "simt/libquactlize_simt_candidates.so"
    if any(sha(ROOT / name) != expected for name, expected in source.items()):
        raise ValueError("gate source changed during package build")
    current = {}
    for record in [*records.values(), *matched.values()]:
        spec = record["spec"]
        kind = spec["route"].endswith("grouped")
        key = (kind, spec["compute"])
        if key not in current:
            cls = GroupedComputeCompiler if kind else DecodeCompiler
            current[key] = cls(args.sdk, args.cache, compute_type=spec["compute"]).identity
        if record["identity"] != current[key]:
            raise ValueError("module source changed during package build")
    if reuse:
        execution_receipt(args.execution)
    manifest = dict(schema="quactlize.bf16-device-gate.v1", modules=records,
        matched_modules=matched,
        simt=dict(path=str(simt.relative_to(out)), sha256=sha(simt)),
        moe=dict(path=helper.name, sha256=sha(helper)), cases=serializable(),
        source=source, build_seconds=time.monotonic()-started, device_validated=False,
        reused_execution=reused_execution,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"BF16_GATE_PACKAGE modules={len(records)} cases={len(manifest['cases'])} device_validated=0 path={out}")


if __name__ == "__main__":
    main()
