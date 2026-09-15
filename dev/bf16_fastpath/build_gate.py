#!/usr/bin/env python3
"""Package the selected Q4 typed gate, or compile its exact closure for CUDA/PPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.bf16_fastpath.gate_plan import SCHEMA, plan
from quactlize.execution import q4_decode_codegen as codegen
from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha


def harness():
    paths = set((ROOT / "dev/bf16_fastpath").glob("gate*.py"))
    paths.add(Path(__file__))
    for name in ("dev/bf16_compute/fixture.py", "dev/gemv_simt/native.py",
                 "quactlize/execution/native.py", "reference/gguf_kpack.py",
                 "tools/kpack_warmup_fixture.py", "tools/run_kpack_pack_gate.py"):
        paths.add(ROOT / name)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def exports(path):
    names = subprocess.check_output(["nm", "-D", "--defined-only", path], text=True)
    for v in (1, 2):
        for name in ("select", "run"):
            if f" quactlize_kpack_q4_decode_{name}_v{v}\n" not in names:
                raise ValueError("execution image lacks the actual typed Q4 C exports")


def build(args):
    out, sdk = args.output.resolve(), args.sdk.resolve(strict=True)
    if not 1 <= args.jobs <= 8:
        raise ValueError("bounded gate compile jobs must be in 1..8")
    if args.execution and args.platform != "ppu":
        raise ValueError("PPU execution receipt cannot be reused on CUDA")
    started = time.monotonic()
    contract, generated = plan(), codegen.sources()
    sources, reused = {}, None
    if args.execution:
        root = args.execution.resolve(strict=True)
        reused = json.loads((root / "manifest.json").read_text())
        payload = (root / reused["library"]).resolve(strict=True)
        if (payload.parent != root or reused.get("schema") != "quactlize.kpack-execution-build.v1" or
                sha(payload) != reused["sha256"] or not reused.get("q4_decode_compute_v2")):
            raise ValueError("source execution build/typed receipt differs")
        sources = reused["source_hashes"]
        if any(sha(ROOT / p) != h for p, h in sources.items()):
            raise ValueError("source execution receipt differs from checkout; do not mix images")
        if reused.get("q4_decode_policy_sha256") != sha(codegen.POLICY):
            raise ValueError("execution policy differs from declared denominator")
        runtime = {f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES}
        if runtime != reused["runtime"]:
            raise ValueError("SDK runtime differs from reused execution")
    else:
        roots = (ROOT / "quactlize/execution", ROOT / "quactlize/include",
                 ROOT / "third_party/actlize/include", ROOT / "policies")
        sources = {str(p.relative_to(ROOT)): sha(p) for root in roots for p in root.rglob("*")
                   if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".cpp", ".cu", ".inc", ".json")}
        sources[str(Path(codegen.__file__).relative_to(ROOT))] = sha(codegen.__file__)
    gate_sources = harness()
    out.mkdir(parents=True, exist_ok=False)
    target = out / "libq4_selected_bf16_gate.so"
    commands = {}
    if args.execution:
        shutil.copy2(payload, target)
    else:
        env = dict(os.environ)
        env["PATH"] = str(sdk / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        includes = [ROOT, ROOT / "quactlize/execution", ROOT / "quactlize/include",
                    ROOT / "third_party/actlize/include"]
        if args.platform == "ppu":
            compiler, flags = sdk / "bin/hgcc", list(FLAGS)
            linker = ["g++", "-shared", "-Wl,-Bsymbolic"]
            libraries = [f"-L{sdk / 'lib'}", *[f"-l{x}" for x in LIBRARIES]]
        else:
            compiler = sdk / "bin/nvcc"
            version = subprocess.check_output([compiler, "--version"], text=True)
            if "NVIDIA" not in version or "HGG" in version:
                raise ValueError("CUDA requires real NVIDIA nvcc, not the PPU SDK alias")
            compat = ROOT / "dev/gemv_cuda/compat"
            includes.insert(0, compat)
            for p in compat.glob("*"):
                if p.is_file():
                    sources[str(p.relative_to(ROOT))] = sha(p)
            flags = ["-std=c++17", "-O3", "-lineinfo", f"-arch={args.arch}",
                "--expt-relaxed-constexpr", "-Xcompiler=-fPIC", "-Xptxas=-v",
                "-DCUTLASS_USE_PACKED_TUPLE=1", "-DCUTE_USE_PACKED_TUPLE=1",
                "-include", str(compat / "compiler_bridge.h")]
            linker = [str(compiler), "-shared", "--cudart=shared", "-Xlinker=-Bsymbolic"]
            libraries = []

        def run(label, command):
            commands[label] = list(map(str, command))
            with (out / f"{label}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"Q4_BF16_GATE_BUILD phase={label} elapsed_s={time.monotonic()-started:.1f}", flush=True)

        files = []
        for name, text in generated.items():
            path = out / name
            path.write_text(text)
            files.append(path)
        files.append(ROOT / "quactlize/execution/q4_decode.cpp")

        def compile_one(path):
            obj = out / (path.stem + ".o")
            if path.suffix == ".cpp":
                cmd = ["g++", "-std=c++17", "-O3", "-fPIC"]
            else:
                cmd = [compiler, *flags]
            run(path.stem, [*cmd, *[f"-I{p}" for p in includes], "-c", path, "-o", obj])
            return obj
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            objects = list(pool.map(compile_one, files))
        run("link", [*linker, *objects, *libraries, "-o", target])
        if args.platform == "ppu":
            run("isa", [sdk / "bin/hgobjdump", "--dump-isa", target])
    exports(target)
    if any(sha(ROOT / p) != h for p, h in (sources | gate_sources).items()):
        raise ValueError("source changed during gate package creation")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    manifest = dict(schema=SCHEMA, platform=args.platform, library=target.name, sha256=sha(target),
        plan=contract, source_hashes=sources, harness=gate_sources, commands=commands,
        generated={name: hashlib.sha256(text.encode()).hexdigest() for name, text in generated.items()},
        source_revision=revision.stdout.strip() if revision.returncode == 0 else "EXPORTED_SOURCE_HASHES_ONLY",
        build_seconds=time.monotonic()-started, reused_execution=reused,
        device_validated=False, performance_admitted=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Q4_BF16_GATE_PACKAGE denominator={json.dumps(contract['denominator'])} output={out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("ppu", "cuda"), required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--arch", default="sm_120")
    build(parser.parse_args())
