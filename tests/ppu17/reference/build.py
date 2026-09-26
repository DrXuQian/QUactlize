"""Build the GPTQ-only comparison with isolated PPU/reference include graphs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
VLLM = "11ba93f3646d4c5476c3b3fd56835589701f0fb1"
CUTLASS = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
PPU_CUTLASS = "023e82d03e80b4d5982f664925e15499033df314"
ACTLIZE = "021c69300864e03926108867d318572871a44d35"
CONTRACT = "GPTQ_SYM_INT4_G128_F16_V1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency(path, revision):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    if actual != revision or dirty:
        raise ValueError(f"reference checkout must be clean and pinned: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vllm", type=Path, required=True)
    p.add_argument("--cutlass", type=Path, required=True)
    p.add_argument("--ppu-cutlass", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compiler", type=Path, required=True)
    p.add_argument('--reader', choices=('conventional','prepacked'), default='conventional')
    args = p.parse_args()
    args.vllm = args.vllm.resolve(strict=True)
    args.cutlass = args.cutlass.resolve(strict=True)
    args.ppu_cutlass = args.ppu_cutlass.resolve(strict=True)
    args.compiler = args.compiler.resolve(strict=True)
    args.output = args.output.resolve()
    dependency(args.vllm, VLLM)
    dependency(args.cutlass, CUTLASS)
    dependency(args.ppu_cutlass, PPU_CUTLASS)
    if args.output == HERE or HERE in args.output.parents:
        raise ValueError("use a build root outside the source directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise ValueError("use a fresh build root; do not overwrite a reference authority")
    common = [str(args.compiler), "-std=c++17", "-O3", "-DNDEBUG",
              "--expt-relaxed-constexpr", "--extended-lambda",
              "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
              "-Xcompiler=-fPIC,-fvisibility=hidden"]
    root = HERE.parents[2]
    dependency(root/'third_party/actlize', ACTLIZE)
    sources = {str(f.relative_to(HERE)): sha(f) for f in sorted(HERE.rglob("*"))
               if f.is_file() and f.suffix in (".cu", ".cuh", ".h", ".py")}
    sources['gptq.cuh'] = sha(root/'quactlize/backends/ppu17/gptq.cuh')
    sources['mixed_input_policy.cuh'] = sha(root/'quactlize/backends/ppu17/mixed_input_policy.cuh')
    sources['target.cuh'] = sha(root/'quactlize/backends/ppu17/target.cuh')
    sources['gptq_prepacked.cuh'] = sha(root/'quactlize/backends/ppu17/gptq_prepacked.cuh')
    sources['marlin_scheduler.cuh'] = sha(root/'quactlize/backends/ppu17/marlin_scheduler.cuh')
    sources['marlin_core'] = sha(root/'third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp')
    bound_files = {str(f): sha(f) for f in sorted(HERE.rglob('*'))
                   if f.is_file() and f.suffix in ('.cu', '.cuh', '.h', '.py')}
    for name in ('gptq.cuh', 'gptq_prepacked.cuh', 'mixed_input_policy.cuh', 'target.cuh', 'marlin_scheduler.cuh'):
        path = root/'quactlize/backends/ppu17'/name
        bound_files[str(path)] = sha(path)
    core = root/'third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp'
    bound_files[str(core)] = sha(core)
    manifest = {"schema": 2, "scope": "PHYSICAL_HOPPER_GPTQ_COMPARISON_NOT_PRODUCT",
                "contract": CONTRACT, "ppu_cutlass": PPU_CUTLASS,
                "actlize_integer_core": ACTLIZE,
                "quactlize_reader": args.reader,
                "vllm": VLLM, "cutlass": CUTLASS, "sources": sources,
                "compiler": str(args.compiler), "compiler_sha256": sha(args.compiler),
                "compiler_version": subprocess.check_output([str(args.compiler), "--version"], text=True),
                "flags": common[1:], "libraries": {}}
    def build(arm):
        temp = args.output / (arm + "-tmp")
        temp.mkdir(exist_ok=True)
        library = args.output / ("lib" + arm + "_reference.so")
        if arm == 'quactlize':
            includes = ['-I'+str(root), '-I'+str(args.ppu_cutlass/'include')]
            if args.reader == 'prepacked':
                includes += ['-I'+str(HERE/'shim'), '-I'+str(args.vllm/'csrc')]
        else:
            includes = ['-I'+str(HERE/'shim'), '-I'+str(args.vllm/'csrc'),
                        '-I'+str(args.cutlass/'include')]
        command = common + includes
        if arm != 'marlin':
            command += ["-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1"]
        source_name = 'quactlize_prepacked' if arm == 'quactlize' and args.reader == 'prepacked' else arm
        command += ["-shared", str(HERE / (source_name + ".cu")), "-o", str(library)]
        if arm != "marlin":
            # CUTLASS's direct tensor-map API is supplied by the installed
            # driver at runtime; only use the toolkit stub for the link step.
            command += ["-L" + str(args.compiler.parent.parent / "lib64/stubs"), "-lcuda"]
        (args.output / (arm + ".command.json")).write_text(json.dumps(command, indent=2)+"\n")
        began = time.monotonic()
        with (args.output / (arm + ".build.log")).open("w") as log:
            subprocess.run(command, env={**os.environ, "TMPDIR": str(temp)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return arm, {"file": library.name, "sha256": sha(library),
                     "seconds": time.monotonic()-began}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for arm, row in pool.map(build, ("marlin", "machete", "quactlize")):
            manifest["libraries"][arm] = row
            print("PPU17_REFERENCE_BUILD " + arm + " PASS", flush=True)
    if any(sha(path) != digest for path, digest in bound_files.items()):
        raise ValueError('source changed during compilation; no build authority published')
    dependency(args.vllm, VLLM)
    dependency(args.cutlass, CUTLASS)
    dependency(args.ppu_cutlass, PPU_CUTLASS)
    dependency(root/'third_party/actlize', ACTLIZE)
    manifest['config_counts'] = {'machete': 9, 'marlin': 3, 'quactlize': 9 if args.reader == 'prepacked' else 12}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
