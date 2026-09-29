"""Build the GPTQ-only comparison with isolated PPU/reference include graphs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
VLLM = "11ba93f3646d4c5476c3b3fd56835589701f0fb1"
CUTLASS = "da5e086dab31d63815acafdac9a9c5893b1c69e2"
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'quactlize/backends/ppu17'))
from dependency import REVISION as PPU_CUTLASS, require_cutlass
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
    p.add_argument('--reader', choices=('conventional','prepacked','aligned'), default='conventional',
                   help='aligned retains the prepacked provider and adds the complete Marlin provider')
    args = p.parse_args()
    args.vllm = args.vllm.resolve(strict=True)
    args.cutlass = args.cutlass.resolve(strict=True)
    args.ppu_cutlass = args.ppu_cutlass.resolve(strict=True)
    args.compiler = args.compiler.resolve(strict=True)
    args.output = args.output.resolve()
    dependency(args.vllm, VLLM)
    dependency(args.cutlass, CUTLASS)
    ppu_dependency = require_cutlass(args.ppu_cutlass)
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
    sources['target.cuh'] = sha(root/'quactlize/backends/ppu17/target.cuh')
    sources['gptq_prepacked.cuh'] = sha(root/'quactlize/backends/ppu17/gptq_prepacked.cuh')
    sources['gptq_marlin.cuh'] = sha(root/'quactlize/backends/ppu17/gptq_marlin.cuh')
    sources['marlin_format_ppu.hpp'] = sha(root/'quactlize/include/marlin_format_ppu.hpp')
    sources['marlin_scheduler.cuh'] = sha(root/'quactlize/backends/ppu17/marlin_scheduler.cuh')
    sources['marlin_core'] = sha(root/'third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp')
    bound_files = {str(f): sha(f) for f in sorted(HERE.rglob('*'))
                   if f.is_file() and f.suffix in ('.cu', '.cuh', '.h', '.py')}
    for name in ('gptq.cuh', 'gptq_prepacked.cuh', 'gptq_marlin.cuh', 'target.cuh', 'marlin_scheduler.cuh'):
        path = root/'quactlize/backends/ppu17'/name
        bound_files[str(path)] = sha(path)
    core = root/'third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp'
    bound_files[str(core)] = sha(core)
    host_format = root/'quactlize/include/marlin_format_ppu.hpp'
    bound_files[str(host_format)] = sha(host_format)
    identity_source = ROOT/'quactlize/backends/ppu17/dependency.py'
    bound_files[str(identity_source)] = sha(identity_source)
    manifest = {"schema": 2, "scope": "PHYSICAL_HOPPER_GPTQ_COMPARISON_NOT_PRODUCT",
                "contract": CONTRACT, "ppu_cutlass": PPU_CUTLASS,
                "ppu_cutlass_identity": ppu_dependency,
                "dependency_validator_sha256": sha(identity_source),
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
        if arm.startswith('quactlize'):
            includes = ['-I'+str(root), '-I'+str(args.ppu_cutlass/'include')]
            if args.reader in ('prepacked','aligned'):
                includes += ['-I'+str(HERE/'shim'), '-I'+str(args.vllm/'csrc')]
        else:
            includes = ['-I'+str(HERE/'shim'), '-I'+str(args.vllm/'csrc'),
                        '-I'+str(args.cutlass/'include')]
        command = common + includes
        if arm not in ('marlin','quactlize_marlin'):
            command += ["-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1"]
        source_name = 'quactlize_prepacked' if arm == 'quactlize' and args.reader in ('prepacked','aligned') else arm
        command += ["-shared", str(HERE / (source_name + ".cu")), "-o", str(library)]
        if arm not in ('marlin','quactlize_marlin'):
            # CUTLASS's direct tensor-map API is supplied by the installed
            # driver at runtime; only use the toolkit stub for the link step.
            command += ["-L" + str(args.compiler.parent.parent / "lib64/stubs"), "-lcuda"]
        (args.output / (arm + ".command.json")).write_text(json.dumps(command, indent=2)+"\n")
        began = time.monotonic()
        with (args.output / (arm + ".build.log")).open("w") as log:
            subprocess.run(command, env={**os.environ, "TMPDIR": str(temp)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        reader = ('marlin' if arm in ('marlin','quactlize_marlin') else
                  'prepacked' if arm == 'machete' or args.reader in ('prepacked','aligned') else 'conventional')
        return arm, {"file": library.name, "sha256": sha(library),
                     "reader": reader, "seconds": time.monotonic()-began}
    arms = ['marlin','machete','quactlize']
    if args.reader == 'aligned': arms.append('quactlize_marlin')
    with ThreadPoolExecutor(max_workers=3) as pool:
        for arm, row in pool.map(build, arms):
            manifest["libraries"][arm] = row
            print("PPU17_REFERENCE_BUILD " + arm + " PASS", flush=True)
    if any(sha(path) != digest for path, digest in bound_files.items()):
        raise ValueError('source changed during compilation; no build authority published')
    dependency(args.vllm, VLLM)
    dependency(args.cutlass, CUTLASS)
    require_cutlass(args.ppu_cutlass)
    dependency(root/'third_party/actlize', ACTLIZE)
    manifest['config_counts'] = {'machete': 9, 'marlin': 3, 'quactlize': 19 if args.reader in ('prepacked','aligned') else 12}
    if args.reader == 'aligned': manifest['config_counts']['quactlize_marlin'] = 3
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
