"""Build diagnostic reference kernels; never link them into Quactlize."""
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compiler", type=Path, required=True)
    args = p.parse_args()
    args.vllm = args.vllm.resolve(strict=True)
    args.cutlass = args.cutlass.resolve(strict=True)
    args.compiler = args.compiler.resolve(strict=True)
    args.output = args.output.resolve()
    dependency(args.vllm, VLLM)
    dependency(args.cutlass, CUTLASS)
    if args.output == HERE or HERE in args.output.parents:
        raise ValueError("use a build root outside the source directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise ValueError("use a fresh build root; do not overwrite a reference authority")
    common = [str(args.compiler), "-std=c++17", "-O3", "-DNDEBUG",
              "--expt-relaxed-constexpr", "--extended-lambda",
              "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
              "-Xcompiler=-fPIC,-fvisibility=hidden", "-I" + str(HERE / "shim"),
              "-I" + str(args.vllm / "csrc"), "-I" + str(args.cutlass / "include")]
    sources = {str(f.relative_to(HERE)): sha(f) for f in sorted(HERE.rglob("*"))
               if f.is_file() and f.suffix in (".cu", ".cuh", ".h", ".py")}
    manifest = {"schema": 1, "scope": "PHYSICAL_HOPPER_REFERENCE_NOT_PRODUCT",
                "vllm": VLLM, "cutlass": CUTLASS, "sources": sources,
                "compiler": str(args.compiler), "compiler_sha256": sha(args.compiler),
                "compiler_version": subprocess.check_output([str(args.compiler), "--version"], text=True),
                "flags": common[1:], "libraries": {}}
    def build(arm):
        temp = args.output / (arm + "-tmp")
        temp.mkdir(exist_ok=True)
        library = args.output / ("lib" + arm + "_reference.so")
        command = common + (["-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1"] if arm == "machete" else [])
        command += ["-shared", str(HERE / (arm + ".cu")), "-o", str(library)]
        if arm == "machete":
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
    with ThreadPoolExecutor(max_workers=2) as pool:
        for arm, row in pool.map(build, ("marlin", "machete")):
            manifest["libraries"][arm] = row
            print("PPU17_REFERENCE_BUILD " + arm + " PASS", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
