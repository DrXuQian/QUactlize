"""Host, ABI, target-negative and live SM90a-code gates; no GPU execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from build import HERE, ROOT, QTYPES, sha


def entries(ptx):
    matches = list(re.finditer(r"(?:\.visible|\.weak)\s+\.entry\s+(\w+)", ptx))
    result = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ptx)
        result[match.group(1)] = ptx[match.end():end]
    return result


def inspect(ptx, q):
    if ".target sm_90a" not in ptx:
        raise ValueError("wrong emitted architecture")
    bodies = entries(ptx)
    families = {"wgmma": 0, "tma_wgmma": 0, "simt": 0, "expand_scales": 0}
    for name, body in bodies.items():
        family = next((f for f in ("tma_wgmma", "wgmma", "simt", "expand_scales") if f in name), None)
        if not family or f"ILi{q}E" not in name:
            raise ValueError("unexpected kernel entry: " + name)
        families[family] += 1
        if "st.global" not in body:
            raise ValueError("missing live output in " + name)
        if family in ("wgmma", "tma_wgmma"):
            dtype = "bf16" if "bfloat16" in name else "f16"
            if not re.search(r"wgmma\.mma_async[^;\n]*\.f32\." + dtype + r"\." + dtype, body):
                raise ValueError("missing typed FP32-accumulating GMMA in " + name)
            if "wgmma.wait_group" not in body:
                raise ValueError("missing async completion wait")
        if family == "tma_wgmma" and "cp.async.bulk.tensor" not in body:
            raise ValueError("missing TMA in " + name)
        if family == "simt" and "fma.rn.f32" not in body:
            raise ValueError("missing FP32 SIMT multiply in " + name)
        if re.search(r"\bppu\.[a-z]", body):
            raise ValueError("legacy PPU assembly leaked into SM90a source target")
    expected = {"wgmma": 12, "tma_wgmma": 4, "simt": 8, "expand_scales": 2}
    if families != expected:
        raise ValueError(f"incomplete specialization coverage: {families}")
    return families


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    if args.jobs < 1: p.error("jobs must be positive")
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["target"] != "cuda-source":
        raise ValueError("this gate inspects CUDA SM90a; native admission requires the matching native inspector")
    for name, digest in manifest["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("stale compiled source: " + name)
    if sha(bundle / manifest["library"]) != manifest["library_sha256"]:
        raise ValueError("library differs from manifest")
    out = args.output.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError("use an output directory outside the checkout")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "compiler-tmp"
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(tmp)}
    compiler = manifest["compiler"]
    flags = manifest["flags"]
    command = [compiler, *flags]

    def run(cmd, label, success=True):
        log = out / (label + ".log")
        with log.open("w") as f:
            f.write(json.dumps(cmd) + "\n"); f.flush()
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if (result.returncode == 0) != success:
            raise ValueError(f"unexpected exit {result.returncode}: {log}")
        return log.read_text()

    run(command + [str(ROOT / "tests/ppu17/contract.cu"), str(HERE / "host.cpp"),
                   "-o", str(out / "contract")], "host-build")
    contract_log = run([str(out / "contract")], "host-contract")
    if "PPU17_HOST_CONTRACT PASS" not in contract_log: raise ValueError("host contract did not finish")
    run(command + [str(ROOT / "tests/ppu17/single_launch.cu"), "-L" + str(bundle),
                   "-lquactlize_ppu17", "-Xlinker=-rpath=" + str(bundle),
                   "-o", str(out / "single-launch")], "application-build")
    for q in QTYPES:
        for dtype in (0, 1):
            run([str(out / "single-launch"), "--cpu-only", "--qtype", str(q), "--compute", str(dtype),
                 "--input", str(dtype), "--grouped"], f"cpu-fixture-q{q}-t{dtype}")

    old_flags = [f for f in flags if not f.startswith("-gencode=")]
    negative = run([compiler, *old_flags, "-gencode=arch=compute_80,code=sm_80", "--ptx",
                    str(HERE / "probe.cu"), "-o", str(out / "wrong-target.ptx")], "wrong-target", False)
    if "compiler-provided SM90A" not in negative: raise ValueError("wrong-target negative failed for another reason")
    negative = run(command + ["-DQ17_NATIVE=1", "-DACOMPUTE_VERSION=10700", "--ptx",
                    str(HERE / "probe.cu"), "-o", str(out / "fake-native.ptx")], "fake-native", False)
    if "requires the PPU compiler" not in negative: raise ValueError("native provenance negative failed for another reason")

    def codegen(q):
        ptx = out / f"q{q}.ptx"
        run(command + [f"-DQ17_QTYPE={q}", "--ptx", str(HERE / "module.cu"), "-o", str(ptx)], f"ptx-q{q}")
        text = ptx.read_text()
        families = inspect(text, q)
        # Plant one missing body, one empty compute family, one absent output.
        for tag, changed in (
            ("missing-entry", text.replace(".entry", ".not_an_entry", 1)),
            ("no-gmma", text.replace("wgmma.mma_async", "missing_gmma")),
            ("no-output", text.replace("st.global", "missing_store")),
        ):
            try: inspect(changed, q)
            except ValueError: pass
            else: raise ValueError("negative not caught: " + tag)
        inspector = Path(compiler).parent / "cuobjdump"
        sass = run([str(inspector), "--dump-sass", str(bundle / f"q{q}.o")], f"sass-q{q}")
        if "arch = sm_90a" not in sass or "HGMMA" not in sass:
            raise ValueError("no assembled SM90a GMMA")
        deps = run(command + [f"-DQ17_QTYPE={q}", "-M", str(HERE / "module.cu")], f"includes-q{q}")
        if "/third_party/actlize/" in deps or "/actlize_extensions/" in deps or "/third_party/cutlass/" in deps:
            raise ValueError("mixed legacy/new include graph")
        return {"qtype": q, "families": families, "ptx_sha256": sha(ptx), "assembled": True}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(codegen, QTYPES))
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(bundle / manifest["library"])], text=True)
    names = [line.split()[-1] for line in symbols.splitlines() if line.strip()]
    if not names or any(not n.startswith("q17_") for n in names):
        raise ValueError("C++ symbols exported across the backend boundary")
    sys.path.insert(0, str(ROOT))
    from quactlize.backends.ppu17 import Backend
    try: Backend(bundle)
    except ValueError: pass
    else: raise ValueError("source-only bundle admitted as native")
    backend = Backend(bundle, allow_source_validation=True)
    problem = backend.problem(qtype=12, m=1, n=256, k=512, experts=1, max_rows=1,
                              compute_type=1, input_type=1, output_type=2)
    if len(backend.configs(problem)) != 6: raise ValueError("Python/C ABI differs")
    result = {"status": "PASS", "scope": "HOST_AND_CUDA_SM90A_CODEGEN_ONLY", "bundle": str(bundle),
              "library_sha256": manifest["library_sha256"], "specializations": rows,
              "host_contract": contract_log.splitlines()[-1], "single_launch_application": str(out / "single-launch"),
              "target_negatives": 2, "codegen_negatives": 18, "exports": names,
              "device_launches": 0, "native_ppu17": "NOT_RUN", "device_numerics": "NOT_RUN", "performance": "NOT_RUN"}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PPU17_CHECK PASS specializations=156 host_and_abi=PASS device_admission=PENDING output=" + str(out))


if __name__ == "__main__":
    main()
