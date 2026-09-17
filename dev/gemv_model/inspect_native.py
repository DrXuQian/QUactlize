#!/usr/bin/env python3
"""Check emitted candidates; keep this module distinct from stdlib inspect."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_model.plan import POINTS, candidates
from dev.gemv_model.run import verify
from tools.build_kpack_dequant import resource_usage
from quactlize.runtime.compiler import sha


def inspect(bundle):
    manifest = verify(bundle)
    results = {}
    for point in POINTS:
        isa = (bundle / (point.name + ".isa.txt")).read_text()
        resources = resource_usage((bundle / (point.name + ".resources.txt")).read_text())
        entries = list(re.finditer(r"Disassembly of section \.text\.kernel\.([^\n]+):", isa))
        values = {}
        for index, entry in enumerate(entries):
            symbol = entry[1]
            candidate = re.search(r"model_gemv\d+point_\d+_\d+_\d+_arm_(\d+)", symbol)
            if not candidate:
                continue
            arm = int(candidate[1])
            body = isa[entry.end():entries[index + 1].start() if index + 1 < len(entries) else None]
            ops = Counter(re.findall(r"\t([a-z][\w.]+)\s", body))
            if arm in values or symbol not in resources:
                raise ValueError("duplicate/missing candidate resource")
            if not any(op.startswith("v.fma.f32") for op in ops) or not ops["v.lop3.b32"]:
                raise ValueError("candidate lost fast code decode or FP32 FMA: " + symbol)
            if not any(op.startswith("vmem.ld.b32x") for op in ops):
                raise ValueError("candidate vector loads missing: " + symbol)
            values[arm] = dict(name=candidates(point)[arm].name, symbol=symbol, resources=resources[symbol],
                               operations=dict(sorted(ops.items())), scope="STATIC_ISA_NOT_DYNAMIC_COUNTERS")
        if set(values) != set(range(len(candidates(point)))):
            raise ValueError("compiled candidate set differs: " + point.name)
        results[point.name] = values
    output = bundle / "native-inspection.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    # This adds inspection evidence, never changes a compiled payload or key.
    manifest["payloads"][output.name] = sha(output)
    manifest["inspection_script_sha256"] = sha(__file__)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"MODEL_GEMV_ISA PASS points={len(results)} kernels={sum(map(len,results.values()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    inspect(parser.parse_args().bundle.resolve())
