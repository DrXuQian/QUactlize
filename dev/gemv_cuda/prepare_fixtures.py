#!/usr/bin/env python3
"""Export small, independent GGUF fixtures for remote CUDA diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from reference import gguf_kpack as ref
from tools.kpack_warmup_fixture import activation_values, prepare_expert


def export(output, q, n, k, experts, selected, channels):
    spec = ref.SPECS[q]
    category = np.random.default_rng(81811 + k).integers(0, 4, k, dtype=np.uint8)
    coefficients = activation_values(np.arange(channels))
    a = coefficients[:, category].astype("<f4")
    planes, golden, denom = {}, [], []
    for slot, e in enumerate(selected):
        rng = np.random.default_rng(np.random.SeedSequence([81923, q, n, k, e]))
        raw = rng.integers(0, 256, (n * (k // 256), spec.raw_bytes), dtype=np.uint8)
        for offset in (spec.d_offset, spec.dmin_offset):
            if offset >= 0:
                h = rng.uniform(0.005, 0.03, len(raw)).astype("<f2")
                raw[:, offset : offset + 2] = h.view("u1").reshape(-1, 2)
        placed = prepare_expert(raw, q, n, k)
        for name in ("low", "high", "units"):
            planes.setdefault(name, []).append(placed[name])
        official = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k)
        activation = a[slot % channels].astype("f8")
        golden.append(official.astype("f8") @ activation)
        denom.append(np.abs(official.astype("f8")) @ np.abs(activation))
    path = output / f"q{q}-n{n}-k{k}-e{experts}-c{channels}.npz"
    np.savez_compressed(
        path,
        **{name: np.stack(value) for name, value in planes.items()},
        a=a,
        golden=np.array(golden),
        denom=np.array(denom),
        ids=np.array(selected, dtype="i4"),
        q=q,
        n=n,
        k=k,
        experts=experts,
        channels=channels,
    )
    receipt = dict(
        path=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        q=q,
        n=n,
        k=k,
        experts=experts,
        selected=selected,
        channels=channels,
        oracle="OFFICIAL_GGUF_FP64_DOT",
        input_type="F32_FP16_EXACT",
    )
    print("CUDA_GEMV_FIXTURE " + json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [export(args.output, q, 256, 512, 1, [0], 1) for q in range(10, 15)]
    selected = list(range(0, 8 * 17, 17))
    cases += [
        export(args.output, 12, 512, 2048, 256, selected, 1),
        export(args.output, 13, 2048, 512, 256, selected, 8),
        export(args.output, 12, 4096, 2048, 1, [0], 1),
    ]
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
