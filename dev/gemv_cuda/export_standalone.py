#!/usr/bin/env python3
"""Export existing independent NPZ fixtures for a CUDA-only profiler runner."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reference import gguf_kpack as ref


def export(source, output):
    with np.load(source, allow_pickle=False) as f:
        q, n, k, experts, channels = (int(f[x]) for x in ("q", "n", "k", "experts", "channels"))
        ids = np.ascontiguousarray(f["ids"], dtype="<i4")
        rows = len(ids)
        if len(set(ids.tolist())) != rows or np.any(ids < 0) or np.any(ids >= experts):
            raise ValueError("fixture must contain distinct in-range expert IDs")
        planes = [np.ascontiguousarray(f[x]).tobytes() for x in ("raw", "low", "high", "units")]
        arrays = [np.ascontiguousarray(f[x], dtype=d).tobytes()
                  for x, d in (("a", "<f4"), ("ids", "<i4"), ("golden", "<f8"), ("denom", "<f8"))]
        if q == 8:
            arrangement = (2, 4, 8, 0, 0, 32, 32, 0, 0x51384B5032540001)
        else:
            a = asdict(ref.canonical_arrangement(q))
            arrangement = tuple(a[x] for x in ("version", "layout", "bits", "high_bits", "artifact_tile_k",
                                                "transport_tile_k", "group_size", "reserved", "mapping_id"))
        chunks = planes + arrays
        header = struct.pack("<Q8i8iQ8Q", 0x3146584D5647514B,
                             1, q, n, k, experts, rows, channels, 0, *arrangement,
                             *(len(x) for x in chunks))
        with output.open("xb") as stream:
            stream.write(header)
            for chunk in chunks:
                stream.write(chunk)
    return dict(path=output.name, source=source.name,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                q=q, n=n, k=k, experts=experts, rows=rows, channels=channels,
                oracle="EXISTING_INDEPENDENT_GGUF_DOT", selected_ids=ids.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = [export(p, args.output / (p.stem + ".bin")) for p in sorted(args.fixtures.glob("*.npz"))]
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"GEMV_STANDALONE_FIXTURES count={len(records)} output={args.output}")
