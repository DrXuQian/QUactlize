#!/usr/bin/env python3
"""Exhaust the declared dense Marlin scheduler composition space.

The raw Cartesian product is generated from authorities, never transcribed:
three committed tactic tables, fixtures.py/workloads.py, the dense benchmark's
A0 defaults, and the decode shape in the Marlin box recipe.  L133 maps every
raw tuple through the production CUTLASS_HOST_DEVICE core, deduplicates only
after that mapping, and exhausts every production work segment.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.fixtures import fixtures  # noqa: E402
from benchmarks.workloads import MODELS  # noqa: E402

TABLES = (
    ROOT / "benchmarks/lowbit_dense_configs.inc",
    ROOT / "benchmarks/lowbit_dense_i2_configs.inc",
    ROOT / "benchmarks/lowbit_dense_i1_configs.inc",
)
BENCH = ROOT / "benchmarks/test_lowbit_dense_bench.cu"
BOX = ROOT / "tools/run_dense_marlin_box.sh"
CORE = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"
ORACLE = ROOT / "dev/fold_derivation/l133_marlin_exhaustive.cu"
RUNNER = ROOT / "dev/fold_derivation/run_l133_marlin_exhaustive.sh"
DOC = ROOT / "dev/fold_derivation/MARLIN_CORRECTNESS_PROOF.md"


def tactics() -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    pat = re.compile(r"\bX\((\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),")
    for path in TABLES:
        for m in pat.finditer(path.read_text()):
            rows.append(tuple(map(int, m.groups()[:3])))
    return rows


def declared_shapes() -> list[tuple[int, int, int]]:
    shapes = {
        (row[4], row[2], row[3])
        for _, cfg in MODELS.items()
        for row in fixtures(_, cfg)
        if row[0] == "dense"
    }
    # The two anchors come from the executable/recipe that consumes them.
    a0 = re.search(r"int m = (\d+), n = (\d+), k = (\d+);", BENCH.read_text())
    decode = re.search(r"--m=(\d+) --n=(\d+) --k=(\d+)", BOX.read_text())
    if not a0 or not decode:
        raise RuntimeError("cannot derive A0/decode anchors from production sources")
    shapes.add(tuple(map(int, a0.groups())))
    shapes.add(tuple(map(int, decode.groups())))
    return sorted(shapes)


def write_manifest(path: Path, rows: list[tuple[int, int, int]],
                   shapes: list[tuple[int, int, int]]) -> tuple[int, int]:
    deployment = 0
    cross_l = 0
    with path.open("w") as f:
        for m, n, k in shapes:
            for tm, tn, tk in rows:
                for cu in (32, 72):  # both PPU CU counts observed by the gates
                    f.write(f"D {m} {n} {k} 1 {tm} {tn} {tk} {cu}\n")
                    deployment += 1
        # Every committed tactic gets the same tile-count construction:
        # Mt=2,Nt=2,L=2,Kt=31,CU=9. It forces continuation across N/M/L.
        for tm, tn, tk in rows:
            f.write(f"X {2*tm} {2*tn} {31*tk} 2 {tm} {tn} {tk} 9\n")
            cross_l += 1
    return deployment, cross_l


def run_oracle(manifest: Path, override: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["QUACTLIZE_L133_OUT"] = str(manifest.parent / ("out-" + (override.name if override else "green")))
    if override is not None:
        env["L133_CORE_OVERRIDE"] = str(override)
    return subprocess.run(
        ["bash", str(RUNNER), str(manifest)], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> int:
    required = (*TABLES, BENCH, BOX, CORE, ORACLE, RUNNER, DOC)
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        print("[marlin-exhaustive] FAIL: missing " + ", ".join(missing))
        return 1
    rows = tactics()
    shapes = declared_shapes()
    if len(rows) != 4790 or len(shapes) != 68:
        print(f"[marlin-exhaustive] FAIL: authority cardinality drift rows={len(rows)} shapes={len(shapes)}")
        return 1

    with tempfile.TemporaryDirectory(prefix="quactlize-marlin-exhaustive-") as td:
        tmp = Path(td)
        manifest = tmp / "cases.tsv"
        deployment, cross_l = write_manifest(manifest, rows, shapes)
        if deployment != 651440 or cross_l != 4790:
            print(f"[marlin-exhaustive] FAIL: manifest cardinality {deployment}+{cross_l}")
            return 1
        green = run_oracle(manifest)
        required_output = (
            "raw deployment=651440 cross-L=4790 scanned=656230 remaining=0",
            "equivalence unique=2815 checked=2815 remaining=0 protected=2465 "
            "stripe-regime=350 actual-split=344 q<CU-unsplit=6 "
            "raw-protected/stripe/actual/q<CU-unsplit=435252/220978/218313/2665",
            "production segments=42231743 logical-(q,k)-cells=2632768288 "
            "outputs=42215890 handoffs=15853 cross(N/M/L)=4717/767/1",
            "fixture=marlin-cell-exact contribution={-1,0,1} max-terms=400 < 2048 "
            "ORDER-INDEPENDENT+FP16-EXACT criterion=raw-integer-equality-before-run",
            "proposition-A exact-once=>DP result=PASS raw-remaining=0 "
            "unique-remaining=0 failures=0",
        )
        if green.returncode != 0 or any(x not in green.stdout for x in required_output):
            print("[marlin-exhaustive] FAIL: positive oracle\n" + green.stdout[-2400:])
            return 1

        source = CORE.read_text()
        plants = (
            ("grid-times-occupancy",
             (("p.grid_blocks_ = p.output_tiles_ >= cu_count ? p.output_tiles_ : cu_count;",
               "p.grid_blocks_ = cu_count * 4;"),)),
            ("floor-stripe",
             (("p.iters_per_block_ = ceil_div_u64(p.total_k_tiles_, p.grid_blocks_);",
               "p.iters_per_block_ = p.total_k_tiles_ / p.grid_blocks_;"),)),
            ("no-q-boundary-clip",
             (("uint64_t const segment_end = stripe_end < q_end ? stripe_end : q_end;",
               "uint64_t const segment_end = stripe_end;"),)),
            ("N-fast-lost",
             (("uint64_t const q = cursor / p.k_tiles_per_output_;",
               "uint64_t const q = cursor % p.output_tiles_;"),
              ("uint64_t const k = cursor % p.k_tiles_per_output_;",
               "uint64_t const k = cursor / p.output_tiles_;"))),
            ("M-N-decode-swapped",
             (("out.N_idx = int32_t(q_mn % p.tiles_n_);",
               "out.N_idx = int32_t(q_mn % p.tiles_m_);"),)),
            ("fetch-stops-after-first-segment",
             (("work.is_valid() && work.linear_next < work.linear_end",
               "work.is_valid() && false"),)),
            ("N-local-lock",
             (("out.lock_idx = q;", "out.lock_idx = uint64_t(out.N_idx);"),)),
            ("natural-peer-order",
             (("out.slice_idx = uint32_t(last_peer - block_idx);",
               "out.slice_idx = uint32_t(block_idx - first_peer);"),)),
        )
        for label, replacements in plants:
            planted = source
            for old, new in replacements:
                if planted.count(old) != 1:
                    print(f"[marlin-exhaustive] FAIL: cannot plant {label}: {old!r}")
                    return 1
                planted = planted.replace(old, new, 1)
            override = tmp / label / "cutlass/gemm/kernel"
            override.mkdir(parents=True)
            (override / CORE.name).write_text(planted)
            red = run_oracle(manifest, override.parents[2])
            if red.returncode != 1 or "proposition-A exact-once=>DP result=FAIL" not in red.stdout:
                print(f"[marlin-exhaustive] FAIL: plant {label} did not red\n{red.stdout[-1800:]}")
                return 1

    print("[marlin-exhaustive] PASS: 656230/656230 raw tuples -> 2815/2815 production Params; "
          "344 real split classes, 2632768288 logical cells exact-once, remaining=0; "
          "8 compiled core plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
