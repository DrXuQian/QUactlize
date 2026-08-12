#!/usr/bin/env python3
"""Exhaust grouped Marlin over committed format tables and MoE fixtures."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks.fixtures import fixtures  # noqa: E402
from benchmarks.workloads import MODELS  # noqa: E402

TABLES = tuple(sorted((ROOT / "benchmarks").glob("lowbit_grouped*configs.inc")))
RUNNER = ROOT / "dev/fold_derivation/run_l136_grouped_marlin_exhaustive.sh"
ORACLE = ROOT / "dev/fold_derivation/l136_grouped_marlin_exhaustive.cu"
GEOMETRY = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_grouped_ragged_geometry.hpp"
CORE = ROOT / "third_party/actlize/include/cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"
KERNEL = ROOT / "quactlize/include/quactlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_group_marlin.hpp"
TYPES = ROOT / "dev/fold_derivation/l135_grouped_marlin_types.cu"


def tactics() -> Counter[tuple[int, int, int]]:
    out: Counter[tuple[int, int, int]] = Counter()
    pat = re.compile(r"\bX\((\d+),(\d+),(\d+),")
    for path in TABLES:
        for m in pat.finditer(path.read_text()):
            out[tuple(map(int, m.groups()))] += 1
    return out


def shapes() -> Counter[tuple[int, int, int, int, int]]:
    out: Counter[tuple[int, int, int, int, int]] = Counter()
    for model, cfg in MODELS.items():
        for kind, _, n, k, tokens, extra in fixtures(model, cfg):
            if kind == "moe":
                out[(n, k, tokens, extra["experts"], extra["topk"])] += 1
    return out


def write_manifest(path: Path, ts, ss) -> int:
    raw = 0
    with path.open("w") as f:
        for (n, k, tokens, experts, topk), shape_mult in sorted(ss.items()):
            for (tm, tn, tk), tactic_mult in sorted(ts.items()):
                mult = shape_mult * tactic_mult
                for cu in (32, 72):
                    f.write(f"{n} {k} {tokens} {experts} {topk} "
                            f"{tm} {tn} {tk} {cu} {mult}\n")
                    raw += mult
    return raw


def run(manifest: Path, *, geometry_override: str | None = None,
        core_override: str | None = None):
    env = dict(os.environ)
    env["QUACTLIZE_L136_OUT"] = str(manifest.parent / "out")
    includes: list[str] = []
    if geometry_override:
        includes.append(geometry_override)
    if core_override:
        includes.append(core_override)
    if includes:
        env["L136_INCLUDE_OVERRIDE"] = os.pathsep.join(includes)
    return subprocess.run(["bash", str(RUNNER), str(manifest)], cwd=ROOT,
                          env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


def main() -> int:
    required = (*TABLES, RUNNER, ORACLE, GEOMETRY, CORE, KERNEL, TYPES)
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        print("[grouped-marlin-exhaustive] FAIL: missing " + ", ".join(missing))
        return 1
    ts, ss = tactics(), shapes()
    if sum(ts.values()) != 9138 or len(ts) != 85 or sum(ss.values()) != 36 or len(ss) != 24:
        print(f"[grouped-marlin-exhaustive] FAIL: authority drift tactics={sum(ts.values())}/{len(ts)} shapes={sum(ss.values())}/{len(ss)}")
        return 1
    with tempfile.TemporaryDirectory(prefix="quactlize-grouped-marlin-") as td:
        tmp = Path(td)
        manifest = tmp / "manifest.tsv"
        raw = write_manifest(manifest, ts, ss)
        if raw != 657936:
            print(f"[grouped-marlin-exhaustive] FAIL: raw manifest={raw}")
            return 1
        green = run(manifest)
        must = (
            "raw=657936/657936 remaining=0",
            "criterion=raw-integer-equality-before-run fixture=grouped-marlin-cell-exact ORDER-INDEPENDENT+FP16-EXACT",
            "Q>=CU classes=",
            "handoffs=0 criterion=",
            "proposition-A grouped-ragged exact-once=>DP result=PASS failures=0",
        )
        if green.returncode != 0 or any(x not in green.stdout for x in must):
            print("[grouped-marlin-exhaustive] FAIL: positive oracle\n" + green.stdout[-3000:])
            return 1

        geometry_source = GEOMETRY.read_text()
        geometry_plants = (
            ("zero-expert-consumes-q", "out = mt * nt;", "out = (mt == 0 ? 1 : mt * nt);"),
            ("lower-bound-decode", "if (prefix[mid + 1] <= q)", "if (prefix[mid + 1] < q)"),
            ("expert-local-lock-prefix", "return lo < groups ? lo : -1;", "return lo < groups ? (lo == 0 ? lo : lo - 1) : -1;"),
        )
        for label, old, new in geometry_plants:
            if geometry_source.count(old) != 1:
                print(f"[grouped-marlin-exhaustive] FAIL: cannot plant {label}")
                return 1
            root = tmp / label
            target = root / "quactlize_extensions/cutlass/gemm/kernel"
            target.mkdir(parents=True)
            (target / GEOMETRY.name).write_text(geometry_source.replace(old, new, 1))
            red = run(manifest, geometry_override=str(root))
            if red.returncode != 1 or "result=FAIL" not in red.stdout:
                print(f"[grouped-marlin-exhaustive] FAIL: plant {label} stayed green\n{red.stdout[-1800:]}")
                return 1

        core_source = CORE.read_text()
        core_plants = (
            ("drop-output-tile-floor",
             "p.grid_blocks_ = p.output_tiles_ >= launch_capacity\n"
             "        ? p.output_tiles_ : launch_capacity;",
             "p.grid_blocks_ = launch_capacity;"),
            ("floor-iters",
             "p.iters_per_block_ = ceil_div_u64(p.total_k_tiles_, p.grid_blocks_);",
             "p.iters_per_block_ = p.total_k_tiles_ / p.grid_blocks_;"),
            ("local-lock", "out.lock_idx = q;", "out.lock_idx = uint64_t(out.N_idx);"),
        )
        for label, old, new in core_plants:
            if core_source.count(old) != 1:
                print(f"[grouped-marlin-exhaustive] FAIL: cannot plant {label}")
                return 1
            root = tmp / label
            target = root / "cutlass/gemm/kernel"
            target.mkdir(parents=True)
            (target / CORE.name).write_text(core_source.replace(old, new, 1))
            red = run(manifest, core_override=str(root))
            if red.returncode != 1 or "result=FAIL" not in red.stdout:
                print(f"[grouped-marlin-exhaustive] FAIL: plant {label} stayed green\n{red.stdout[-1800:]}")
                return 1

    print("[grouped-marlin-exhaustive] PASS: 657936/657936 committed format×MoE tuples exhausted; zero experts omitted, global-q locks unique, Q>=CU handoffs zero; six plants rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
