#!/usr/bin/env python3
"""Exhaust the host-only GEMV tactic authority and pin it to production seams."""

from __future__ import annotations

import csv
import io
import re
import subprocess
import tempfile
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "quactlize/include/gemv_lowbit/gemv_tactic_space.hpp"
EMITTER = ROOT / "dev/fold_derivation/emit_gemv_tactic_space.cpp"
REGISTRY = ROOT / "quactlize/include/ppu_format_config.inc"
BACKEND = ROOT / "quactlize/csrc/device/ppu_backend.cu"
DETAILS = ROOT / "quactlize/include/gemv_lowbit/gemv_details.hpp"
WFORMAT = ROOT / "quactlize/include/gemv_lowbit/gemv_wformat.hpp"
KERNEL = ROOT / "quactlize/include/gemv_lowbit/gemv_kernel.hpp"
LAUNCHER = ROOT / "quactlize/include/gemv_lowbit/gemv_launcher.hpp"

FORMATS = ("int4", "int2", "int1", "q3", "q6")
LAYOUT_TILES = (("native", 0), ("tileK", 256))
STEP_K = (8, 16, 32)
THREADS = (64, 128, 256)
DENSE_M = tuple(range(1, 16))
GROUPED_M = tuple(range(1, 5))
CTA_N = (2, 4, 8, 16)
CHUNK = (2, 4, 8, 16)
EXPECTED_TOTAL = 27_360
EXPECTED_LEGAL = 10_260
EXPECTED_EXCLUSIONS = Counter({
    "STEP_TOO_SMALL_FOR_SPARSEST_PLANE": 10_944,
    "CTA_N_NOT_WHOLE_CHUNKS": 6_156,
})


def build_run(include_first: Path | None = None, arg: str | None = None) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="qz-gemv-space-bin-") as td:
        exe = Path(td) / "emit"
        cmd = ["g++", "-std=c++17"]
        if include_first is not None:
            cmd += ["-I", str(include_first)]
        cmd += ["-I", str(ROOT / "quactlize/include"), str(EMITTER), "-o", str(exe)]
        build = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        if build.returncode:
            return build
        run_cmd = [str(exe)] + ([arg] if arg else [])
        return subprocess.run(run_cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)


def parse_summary(text: str) -> tuple[dict[str, int], Counter[str], list[tuple[str, ...]]]:
    census: dict[str, int] = {}
    exclusions: Counter[str] = Counter()
    anchors: list[tuple[str, ...]] = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if row[0] == "CENSUS":
            census[row[1]] = int(row[2])
        elif row[0] == "EXCLUSION":
            exclusions[row[1]] = int(row[2])
        elif row[0] == "ANCHOR":
            anchors.append(tuple(row[1:]))
    return census, exclusions, anchors


def expected_cells() -> set[tuple[str, str, int, int, int, str, int, int, int]]:
    cells = set()
    for fmt, (layout, tk), sk, th, cn, ch in product(
            FORMATS, LAYOUT_TILES, STEP_K, THREADS, CTA_N, CHUNK):
        cells.update((fmt, layout, tk, sk, th, "dense", cm, cn, ch) for cm in DENSE_M)
        cells.update((fmt, layout, tk, sk, th, "grouped", cm, cn, ch) for cm in GROUPED_M)
    return cells


def production_constraints_present() -> list[str]:
    checks = {
        DETAILS: (
            "kStepK * kMinBits >= 32",
            "kStepK % (32 / kLoBits) == 0",
            "kStepK % (32 / (kHiBits ? kHiBits : 1)) == 0",
        ),
        WFORMAT: (
            "TileSizeK % StepK == 0",
            "Threads % (TileSizeK / StepK) == 0",
        ),
        KERNEL: (
            "CtaN % Chunk == 0",
            "Chunk % 2 == 0 && Chunk >= 2",
        ),
    }
    missing = []
    for path, needles in checks.items():
        text = path.read_text()
        missing += [f"{path.name}:{needle}" for needle in needles if needle not in text]
    return missing


def registry_groups() -> dict[int, int]:
    text = REGISTRY.read_text()
    rows = re.findall(
        r"X\([^,]+,\s*\"[^\"]+\",\s*(\d+),\s*\d+,\s*\d+,\s*(\d+),",
        text,
    )
    return {int(qtype): int(gs) for qtype, gs in rows}


def backend_defaults() -> dict[int, tuple[int, int]]:
    text = BACKEND.read_text()
    # Use the public config-valid switch.  It is the production admission path,
    # while tests and perf tiers are deliberately broader.
    region = text[text.index('extern "C" int32_t quactlize_ppu_gemv_lowbit_config_valid_v1'):]
    region = region[:region.index('extern "C" int32_t quactlize_ppu_vecdot_moe_config_valid_v1')]
    rows = re.findall(
        r"case\s+(\d+):\s+return\s+lowbit_dense_config_valid<[^,]+,\s*(\d+),\s*(\d+)>",
        region,
    )
    return {int(q): (int(sk), int(th)) for q, sk, th in rows}


def backend_geometry() -> tuple[str, int, int]:
    text = BACKEND.read_text()
    fn = text[text.index("int lowbit_device("):]
    fn = fn[:fn.index("template <ppu_gemv::WFormat F, int StepK, int Threads>\nint lowbit(")]
    layout = re.search(r"KernelDetails<[^,]+,\s*F,\s*ppu_gemv::WLayout::(\w+)", fn)
    launch = re.search(r"launch_gemv<D,\s*(\d+),\s*(\d+)>", fn)
    if not layout or not launch:
        raise RuntimeError("cannot read production lowbit_device layout/CtaN/Chunk")
    return layout.group(1), int(launch.group(1)), int(launch.group(2))


def production_ctam_limits() -> tuple[int, int]:
    text = LAUNCHER.read_text()
    dense = re.search(r"#define GEMV_CTAM_MAX\s+(\d+)", text)
    grouped = re.search(r"#define GEMV_GROUPED_CTAM_MAX\s+(\d+)", text)
    if not dense or not grouped:
        raise RuntimeError("cannot read production dense/grouped CtaM limits")
    return int(dense.group(1)), int(grouped.group(1))


def audit_output(text: str, expect_rows: bool) -> list[str]:
    bad: list[str] = []
    census, exclusions, anchors = parse_summary(text)
    if census.get("total") != EXPECTED_TOTAL:
        bad.append(f"total={census.get('total')} want={EXPECTED_TOTAL}")
    if census.get("legal") != EXPECTED_LEGAL:
        bad.append(f"legal={census.get('legal')} want={EXPECTED_LEGAL}")
    if census.get("legal", 0) + census.get("rejected", 0) != EXPECTED_TOTAL:
        bad.append("legal + rejected does not conserve the Cartesian domain")
    if sum(exclusions.values()) != census.get("rejected"):
        bad.append("reason histogram does not sum to rejected")
    if exclusions != EXPECTED_EXCLUSIONS:
        bad.append(f"reason histogram={dict(exclusions)} want={dict(EXPECTED_EXCLUSIONS)}")
    if "RESULT,PASS" not in text:
        bad.append("emitter did not report PASS")

    groups, defaults = registry_groups(), backend_defaults()
    backend_layout, backend_cta_n, backend_chunk = backend_geometry()
    dense_ctam, grouped_ctam = production_ctam_limits()
    if (backend_layout, backend_cta_n, backend_chunk) != ("Native", 8, 2):
        bad.append(
            "production launch geometry="
            f"{(backend_layout, backend_cta_n, backend_chunk)} want=('Native', 8, 2)"
        )
    if (dense_ctam, grouped_ctam) != (max(DENSE_M), max(GROUPED_M)):
        bad.append(
            f"production CtaM limits={(dense_ctam, grouped_ctam)} "
            f"axis={(max(DENSE_M), max(GROUPED_M))}"
        )
    got_anchor: dict[int, tuple[str, int, int, int, int]] = {}
    int1_unshipped = False
    for a in anchors:
        if a == ("int1", "NOT_SHIPPED"):
            int1_unshipped = True
        elif len(a) == 9:
            fmt, qtype, gs, layout, tk, sk, th, cn, ch = a
            if layout != "native" or int(tk) != 0 or int(cn) != 8 or int(ch) != 2:
                bad.append(f"anchor geometry drift: {a}")
            got_anchor[int(qtype)] = (fmt, int(gs), int(sk), int(th), int(tk))
    expected_qtypes = {10, 11, 12, 14}
    if set(got_anchor) != expected_qtypes or not int1_unshipped:
        bad.append(f"shipping anchor domain={sorted(got_anchor)}, int1_unshipped={int1_unshipped}")
    for qtype in expected_qtypes:
        if qtype not in groups or qtype not in defaults or qtype not in got_anchor:
            continue
        _, gs, sk, th, _ = got_anchor[qtype]
        if gs != groups[qtype] or (sk, th) != defaults[qtype]:
            bad.append(
                f"qtype {qtype} anchor={(gs, sk, th)} registry/backend={(groups[qtype], *defaults[qtype])}"
            )

    if expect_rows:
        rows = []
        for row in csv.reader(io.StringIO(text)):
            if row and row[0] == "ROW":
                rows.append((row[1], row[2], int(row[3]), int(row[4]), int(row[5]),
                             row[6], int(row[7]), int(row[8]), int(row[9])))
                if (row[10] == "1") != (row[11] == "NONE"):
                    bad.append(f"row legal/reason disagreement: {row}")
                    break
        expected = expected_cells()
        if len(rows) != EXPECTED_TOTAL:
            bad.append(f"row count={len(rows)} want={EXPECTED_TOTAL}")
        if len(set(rows)) != len(rows):
            bad.append(f"duplicate Cartesian cells={len(rows) - len(set(rows))}")
        got = set(rows)
        if got != expected:
            bad.append(f"Cartesian cell difference missing={len(expected-got)} extra={len(got-expected)}")
    return bad


def planted_header(old: str, new: str) -> str:
    source = HEADER.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"plant seam count={source.count(old)} for {old!r}")
    return source.replace(old, new, 1)


def run_plant(old: str, new: str) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="qz-gemv-space-plant-") as td:
        inc = Path(td) / "gemv_lowbit"
        inc.mkdir()
        (inc / HEADER.name).write_text(planted_header(old, new))
        run = build_run(Path(td))
    if run.returncode:
        # A compile-time authority assertion rejecting the plant is a valid red.
        return ["compile-red"]
    return audit_output(run.stdout, False)


def main() -> int:
    required = (HEADER, EMITTER, REGISTRY, BACKEND, DETAILS, WFORMAT, KERNEL, LAUNCHER)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("[gemv-tactic-space] FAIL missing: " + ", ".join(missing))
        return 1
    missing_constraints = production_constraints_present()
    if missing_constraints:
        print("[gemv-tactic-space] FAIL production constraint drift: " + ", ".join(missing_constraints))
        return 1

    summary = build_run()
    if summary.returncode:
        print("[gemv-tactic-space] FAIL build/summary:\n" + summary.stdout)
        return 1
    bad = audit_output(summary.stdout, False)
    rows = build_run(arg="--rows")
    if rows.returncode:
        print("[gemv-tactic-space] FAIL rows:\n" + rows.stdout)
        return 1
    bad += audit_output(rows.stdout, True)
    if bad:
        print("[gemv-tactic-space] FAIL: " + "; ".join(bad[:12]))
        return 1

    # Three source-level plants: shrink an axis, weaken a legality rule, and
    # drift a production anchor.  Each must be rejected by compilation or by
    # the independent Cartesian/census/production audit above.
    plants = (
        ("inline constexpr std::array<int, 4> kCtaNs{{2, 4, 8, 16}};",
         "inline constexpr std::array<int, 3> kCtaNs{{2, 4, 8}};", "axis"),
        ("if (c.step_k * min_bits < 32) return Exclusion::StepTooSmallForSparsestPlane;",
         "if (c.step_k * min_bits < 16) return Exclusion::StepTooSmallForSparsestPlane;", "legality"),
        ("{Format::Q3,   11, ppu_formats::for_qtype(11).group_size, 32,  64, 8, 2},",
         "{Format::Q3,   11, ppu_formats::for_qtype(11).group_size, 16,  64, 8, 2},", "anchor"),
    )
    plant_results = []
    for old, new, label in plants:
        red = run_plant(old, new)
        if not red:
            print(f"[gemv-tactic-space] FAIL {label} plant escaped")
            return 1
        plant_results.append(label)

    census, exclusions, _ = parse_summary(summary.stdout)
    reasons = ",".join(f"{k}={v}" for k, v in sorted(exclusions.items()))
    print(f"[gemv-tactic-space] PASS total={census['total']} legal={census['legal']} "
          f"rejected={census['rejected']} reasons=[{reasons}] rows=unique+complete "
          f"anchors=registry+ppu_backend plants={','.join(plant_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
