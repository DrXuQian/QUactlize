#!/usr/bin/env python3
"""Prove the production GEMV manifest/exact-CtaM/raw-event seams are one graph."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.sweep_gemv_perf import ContractError, validate_manifest  # noqa: E402

ORACLE = ROOT / "dev/fold_derivation/l145_gemv_sweep_plan.cpp"
MAIN = ROOT / "benchmarks/test_gemv_perf.cu"
COMMON = ROOT / "benchmarks/gemv_perf_common.hpp"
UNIT = ROOT / "benchmarks/gemv_perf_unit.inc"
CMAKE = ROOT / "quactlize/csrc/CMakeLists.txt.in"
RUNNER = ROOT / "tools/run_gemv_sweep_box.sh"


def compile_oracle(source: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(prefix="quactlize-l145-") as td:
        exe = Path(td) / "l145"
        build = subprocess.run(
            ["g++", "-std=c++17", "-I", str(ROOT), "-I", str(ROOT / "quactlize/include"),
             str(source), "-o", str(exe)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            return build.returncode, build.stdout
        outputs = []
        for argv in ([], ["i4-native"]):
            run = subprocess.run([str(exe), *argv], cwd=ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if run.returncode:
                return run.returncode, run.stdout
            try:
                plan = json.loads(run.stdout)
                validate_manifest(plan)
            except (json.JSONDecodeError, ContractError) as exc:
                return 1, run.stdout + f"\nmanifest rejected: {exc}\n"
            outputs.append(plan)
        full, partial = outputs
        if full["partial_space"] or len(full["jobs"]) != 86:
            return 1, f"full manifest identity/count drift: {full['partial_space']=} jobs={len(full['jobs'])}\n"
        if not partial["partial_space"] or len(partial["jobs"]) != 18:
            return 1, f"partial manifest identity/count drift: {partial['partial_space']=} jobs={len(partial['jobs'])}\n"
        if any(job["formats"] != ["int4"] for job in partial["jobs"]):
            return 1, "i4-native partial plan leaked another format\n"
        return 0, (f"full={full['counts']} jobs=86; "
                   f"i4-native={partial['counts']} jobs=18")


def structural() -> list[str]:
    bad: list[str] = []
    texts = {p: p.read_text() for p in (MAIN, COMMON, UNIT, CMAKE, RUNNER)}
    required = {
        MAIN: ("gemv_perf_plan::manifest_json(", "--shape-case=", "GEMV_SWEEP_JOB_ID",
               "gemv_run_all(sh, bests.data(), sweep)", "parsed != 20",
               "GEMV_SWEEP_SAMPLES must be exactly 20"),
        COMMON: ("launch_gemv_exact_ctam<Details, Grouped, CtaM, CtaN, Chunk>",
                 "measure_raw_launches(go, sweep.measured_launches)",
                 "gemv_perf_manifest::config_json(candidate)",
                 "write_excluded(attempt, \"output witness mismatch\")"),
        UNIT: ("GEMV_GROUPED_CTAM_MAX", "GEMV_CTAM_MAX", "UNIT_ARTIFACT_TK",
               "UNIT_AUTH_FORMAT", "UNIT_AUTH_LAYOUT"),
        CMAKE: ("GemvCompiledGroup gemv_compiled_groups[]", "UNIT_AUTH_FORMAT",
                "UNIT_AUTH_LAYOUT", "sh.format == ppu_gemv::tactic_space::Format::"),
        RUNNER: ("git status --porcelain=v1 --untracked-files=all", "JOBS=\"$CORES\"",
                 "BIN_SHA=$(sha256sum", "protocol:$PROTOCOL", "flock -n 9"),
    }
    for path, tokens in required.items():
        for token in tokens:
            if token not in texts[path]:
                bad.append(f"{path.relative_to(ROOT)} lost {token!r}")
    for forbidden in ("launch_gemv<Details, CtaN, Chunk>(p, 0)", "time_it(go, 100)"):
        if forbidden in texts[COMMON]:
            bad.append(f"adaptive/wall-clock sweep path returned: {forbidden}")
    return bad


def main() -> int:
    missing = [p for p in (ORACLE, MAIN, COMMON, UNIT, CMAKE, RUNNER) if not p.is_file()]
    if missing:
        print("[gemv-sweep-integration] FAIL missing: " + ", ".join(map(str, missing)))
        return 1
    bad = structural()
    if bad:
        print("[gemv-sweep-integration] FAIL: " + "; ".join(bad))
        return 1
    rc, output = compile_oracle(ORACLE)
    if rc:
        print("[gemv-sweep-integration] FAIL oracle:\n" + output)
        return 1

    # Native's artifact identity must be zero.  Planting TileK=256 makes every
    # native candidate static-illegal and the shared production planner must red.
    source = ORACLE.read_text()
    old = '{"i4-native", F::Int4, L::Native, 0}'
    if source.count(old) != 1:
        print("[gemv-sweep-integration] FAIL cannot plant native TileK identity")
        return 1
    with tempfile.TemporaryDirectory(prefix="quactlize-l145-plant-") as td:
        planted = Path(td) / ORACLE.name
        planted.write_text(source.replace(old, '{"i4-native", F::Int4, L::Native, 256}', 1))
        red_rc, red_out = compile_oracle(planted)
    if red_rc == 0 or "no legal candidates" not in red_out:
        print("[gemv-sweep-integration] FAIL native TileK plant did not red:\n" + red_out)
        return 1
    print("[gemv-sweep-integration] PASS " + output +
          "; exact-CtaM/raw-event seams present; native TileK identity plant red")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
