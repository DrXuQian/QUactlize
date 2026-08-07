#!/usr/bin/env python3
"""Exercise every advertised MoE sweep restriction against the real CMake generator."""
from __future__ import annotations

import concurrent.futures
import os
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = ROOT / "dev/fold_derivation/gen_moe_units_check.sh"

POSITIVE = {
    "default": {},
    "format": {"MOE_FORMATS": "i4"},
    "tile-m": {"MOE_TM_LIST": "16"},
    "tile-n": {"MOE_TN_LIST": "32"},
    "warp-m": {"MOE_WM_LIST": "16"},
    "stages": {"MOE_STAGES": "12"},
}
NEGATIVE_CONTROLS = {
    "tile-filter-bypass": {"BAD": "5", "MOE_TM_LIST": "16"},
    "stage-device-flag-drop": {"BAD": "6", "MOE_STAGES": "12"},
}
LEGACY_POSITIVE = {
    "legacy-stage": {"PPU_DEFS": "MOE_STAGES_4"},
}
INVALID = {
    "partial-format-typo": {"MOE_FORMATS": "i4;typo"},
    "bad-tile-m": {"MOE_TM_LIST": "17"},
    "bad-tile-n": {"MOE_TN_LIST": "17"},
    "bad-warp-m": {"MOE_WM_LIST": "17"},
    "bad-stage": {"MOE_STAGES": "5"},
}
EXPECTED_FAILURES = {
    "legacy-stage-conflict": (
        {"PPU_DEFS": "MOE_STAGES_4", "MOE_STAGES": "2"},
        "both select the stage axis",
    ),
    "unknown-legacy-stage": (
        {"PPU_DEFS": "MOE_STAGES_5"},
        "unknown legacy stage selector",
    ),
    "global-filter-empties-decode": (
        {"MOE_TM_LIST": "64"},
        "decode sweep produced no legal shapes",
    ),
    "zero-cores-before-use": (
        {"MOE_CHECK_CORES": "0"},
        "MOE_CORES must be a positive integer",
    ),
}


def run_case(item):
    name, extra = item
    env = os.environ.copy()
    for key in ("BAD", "PPU_DEFS", "MOE_CHECK_CORES", "MOE_FORMATS", "MOE_TM_LIST", "MOE_TN_LIST",
                "MOE_WM_LIST", "MOE_STAGES"):
        env.pop(key, None)
    env.update(extra)
    result = subprocess.run(["bash", str(GEN)], cwd=ROOT, env=env, capture_output=True, text=True)
    return name, result.returncode, result.stdout + result.stderr


def counts(log: str):
    found = {}
    for band in ("full", "decode"):
        match = re.search(rf"-- OK {band}: ([0-9]+) shapes", log)
        if not match:
            raise ValueError(f"no independently-checked {band} shape count")
        found[band] = int(match.group(1))
    return found


def main() -> int:
    if not GEN.is_file():
        print(f"[moe-build-knobs] ERROR: {GEN.relative_to(ROOT)} is missing")
        return 1
    expected_failure_envs = {name: spec[0] for name, spec in EXPECTED_FAILURES.items()}
    work = (list(POSITIVE.items()) + list(LEGACY_POSITIVE.items()) + list(NEGATIVE_CONTROLS.items())
            + list(INVALID.items()) + list(expected_failure_envs.items()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict((name, (rc, log)) for name, rc, log in pool.map(run_case, work))

    failures = []
    observed = {}
    for name in POSITIVE:
        rc, log = results[name]
        if rc != 0:
            failures.append(f"{name}: expected pass, exit={rc}: {log.strip().splitlines()[-1:]}")
            continue
        try:
            observed[name] = counts(log)
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
    if "default" in observed:
        base = observed["default"]
        for name, got in observed.items():
            if name == "default":
                continue
            if not any(got[band] < base[band] for band in ("full", "decode")):
                failures.append(f"{name}: restriction did not shrink either band: {got} vs {base}")

    for name in NEGATIVE_CONTROLS:
        rc, log = results[name]
        if rc != 0 or "the generator was REJECTED" not in log:
            failures.append(f"{name}: planted defect was not cleanly rejected (exit={rc})")
    for name in INVALID:
        rc, log = results[name]
        if rc == 0 or "expected a semicolon-separated subset" not in log:
            failures.append(f"{name}: invalid/partly-invalid value was not rejected by validation (exit={rc})")
    for name in LEGACY_POSITIVE:
        rc, log = results[name]
        if rc != 0:
            failures.append(f"{name}: supported legacy selector failed (exit={rc})")
    for name, (_, expected) in EXPECTED_FAILURES.items():
        rc, log = results[name]
        if rc == 0 or expected not in log:
            failures.append(f"{name}: expected failure containing {expected!r}, exit={rc}")

    if failures:
        print(f"[moe-build-knobs] FAIL: {len(failures)} problem(s)")
        for failure in failures:
            print(f"    {failure}")
        return 1
    summary = ", ".join(f"{name}={v['full']}/{v['decode']}" for name, v in observed.items())
    print(f"[moe-build-knobs] PASS: {summary}; legacy positive passed, two plants and nine invalid/policy inputs rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
