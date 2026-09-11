#!/usr/bin/env python3
"""Bounded Q4 CUDA screening/confirmation, including both FP32 controls.

Keeps individual logs and all event samples. Timing includes the full launch;
the unchanged graph harness excludes upload/warmup and completes cache rings.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_cuda.build_h800_candidates import N4
from dev.gemv_cuda.compare_q4_native import parse_timing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidates", type=Path, nargs="*", default=[])
    p.add_argument("--fixtures", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--selection", type=Path, help="JSON of arm -> recipes, freezes confirmation shortlist")
    a = p.parse_args()
    a.control = a.control.resolve(strict=True); a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.rounds < 1: raise ValueError("rounds must be positive")
    control = json.loads((a.control / "manifest.json").read_text())
    for name, sha in control["payloads"].items():
        if digest(a.control / name) != sha: raise ValueError("control payload differs")
    arms = {
        "xplane": dict(library=str(a.control / "libkpack.so"), mode="xplane",
                       recipes=[[c, w, 1] for c in (1, 2, 4, 8) for w in (2, 4, 8)]),
        "raw-reference": dict(library=str(a.control / "libkpack.so"), mode="raw-reference",
                              recipes=control.get("reference_recipes", [[c, 8, 1] for c in (1, 2, 4)])),
        "baseline": dict(library=str(a.control / "libkpack.so"), mode="kpack",
                         recipes=[[*r, 1] for r in N4]),
    }
    for path in a.candidates:
        rows = json.loads((path / "manifest.json").read_text())["arms"]
        for row in rows:
            if row["arm"] in arms: raise ValueError("duplicate arm")
            if digest(Path(row["library"])) != row["sha256"]: raise ValueError("candidate payload differs")
            arms[row["arm"]] = dict(library=row["library"], mode="kpack", recipes=row["recipes"],
                                    launches_per_call=row.get("launches_per_call",1),
                                    weight_arithmetic=row.get("weight_arithmetic","PER_WEIGHT_FP16_AFFINE"))
    if a.selection:
        selection = json.loads(a.selection.read_text())
        if not {"xplane", "raw-reference"} <= selection.keys(): raise ValueError("missing comparison control")
        for name, recipes in selection.items():
            if name not in arms or not recipes or any(r not in arms[name]["recipes"] for r in recipes):
                raise ValueError("uncompiled/empty confirmation selection")
        arms = {name: arms[name] | dict(recipes=rs) for name, rs in selection.items()}
    started = time.monotonic()
    result = dict(status="RUNNING", precision="FP16_A_FP32_DOT_REDUCTION_OUTPUT_SEE_ARM_WEIGHT_ARITHMETIC", arms=arms,
                  rounds=a.rounds, controls_retuned=True, cases=[], failures=[], production_changed=False,
                  control_manifest_sha256=digest(a.control / "manifest.json"))
    def save(): (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for fixture in a.fixtures:
        # Binary fixture shape is also checked inside the independent runner.
        import re
        match = re.fullmatch(r"q12-n(\d+)-k(\d+)-e1-c1.bin", fixture.name)
        if not match: raise ValueError("fixture filename outside this experiment")
        n, k = map(int, match.groups())
        for mode in ("warm", "rotating"):
            records = {name: [] for name in arms}
            for turn in range(a.rounds):
                order = list(arms) if turn % 2 == 0 else list(arms)[::-1]
                for name in order:
                    arm = arms[name]
                    recipes = list(arm["recipes"])
                    if turn % 2: recipes.reverse()
                    log = a.output / f"n{n}-k{k}-{mode}-{turn}-{name}.log"
                    cmd = [str(a.control / "profile"), str(fixture), str(a.control / "libreference.so"),
                           arm["library"], arm["mode"], ",".join(":".join(map(str, r)) for r in recipes),
                           "1", "1", mode, "batch"]
                    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=240)
                    log.write_text(proc.stdout + proc.stderr)
                    found = set()
                    for line in proc.stdout.splitlines():
                        if not line.startswith("Q4_LAYOUT_PROFILE "): continue
                        cfg = re.search(r"config=(\d+)-(\d+)-(\d+)", line)
                        if not cfg: raise ValueError("missing recipe")
                        recipe = list(map(int, cfg.groups()))
                        if recipe not in recipes or tuple(recipe) in found: raise ValueError("unexpected/duplicate recipe")
                        row = parse_timing(line, arm["mode"], recipe, mode, [1, n, k])
                        ring = int(row["copies"]); calls = int(row["calls_per_graph"])
                        if calls % ring or (mode == "rotating" and ring * n * k * 9 // 16 < 2.25 * int(row["L2_bytes"])):
                            raise ValueError("partial/undersized cache ring")
                        found.add(tuple(recipe))
                        records[name].append(row | dict(recipe=recipe, turn=turn, log=log.name, log_sha256=digest(log)))
                    if proc.returncode or len(found) != len(recipes):
                        result["failures"].append(dict(arm=name, shape=[n, k], mode=mode, rc=proc.returncode, log=log.name))
                    print(f"Q4_H800_SCREEN shape={n}x{k} mode={mode} round={turn+1}/{a.rounds} arm={name} completed={len(found)}/{len(recipes)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
            best = {}
            for name, rows in records.items():
                candidates = []
                for recipe in arms[name]["recipes"]:
                    matching = [r for r in rows if r["recipe"] == recipe]
                    if len(matching) != a.rounds: continue
                    candidates.append(dict(recipe=recipe, median_us=statistics.median(float(r["median_us"]) for r in matching)))
                if candidates: best[name] = min(candidates, key=lambda r:r["median_us"])
            row = dict(shape=[1, n, k], mode=mode, best=best, records=records, fixture_sha256=digest(fixture))
            result["cases"].append(row); save()
            print("Q4_H800_BEST " + json.dumps({k:v for k,v in row.items() if k != "records"}), flush=True)
    result["seconds"] = time.monotonic() - started
    result["status"] = "INCOMPLETE" if result["failures"] else "MEASURED"
    save()
    print(f"Q4_H800_SCREEN_COMPLETE status={result['status']} results={a.output}", flush=True)


if __name__ == "__main__": main()
