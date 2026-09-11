#!/usr/bin/env python3
"""Fixed-recipe, same-GPU Q4 N2 instruction A/B; no PyTorch or online tuning."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

if __package__:
    from .run_xplane_ncu import sha
else:
    from run_xplane_ncu import sha


def parse_timing(text, arm, recipe, mode, shape):
    lines = [x for x in text.splitlines() if x.startswith("Q4_LAYOUT_PROFILE ")]
    if len(lines) != 1:
        raise ValueError("expected one runner result")
    head, tail = lines[0].split(" samples=", 1)
    row = dict(re.findall(r"(\w+)=([^ ]+)", head))
    expected = dict(arm=arm, config="-".join(map(str, recipe)), mode=mode,
                    shape="x".join(map(str, shape)), purpose="timing", status="PASS")
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("runner identity differs")
    samples = json.loads(tail)
    if len(samples) != 15 or not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in samples):
        raise ValueError("invalid timing samples")
    err = float(row["error"])
    if not math.isfinite(err) or not 0 <= err < .005:
        raise ValueError("numeric oracle failed")
    median = statistics.median(samples)
    if not math.isfinite(float(row["median_us"])) or abs(median-float(row["median_us"])) > 2e-5:
        raise ValueError("median disagrees with samples")
    return dict(**row, samples_us=samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "fixtures", "recipes", "xplane", "baseline", "candidate", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    receipt = json.loads(args.recipes.read_text())
    candidate_manifest = args.candidate.parent / "manifest.json"
    manifest = json.loads(candidate_manifest.read_text())
    if (manifest.get("reader") != "cuda-q4-n2" or
        manifest.get("pair_column_values_per_thread") != 2 or
        manifest.get("library") != args.candidate.name or
        manifest.get("library_sha256") != sha(args.candidate)):
        raise ValueError("candidate library/manifest identity differs")
    if sha(args.baseline) != receipt["kpack_library_sha256"] or sha(args.xplane) != receipt["xplane_library_sha256"]:
        raise ValueError("baseline library differs from the measured receipt")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = dict(runner=args.runner, recipes=args.recipes, xplane=args.xplane,
                 baseline=args.baseline, candidate=args.candidate, candidate_manifest=candidate_manifest)
    result = dict(status="RUNNING", scope="SAME_GPU_FIXED_5090_RECIPES_NOT_RETUNED_NOT_PPU",
                  arithmetic="N2_BASELINE_CANDIDATE_SAME_FP32_ORDER_XPLANE_FP16_GROUP_DOT",
                  source_sha256=sha(Path(__file__)), rounds=4, samples_per_round=15,
                  excluded_graph_warmups=5,
                  authority={k:dict(path=str(v.resolve()),sha256=sha(v)) for k,v in paths.items()}, cells=[])
    started = time.monotonic()
    device = None
    for case in receipt["cases"]:
        shape = case["shape"]
        if shape[:2] != [1,4096] or shape[2] not in (2048,4096):
            raise ValueError("outside the two Q4 anchors")
        fixture = args.fixtures / f"q12-n4096-k{shape[2]}-e1-c1.bin"
        for mode in ("warm", "rotating"):
            confirmed = case["modes"][mode]["confirmed"]
            def config(name):
                r = confirmed[name]["recipe"]
                return [r[x] for x in ("columns", "warps", "split")]
            arms = dict(xplane=(args.xplane, "xplane", config("xplane_best")),
                        baseline=(args.baseline, "kpack", config("kpack_best")),
                        candidate=(args.candidate, "kpack", config("kpack_best")),
                        baseline_s1=(args.baseline, "kpack", config("kpack_s1_best")),
                        candidate_s1=(args.candidate, "kpack", config("kpack_s1_best")))
            records = {k:[] for k in arms}
            for turn in range(4):
                names = list(arms) if turn%2 == 0 else list(reversed(arms))
                for name in names:
                    library, arm, recipe = arms[name]
                    command = [str(args.runner.resolve()), str(fixture.resolve()), str(args.xplane.resolve()),
                               str(library.resolve()) if arm == "kpack" else str(args.baseline.resolve()),
                               arm, *map(str, recipe), mode, "timing"]
                    proc = subprocess.run(command, text=True, capture_output=True, timeout=180)
                    log = args.output / f"k{shape[2]}-{mode}-{turn}-{name}.log"
                    log.write_text(proc.stdout+proc.stderr)
                    if proc.returncode:
                        raise RuntimeError(f"runner rc={proc.returncode}: {log}")
                    row = parse_timing(proc.stdout, arm, recipe, mode, shape)
                    identity = row["sm"], row["L2_bytes"]
                    if device is not None and identity != device:
                        raise ValueError("device geometry changed")
                    device = identity
                    records[name].append(row)
                print(f"Q4_NATIVE_AB_PROGRESS K={shape[2]} mode={mode} round={turn+1}/4 elapsed_s={time.monotonic()-started:.1f}", flush=True)
            medians = {name:statistics.median(float(x["median_us"]) for x in rows) for name,rows in records.items()}
            result["cells"].append(dict(shape=shape,mode=mode,fixture_sha256=sha(fixture),records=records,
                                        median_us=medians,
                                        candidate_delta_pct=(medians["candidate"]/medians["baseline"]-1)*100,
                                        candidate_s1_delta_pct=(medians["candidate_s1"]/medians["baseline_s1"]-1)*100))
            (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
            print("Q4_NATIVE_AB_RESULT "+json.dumps(dict(shape=shape,mode=mode,median_us=medians)),flush=True)
    if any(sha(v) != result["authority"][k]["sha256"] for k,v in paths.items()):
        raise ValueError("inputs changed during measurement")
    result.update(status="PASS", elapsed_seconds=time.monotonic()-started,
                  device_geometry=dict(sm=int(device[0]),l2_bytes=int(device[1])))
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Q4_NATIVE_AB_COMPLETE status=PASS",flush=True)


if __name__ == "__main__":
    main()
