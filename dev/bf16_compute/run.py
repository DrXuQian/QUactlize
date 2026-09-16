"""Run a bounded prebuilt BF16 capability gate; never compile or select tactics."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.bf16_compute.plan import QTYPES, cases, Parent
from quactlize.runtime.compiler import sha


def validate_package(root):
    root = root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "quactlize.bf16-device-gate.v1":
        raise ValueError("not an explicit BF16 capability package")
    for record in [*manifest["modules"].values(), *manifest.get("matched_modules", {}).values(), manifest["simt"], manifest["moe"]]:
        path = (root / record["path"]).resolve(strict=True)
        if root not in path.parents or sha(path) != record["sha256"]:
            raise ValueError("gate payload missing or differs: " + str(path))
    for name, expected in manifest["source"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("device gate source differs from its package: " + name)
    expected = [case["id"] for case in cases()]
    if [case["id"] for case in manifest["cases"]] != expected:
        raise ValueError("gate case denominator differs from package")
    return manifest


def run_child(args, manifest):
    from quactlize.runtime.native import SDK
    from tools.run_kpack_pack_gate import device_identity
    from tools.run_kpack_grouped_device_gate import graph_bind
    from dev.bf16_compute import cases as device_cases, moe_case
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    identity = device_identity(sdk)
    family, q = args.child.rsplit("-q", 1)
    selected = [case for case in cases() if case["family"] == family and case["q"] == int(q)]
    if not selected:
        raise ValueError("child has no declared cases")
    args.output.mkdir(parents=True, exist_ok=True)
    records, failed = [], False
    started = time.monotonic()
    for i, case in enumerate(selected):
        public = dict(case)
        if "parent" in public:
            public["parent"] = asdict(public["parent"])
        print(f"BF16_GATE_CASE_START id={case['id']} completed={i}/{len(selected)}", flush=True)
        try:
            function = moe_case.run if family == "moe" else getattr(device_cases, family)
            result = function(args.package, manifest, sdk, case, args.repeats, args.samples)
            result.update(case=public, device=identity, correctness_only=not bool(args.samples),
                          timing_scope="WARM_DIAGNOSTIC_NOT_HEURISTIC_AUTHORITY")
        except Exception as error:
            traceback.print_exc()
            result = dict(status="FAIL", case=public, device=identity, error=str(error))
            failed = True
        records.append(result)
        (args.output / (case["id"] + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        print(f"BF16_GATE_CASE id={case['id']} status={result['status']} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        if failed:
            # A bad device context cannot prove later kernels. Other format /
            # family children remain independent and continue in the parent.
            break
    summary = dict(status="FAIL" if failed else "PASS", expected=len(selected),
        completed=len(records), passed=sum(r["status"] == "PASS" for r in records),
        missing=[c["id"] for c in selected[len(records):]], device=identity,
        seconds=time.monotonic()-started, timing_valid=False,
        reason="STOPPED_THIS_CHILD_AFTER_NUMERICAL_OR_RUNTIME_FAILURE" if failed else "COMPLETE")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--family", choices=("all", "grouped", "simt", "moe", "outlier"), default="all")
    parser.add_argument("--qtype", type=int, choices=QTYPES)
    parser.add_argument("--child")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.repeats <= 7 or not 0 <= args.samples <= 11:
        parser.error("repeats must be 2..7 and optional diagnostic samples 0..11")
    manifest = validate_package(args.package)
    if args.child:
        return run_child(args, manifest)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    receipt = dict(package_sha256=sha(args.package / "manifest.json"), repeats=args.repeats,
                   samples=args.samples, family=args.family, qtype=args.qtype)
    receipt_file = args.output / "request.json"
    if receipt_file.exists():
        if json.loads(receipt_file.read_text()) != receipt:
            raise ValueError("resume package/options differ; use a fresh output directory")
    else:
        receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    selected = [case for case in cases() if (args.family == "all" or case["family"] == args.family)
                and (args.qtype is None or case["q"] == args.qtype)]
    if not selected:
        raise ValueError("requested family/format has no declared cases")
    groups = list(dict.fromkeys((c["family"], c["q"]) for c in selected))
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    records = []
    started = time.monotonic()
    for family, q in groups:
        key = f"{family}-q{q}"
        target = args.output / key
        summary_path = target / "summary.json"
        if args.resume and summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "PASS":
            records.append(json.loads(summary_path.read_text()) | dict(part=key, reused=True))
            continue
        target.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--sdk", str(args.sdk), "--package", str(args.package),
                   "--output", str(target), "--repeats", str(args.repeats), "--samples", str(args.samples), "--child", key]
        (target / "command.json").write_text(json.dumps(command) + "\n")
        with (target / "console.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    completed = len(list(target.glob("[0-9]*.json")))
                    print(f"BF16_GATE_PROGRESS part={key} completed={completed} elapsed_minutes={(time.monotonic()-started)/60:.1f} log={target/'console.log'}", flush=True)
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(
            status="FAIL", expected=sum(c["family"] == family and c["q"] == q for c in selected),
            completed=0, passed=0, error="child terminated without a summary")
        summary.update(part=key, process_rc=process.returncode)
        records.append(summary)
        print(f"BF16_GATE_PART part={key} status={summary['status']} passed={summary['passed']}/{summary['expected']} remaining_parts_continue=1", flush=True)
    passed = sum(r["passed"] for r in records)
    status = "PASS" if passed == len(selected) and all(r["status"] == "PASS" for r in records) else "FAIL"
    result = dict(status=status, expected=len(selected), passed=passed, parts=records, request=receipt,
        families=dict(Counter(c["family"] for c in selected)), seconds=time.monotonic()-started,
        device_validated=status == "PASS", performance_admitted=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"BF16_DEVICE_GATE status={status} passed={passed}/{len(selected)} performance_admitted=0 results={args.output}")
    return int(status != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
