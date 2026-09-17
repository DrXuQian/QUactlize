#!/usr/bin/env python3
"""16-point cold-weight A/B; screen bounded configs, confirm both backend winners."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gate_up_perf.plan import points, candidates
from dev.gate_up_perf.bench import Bench, Arm
from dev.gemv_simt.native import Runtime
from dev.gemv_ppu.run import query_l2_attribute, resolve_l2
from tools.run_kpack_gate_up import device_identity
from quactlize.runtime.compiler import sha, LIBRARIES


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(bundle, sdk):
    m = json.loads((bundle / "manifest.json").read_text())
    if m["schema"] != "quactlize.gate-up-perf.v1":
        raise ValueError("wrong performance package")
    runtime = {f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES}
    if runtime != m["runtime"]:
        raise ValueError("runtime differs from bundle")
    for name, want in m["payloads"].items():
        p = (bundle / name).resolve(strict=True)
        if not p.is_relative_to(bundle.resolve()) or sha(p) != want:
            raise ValueError("payload differs: " + name)
    for r in m["modules"]:
        p = (bundle / r["path"]).resolve(strict=True)
        if not p.is_relative_to(bundle.resolve()) or sha(p) != r["sha256"]:
            raise ValueError("TC payload differs")
    # Source hash contract applies to the new post-op, not to intentional
    # immutable incumbent image reuse. The complete original receipts remain.
    name = "dev/gate_up_perf/postop.cu"
    if sha(ROOT / name) != m["source_hashes"][name]:
        raise ValueError("post-op source changed")
    if [dict((k, p[k]) for k in points()[0]) for p in m["points"]] != points():
        raise ValueError("workload inventory changed")
    return m


def identity(args, m):
    return dict(
        manifest_sha256=sha(args.bundle / "manifest.json"),
        l2_bytes=args.l2_bytes,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        rounds=args.rounds,
        samples=args.samples,
        helpers={
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "dev/gate_up_perf").glob("*.py"))
        },
        shared_helpers={
            name: sha(ROOT / name)
            for name in (
                "dev/gemv_simt/native.py",
                "dev/gemv_simt/production.py",
                "dev/bf16_compute/fixture.py",
                "quactlize/decode/native_compute.py",
                "quactlize/execution/native.py",
                "quactlize/fusion/native.py",
                "dev/gemv_ppu/run.py",
                "tools/kpack_warmup_fixture.py",
                "reference/gguf_kpack.py",
            )
        },
    )


def finite_samples(values, count):
    return len(values) == count and all(
        isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in values
    )


def complete(value, point, ident):
    expected = {"incumbent", *candidates()}
    try:
        records = value["screen"]
        keys = [r["arm"]["key"] for r in records]
        if (
            value["status"] != "PASS"
            or value["identity"] != ident
            or value["point"] != point
        ):
            return False
        if len(keys) != len(expected) or set(keys) != expected:
            return False
        for r in records:
            if not finite_samples(r["samples_us"], 3) or r[
                "median_us"
            ] != statistics.median(r["samples_us"]):
                return False
            proof = r["proof"]
            if (
                proof["zero_a_negative"] != "RED"
                or len(proof["errors"]) != 2
                or any(
                    not math.isfinite(x) or x < 0 or x >= 0.005 for x in proof["errors"]
                )
            ):
                return False
        winners = value["winners"]
        for backend in ("simt", "tc"):
            best = min(
                (r for r in records if r["arm"]["key"].startswith(backend + "-")),
                key=lambda r: r["median_us"],
            )
            if winners[backend] != best["arm"]["key"]:
                return False
        selected = {"incumbent", *winners.values()}
        confirmation = value["confirmation"]
        pairs = [(r["round"], r["key"]) for r in confirmation]
        if len(pairs) != ident["rounds"] * 3 or set(pairs) != {
            (i, k) for i in range(ident["rounds"]) for k in selected
        }:
            return False
        if not all(
            finite_samples(r["samples_us"], ident["samples"]) for r in confirmation
        ):
            return False
        f = value["fixture"]
        if (
            f["copies"] * f["weight_bytes"] != f["cold_weight_bytes"]
            or f["cold_weight_bytes"] < 2.25 * f["l2_bytes"]
        ):
            return False
        for k, metric in value["metrics"].items():
            actual = statistics.median(
                [x for r in confirmation if r["key"] == k for x in r["samples_us"]]
            )
            if metric["median_us"] != actual:
                return False
            if (
                metric["delta_pct"]
                != (actual / value["metrics"]["incumbent"]["median_us"] - 1) * 100
            ):
                return False
            if metric["modeled_weight_mbu_pct"] != 100 * f["weight_bytes"] / (
                actual * 2700 * 1000
            ):
                return False
        return set(value["metrics"]) == selected
    except (KeyError, TypeError, ValueError):
        return False


def child(args):
    m = verify(args.bundle, args.sdk)
    rt = Runtime(args.sdk, "ppu")
    arms = []
    point = m["points"][args.point]
    ident = identity(args, m)
    result = dict(
        status="FAIL",
        identity=ident,
        point=point,
        screen=[],
        confirmation=[],
        environment_idle="NOT_PROVEN",
        timing_scope="FULL_CALL_ROTATING_WEIGHTS_NOT_MODEL",
    )
    started = time.monotonic()
    try:
        ident["device"] = device_identity(rt)
        attr = query_l2_attribute(rt.lib)
        if args.l2_bytes and attr["bytes"] > 0 and attr["bytes"] != args.l2_bytes:
            raise ValueError("verified L2 differs from device attribute")
        l2 = resolve_l2(0, args.l2_bytes, attr)
        result["l2"] = l2
        bench = Bench(rt, args.bundle, m, point, l2["l2_bytes"])
        result["fixture"] = bench.proof
        options = [("incumbent", None), *candidates().items()]
        print(
            f"GATE_UP_PERF_POINT key={point['key']} arms={len(options)} copies={bench.copies} active_experts={len(bench.active)} bytes={bench.weight_bytes}",
            flush=True,
        )
        for index, (key, cfg) in enumerate(options):
            arm = Arm(bench, key, cfg)
            arms.append(arm)
            proof = arm.correctness()
            record = dict(arm=arm.receipt, proof=proof, samples_us=[])
            result["screen"].append(record)
            print(
                f"GATE_UP_PERF_NUMERIC key={point['key']} completed={index+1}/{len(options)} arm={key}",
                flush=True,
            )
        # Alternate across all arms for the cheap screen; each graph traverses
        # a full ring. Confirmation starts fresh, so selection samples are not
        # recycled into the claimed winner's timing distribution.
        for round in range(3):
            order = list(range(len(arms)))
            if round % 2:
                order.reverse()
            for i in order:
                result["screen"][i]["samples_us"].append(arms[i].graph.sample())
        for r in result["screen"]:
            r["median_us"] = statistics.median(r["samples_us"])
        result["winners"] = {
            backend: min(
                (
                    r
                    for r in result["screen"]
                    if r["arm"]["key"].startswith(backend + "-")
                ),
                key=lambda r: r["median_us"],
            )["arm"]["key"]
            for backend in ("simt", "tc")
        }
        finalists = [
            a for a in arms if a.key in {"incumbent", *result["winners"].values()}
        ]
        for round in range(args.rounds):
            ordered = finalists if round % 2 == 0 else list(reversed(finalists))
            for a in ordered:
                samples = [a.graph.sample() for _ in range(args.samples)]
                result["confirmation"].append(
                    dict(round=round, key=a.key, samples_us=samples)
                )
                bench.check(a)
            print(
                f"GATE_UP_PERF_CONFIRM key={point['key']} round={round+1}/{args.rounds}",
                flush=True,
            )
        medians = {
            a.key: statistics.median(
                [
                    x
                    for r in result["confirmation"]
                    if r["key"] == a.key
                    for x in r["samples_us"]
                ]
            )
            for a in finalists
        }
        result["metrics"] = {
            k: dict(
                median_us=v,
                delta_pct=(v / medians["incumbent"] - 1) * 100,
                modeled_weight_mbu_pct=100 * bench.weight_bytes / (v * 2700 * 1000),
            )
            for k, v in medians.items()
        }
        result.update(
            status="PASS",
            elapsed_seconds=time.monotonic() - started,
            peak_bandwidth_assumption_gbps=2700,
            production_admission="PENDING_REVIEW_AND_MODEL_GATE",
            native_llama_comparison="NOT_RUN",
        )
        if not complete(result, point, ident):
            raise ValueError("incomplete timing/correctness receipt")
        print(
            "GATE_UP_PERF_RESULT "
            + json.dumps(dict(point=point["key"], metrics=result["metrics"])),
            flush=True,
        )
        return 0
    except Exception as error:
        result.update(status="FAIL", error=str(error))
        traceback.print_exc()
        return 1
    finally:
        save(args.output, result)
        for a in reversed(arms):
            a.close()
        rt.close()


def collect(args):
    m = verify(args.bundle, args.sdk)
    ident = identity(args, m)
    rt = Runtime(args.sdk, "ppu")
    try:
        ident["device"] = device_identity(rt)
    finally:
        rt.close()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    ip = args.output / "identity.json"
    if args.resume and (not ip.is_file() or json.loads(ip.read_text()) != ident):
        raise ValueError("resume identity differs")
    if not args.resume:
        save(ip, ident)
    done = []
    start = time.monotonic()
    for i, point in enumerate(m["points"]):
        output = args.output / (point["key"] + ".json")
        log = output.with_suffix(".log")
        reused = (
            args.resume
            and output.exists()
            and complete(json.loads(output.read_text()), point, ident)
        )
        if not reused:
            for p in (output, log):
                if p.exists():
                    p.rename(p.with_name(p.name + ".previous." + str(time.time_ns())))
            cmd = [
                sys.executable,
                __file__,
                "--bundle",
                str(args.bundle),
                "--sdk",
                str(args.sdk),
                "--output",
                str(output),
                "--point",
                str(i),
                "--l2-bytes",
                str(args.l2_bytes),
                "--rounds",
                str(args.rounds),
                "--samples",
                str(args.samples),
            ]
            save(output.with_suffix(".command.json"), cmd)
            with log.open("w") as stream:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if line.startswith("GATE_UP_PERF_"):
                        print(line.rstrip(), flush=True)
                rc = process.wait()
        else:
            rc = 0
        result = json.loads(output.read_text()) if output.is_file() else {}
        valid = rc == 0 and complete(result, point, ident)
        done.append(
            dict(
                point=point["key"],
                status="PASS" if valid else "FAIL",
                result=output.name,
                sha256=sha(output) if output.is_file() else None,
                reused=reused,
                metrics=result.get("metrics", {}) if valid else {},
            )
        )
        elapsed = time.monotonic() - start
        eta = (
            elapsed / (i + 1) * (len(m["points"]) - i - 1) / 60
            if not args.resume
            else None
        )
        save(
            args.output / "summary.json",
            dict(
                status=(
                    "PASS"
                    if len(done) == 16 and all(x["status"] == "PASS" for x in done)
                    else "INCOMPLETE"
                ),
                expected=16,
                points=done,
                identity=ident,
                elapsed_seconds=elapsed,
                production_selection="UNCHANGED",
            ),
        )
        print(
            f"GATE_UP_PERF_PROGRESS completed={i+1}/16 failures={sum(x['status']!='PASS' for x in done)} elapsed_minutes={elapsed/60:.1f} remaining_minutes={eta} eta=OBSERVED_POINT_MEAN_NOT_GUARANTEE",
            flush=True,
        )
    return int(any(x["status"] != "PASS" for x in done))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "bundle", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--point", type=int, choices=range(16))
    p.add_argument("--l2-bytes", type=int, default=0)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    if a.rounds < 2 or a.samples < 3 or a.l2_bytes < 0:
        p.error("invalid confirmation/L2 settings")
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    if a.verify_only:
        verify(a.bundle, a.sdk)
        print("GATE_UP_PERF_VERIFIED")
    else:
        raise SystemExit(child(a) if a.point is not None else collect(a))
