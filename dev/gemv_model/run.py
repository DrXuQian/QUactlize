#!/usr/bin/env python3
"""Fresh-point numerical/screen/confirmation/ACU loop; prebuilt-only on box."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_model.plan import POINTS, SCHEMA, BASE_EXECUTION, BASE_FUSION, candidates, inventory
from dev.gemv_model.plan import cohort
from dev.gemv_model.fixture import Bench
from dev.gemv_model.engine import Provider, correctness, timing_graph
from dev.gemv_model.access import access
from dev.gemv_simt.native import Runtime, checked
from dev.gemv_simt.q8_vector_run import l2_identity
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command
from quactlize.runtime.compiler import LIBRARIES, sha


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(bundle, cohort_name='model'):
    plan = cohort(cohort_name)
    data = json.loads((bundle / "manifest.json").read_text())
    if data.get("schema") != plan.SCHEMA or data.get("inventory") != json.loads(json.dumps(plan.inventory())):
        raise ValueError("bundle inventory differs")
    declared = {r['point']['name']:r for r in data['inventory']}
    names = [r['point']['name'] for r in data['records']]
    if len(names)!=len(set(names)) or not names:
        raise ValueError('duplicate/empty compiled point set')
    for r in data['records']:
        expected = declared.get(r['point']['name'])
        if not expected or any(r[k]!=expected[k] for k in ('point','candidates')):
            raise ValueError('compiled point differs from inventory')
    for file, digest in data["payloads"].items():
        p = bundle / file
        if p.parent != bundle or sha(p) != digest:
            raise ValueError("payload identity differs: " + file)
    for p, digest in data["source_hashes"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("compile source changed: " + p)
    if sha(bundle / "libquactlize_ppu_execution.so") != plan.BASE_EXECUTION or sha(bundle / "libquactlize_ppu_gate_up.so") != plan.BASE_FUSION:
        raise ValueError("immutable incumbent changed")
    return data


def summarize(samples, weight_bytes, target):
    if not samples or any(len(r) != 15 or any(not math.isfinite(x) or x <= 0 for x in r) for r in samples):
        raise ValueError("confirmation needs fifteen finite positive samples per round")
    if len(samples) != 6:
        raise ValueError("confirmation needs six alternating rounds")
    medians = [statistics.median(r) for r in samples]
    us = statistics.median(medians)
    mbu = weight_bytes / us / 2.7e6 * 100
    return dict(median_us=us, round_medians_us=medians, samples_us=samples,
                modeled_weight_MBU_pct=mbu, target_MBU_pct=target, target_met=mbu >= target)


def child(args):
    plan = cohort(args.cohort)
    manifest = verify(args.bundle,args.cohort)
    point = next(p for p in plan.POINTS if p.name == args.point)
    record = next(r for r in manifest["records"] if r["point"]["name"] == point.name)
    configs = plan.candidates(point)
    rt = Runtime(args.sdk, "ppu")
    providers, graphs = {}, {}
    result = dict(status="FAIL", point=asdict(point), manifest_sha256=sha(args.bundle / "manifest.json"),
                  numerics={}, failures={}, production_selection_changed=False)
    try:
        identity = device_identity(rt)
        l2 = l2_identity(dict(l2_bytes=rt.attribute(38)), args.l2_bytes)
        if l2["l2_bytes"] <= 0:
            raise ValueError("positive L2 capacity required")
        bench = Bench(rt, point, l2["l2_bytes"], profile=bool(args.profile_arm))
        result.update(device=identity, l2=l2, fixture=bench.record,
                      runtime={f"lib{x}.so": sha(args.sdk / "lib" / f"lib{x}.so") for x in LIBRARIES},
                      timing_idle_admission="OPERATOR_IDLE_DEVICE_REQUIRED_NO_INDEPENDENT_AUDIT")
        if args.profile_arm:
            arm = args.profile_arm
            provider = Provider(bench, args.bundle, record, arm)
            providers[arm] = provider
            proof, _ = correctness(provider, repeat_controls=False)
            bench.update(1, 0); bench.poison()
            bench.mapped_call = False
            call = provider.prepare()
            for _ in range(5):
                checked(call(), "excluded profile warmup")
            rt.sync()
            with AcuRange(rt):
                checked(call(), "profile complete call"); rt.sync()
            bench.check()
            result.update(status="PASS", arm=arm, selection=provider.receipt, numerics=proof,
                          scope="ACU_KERNEL_REPLAY_CACHE_ALL_NOT_COLD_FULL_CALL_TIMING")
            save(args.output, result)
            return 0
        keys = ["incumbent"] + [str(i) for i in range(len(configs))]
        baseline_bits = None
        for key in keys:
            provider = Provider(bench, args.bundle, record, key)
            providers[key] = provider
            try:
                proof, snapshots = correctness(provider,token_controls=getattr(plan,'TOKEN_CONTROLS',None))
                if key == "incumbent":
                    baseline_bits = snapshots["m1"]
                elif not point.tc and key == "0" and not np.array_equal(snapshots["m1"], baseline_bits):
                    raise ValueError("same-geometry clone differs from immutable incumbent bits")
                result["numerics"][key] = dict(status="PASS", controls=proof, selection=provider.receipt)
            except ValueError as error:
                result["failures"][key] = str(error)
                traceback.print_exc()
                if key == "incumbent" or (key == "0" and not point.tc):
                    raise
            finally:
                provider.close()
            save(args.output, result)
            print(f"MODEL_GEMV_GATE point={point.name} arm={key} passed={key in result['numerics']} remaining_continue=1", flush=True)
        screen = {}
        for key in result["numerics"]:
            bench.poison()
            graphs[key] = timing_graph(providers[key])
            bench.check()  # Last ring copy and captured path must also be correct.
            screen[key] = []
        for round in range(3):
            order = list(graphs)
            if round % 2:
                order.reverse()
            for key in order:
                screen[key].append(graphs[key].sample())
        eligible = [k for k in graphs if k != "incumbent" and configs[int(k)].name not in ("clone", "generic")]
        if not eligible:
            raise ValueError("no candidate survived numeric gate")
        finalists = sorted(eligible, key=lambda k: statistics.median(screen[k]))[:2]
        selected = ["incumbent", *finalists]
        # The identical clone is a mandatory numeric control, not an optimizer.
        samples = {k: [] for k in selected}
        for round in range(6):
            for key in (selected if round % 2 == 0 else list(reversed(selected))):
                samples[key].append([graphs[key].sample() for _ in range(15)])
            print(f"MODEL_GEMV_CONFIRM point={point.name} round={round+1}/6", flush=True)
        target = 60 if bench.weight_bytes >= 16 * 1024 * 1024 else 40
        summary = {k: summarize(v, bench.weight_bytes, target) | providers[k].receipt for k, v in samples.items()}
        baseline = summary["incumbent"]["median_us"]
        for r in summary.values():
            r["delta_pct"] = 100 * (r["median_us"] / baseline - 1)
        best = min(finalists, key=lambda k: summary[k]["median_us"])
        bases = dict(A=bench.a.ptr % 128, low=bench.call().low % 128,
                     high=(bench.call().high or 0) % 128, units=bench.call().units % 128)
        result.update(status="PASS" if not result["failures"] else "PARTIAL", summary=summary, screen_us=screen,
                      best_candidate=best, candidate_within_5pct=summary[best]["delta_pct"] <= 5,
                      access={str(i): access(point, c, bases) for i, c in enumerate(configs)},
                      scope="ROTATING_COMPLETE_CALL_NOT_MODEL_TPOT")
        save(args.output, result)
        print("MODEL_GEMV_RESULT " + json.dumps(dict(point=point.name, incumbent_us=baseline,
              candidate=summary[best]["name"], candidate_us=summary[best]["median_us"],
              delta_pct=summary[best]["delta_pct"], MBU_pct=summary[best]["modeled_weight_MBU_pct"],
              status=result["status"])), flush=True)
        return int(result["status"] != "PASS")
    except Exception as error:
        result["error"] = str(error)
        save(args.output, result)
        raise
    finally:
        for graph in graphs.values():
            graph.close()
        for provider in providers.values():
            provider.close()
        rt.close()


def logged(command, path, label, echo=False):
    start = time.monotonic()
    with path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        pos = 0
        while proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if echo:
                with path.open(errors="replace") as src:
                    src.seek(pos)
                    for line in src:
                        if line.startswith("MODEL_GEMV_"):
                            print(line.rstrip(), flush=True)
                    pos = src.tell()
            if proc.poll() is None:
                print(f"MODEL_GEMV_WAIT {label} elapsed_s={time.monotonic()-start:.0f}", flush=True)
    return proc.returncode


def profile_records(text, p, arm, configs=None):
    offset = text.find('"ID"')
    if offset < 0:
        raise ValueError("ACU raw CSV missing")
    rows = [r for r in csv.DictReader(io.StringIO(text[offset:])) if r.get("Kernel Name")]
    config = (candidates(p) if configs is None else configs)[0 if arm == "incumbent" else int(arm)]
    split = p.tc[-1] if arm == "incumbent" and p.tc else config.split
    if len(rows) != 1 + int(split > 1):
        raise ValueError("ACU did not capture exactly the complete producer/reducer call")
    name = re.sub(r"\s+", "", rows[0]["Kernel Name"])
    if arm != "incumbent":
        if f"point_{p.q}_{p.physical_n}_{p.k}_arm_{arm}(" not in name:
            raise ValueError("ACU candidate symbol differs")
        block = f"({config.warps*32},1,1)"
        grid = f"({(8 if p.mode else 1)*config.split*p.physical_n//config.tile_n},1,1)"
        if rows[0]["Block Size"].replace(" ", "") != block or rows[0]["Grid Size"].replace(" ", "") != grid:
            raise ValueError("ACU candidate geometry differs")
    elif p.paired:
        if not re.search(rf'simt_gate_up(?:_model)?<{p.q},1,{p.compute},8(?:,\d+,\d+)?>',name):
            raise ValueError("ACU paired incumbent differs")
    elif not p.tc:
        if not any(s in name for s in ('q8_vector::kernel<','q8_vector::kernel_model<',
                                       'q8_vector::kernel_s1<','register_reuse<','register_reuse_model<')):
            raise ValueError("ACU SIMT incumbent differs")
    elif "cutlass" not in name or "kernel" not in name.lower():
        raise ValueError("ACU TC incumbent missing")
    if split > 1 and not any(w in rows[1]["Kernel Name"].lower() for w in ("reduce", "reduction")):
        raise ValueError("ACU complete-call reducer missing")
    return [dict(kernel=r["Kernel Name"], block=r["Block Size"], grid=r["Grid Size"]) for r in rows]


def collect(args):
    plan = cohort(args.cohort)
    manifest = verify(args.bundle,args.cohort)
    available = {r['point']['name'] for r in manifest['records']}
    selected = [p for p in plan.POINTS if p.name in available and (not args.points or p.name in args.points.split(','))]
    if not selected or (args.points and set(args.points.split(",")) - available):
        raise ValueError("unknown/empty point set")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(manifest=sha(args.bundle / "manifest.json"), points=[p.name for p in selected],
                    l2=args.l2_bytes, device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    runner={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob("*.py")},
                    runtime={f"lib{x}.so": sha(args.sdk / "lib" / f"lib{x}.so") for x in LIBRARIES})
    if args.resume:
        if json.loads((args.output / "identity.json").read_text()) != identity:
            raise ValueError("resume identity differs")
    else:
        save(args.output / "identity.json", identity)
    records = []
    previous = {}
    if args.resume and (args.output / "summary.json").is_file():
        previous = {r["point"]: r for r in json.loads((args.output / "summary.json").read_text()).get("records", [])}
    for p in selected:
        dest, log = args.output / (p.name + ".json"), args.output / (p.name + ".log")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--sdk", str(args.sdk),
               "--bundle", str(args.bundle), "--point", p.name, "--l2-bytes", str(args.l2_bytes),
               '--cohort',args.cohort]
        old = json.loads(dest.read_text()) if args.resume and dest.is_file() else {}
        rc = 0 if old.get("status") == "PASS" else logged(cmd + ["--output", str(dest)], log, "point=" + p.name, True)
        value = json.loads(dest.read_text()) if dest.is_file() else {}
        row = dict(point=p.name, status=value.get("status", "FAIL"), rc=rc, log=log.name, profiles=[])
        if "summary" in value:
            row.update(summary=value["summary"], best_candidate=value["best_candidate"])
        if args.acu and value.get("status") in ("PASS", "PARTIAL"):
            for arm in ("incumbent", value["best_candidate"]):
                admitted = [r for r in previous.get(p.name, {}).get("profiles", [])
                            if r.get("arm") == arm and r.get("status") == "PASS"]
                if admitted:
                    old_profile = admitted[0]
                    old_report = args.output / old_profile["report"]
                    if old_report.is_file() and sha(old_report) == old_profile["sha256"]:
                        profile_records((args.output / old_profile["csv"]).read_text(errors="replace"), p, arm,plan.candidates(p))
                        row["profiles"].append(old_profile)
                        print(f"MODEL_GEMV_RESUME point={p.name} arm={arm} profile=REUSED", flush=True)
                        continue
                stem = args.output / f"{p.name}-{arm}.acu"
                attempt = 1
                while any(args.output.glob(stem.name + ".*")):
                    stem = args.output / f"{p.name}-{arm}.acu-attempt{attempt}"
                    attempt += 1
                report, receipt, raw = [Path(str(stem) + suffix) for suffix in (".acurep", ".json", ".csv")]
                profile = dict(arm=arm, status="FAIL", report=report.name)
                try:
                    command = acu_launch_command(args.acu, stem, cmd + ["--profile-arm", arm, "--output", str(receipt)])
                    save(Path(str(stem) + ".command.json"), command)
                    rc = logged(command, Path(str(stem) + ".log"), f"ACU point={p.name} arm={arm}", True)
                    proof = json.loads(receipt.read_text())
                    if rc or proof.get("status") != "PASS" or not report.is_file():
                        raise ValueError("ACU report or independent numerical receipt missing")
                    if proof["manifest_sha256"] != value["manifest_sha256"] or proof["device"] != value["device"] or proof["fixture"]["raw_sha256"] != value["fixture"]["raw_sha256"]:
                        raise ValueError("ACU receipt differs from timed fixture/device")
                    rc = logged([str(args.acu), "--import", str(report), "--page", "raw", "--csv"], raw, "ACU export=" + p.name)
                    if rc:
                        raise ValueError("ACU export failed")
                    profile.update(status="PASS", kernels=profile_records(raw.read_text(errors="replace"), p, arm,plan.candidates(p)),
                                   sha256=sha(report), csv=raw.name)
                except Exception as error:
                    profile["error"] = str(error)
                    print(f"MODEL_GEMV_ACU FAIL point={p.name} arm={arm} error={error}", flush=True)
                    row["status"] = "PARTIAL"
                row["profiles"].append(profile)
        records.append(row)
        save(args.output / "summary.json", dict(complete=False, records=records))
        print(f"MODEL_GEMV_PROGRESS completed={len(records)}/{len(selected)} point={p.name} status={row['status']}", flush=True)
    status = "PASS" if all(r["status"] == "PASS" for r in records) else "INCOMPLETE"
    save(args.output / "summary.json", dict(status=status, complete=True, records=records,
                                           production_selection_changed=False, scope="SINGLE_CALL_NOT_MODEL_TPOT"))
    with (args.output / "summary.tsv").open("w") as dst:
        dst.write("point\tarm\tname\tfull_call_us\tdelta_pct\tmodeled_weight_MBU_pct\tstatus\n")
        for r in records:
            for a, s in r.get("summary", {}).items():
                dst.write(f"{r['point']}\t{a}\t{s['name']}\t{s['median_us']:.6f}\t{s['delta_pct']:.3f}\t{s['modeled_weight_MBU_pct']:.3f}\t{r['status']}\n")
    print(f"MODEL_GEMV_DONE status={status} results={args.output}", flush=True)
    return int(status != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("sdk", "bundle", "output"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--l2-bytes", type=int, default=67108864)
    parser.add_argument("--acu", type=Path)
    parser.add_argument('--cohort',choices=('model','tp2'),default='model')
    parser.add_argument("--point")
    parser.add_argument("--points")
    parser.add_argument("--profile-arm")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.sdk, args.bundle, args.output = (p.resolve() for p in (args.sdk, args.bundle, args.output))
    if args.verify_only:
        verify(args.bundle,args.cohort)
        print("MODEL_GEMV_PACKAGE PASS prebuilt=1 production_selection_changed=0")
        return 0
    return child(args) if args.point else collect(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
