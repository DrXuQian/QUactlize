#!/usr/bin/env python3
"""Matched small-M supplements, preserving independent per-candidate receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
from pathlib import Path
import queue
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dev.smallm_closure.plan import make_plan, model_inventory, validate, candidate_id, SCHEMA
from dev.smallm_closure.results import write, seal, read, shortlist, adjudicate, policy_review, tpot_estimate
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def verify_execution(bundle, plan):
    manifest = json.loads((bundle / "manifest.json").read_text())
    image = bundle / "libquactlize_ppu_execution.so"
    if sha(image) != manifest["execution_sha256"]:
        raise ValueError("execution image differs or is an LFS pointer")
    receipt = manifest["execution_receipt"]
    for name, want in receipt["source_hashes"].items():
        if sha(ROOT / name) != want:
            raise ValueError("execution source differs from checkout: " + name)
    if not set(plan["computes"]) <= set(receipt["simt_compute_v2"]["compute"]):
        raise ValueError("execution lacks an explicit compute type")
    for p in plan["points"]:
        available = {tuple(r[f] for f in ("variant", "columns", "warps", "values", "split"))
                     for r in receipt["simt_configs"][str(p["q"]) ]}
        for cid in p["candidates"]:
            c = plan["candidates"][cid]
            if c["kind"] == "simt" and tuple(c["recipe"][f] for f in ("variant", "columns", "warps", "values", "split")) not in available:
                raise ValueError("shortlisted SIMT recipe not in reused image")
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(image)], text=True)
    for symbol in ("quactlize_kpack_simt_query_v2", "quactlize_kpack_simt_run_v2", "quactlize_kpack_q4_decode_select_v1", "quactlize_kpack_q4_decode_run_v1"):
        if not any(line.split()[-1:] == [symbol] for line in symbols.splitlines()):
            raise ValueError("missing execution export: " + symbol)


def authority(args, plan):
    paths = [p for folder in (ROOT / "dev/smallm_closure", ROOT / "dev/bf16_compute", ROOT / "quactlize/decode")
             for p in folder.glob("*.py")]
    paths += [Path(__file__), ROOT / "quactlize/execution/native.py", ROOT / "quactlize/execution/simt_codegen.py",
              ROOT / "tools/run_kpack_gemv_gate.py", ROOT / "tools/run_kpack_grouped_decode_probe.py"]
    return dict(plan=plan["plan_sha256"], sources={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
                execution_sha256=sha(args.bundle / "libquactlize_ppu_execution.so"), l2_bytes=args.l2_bytes)


def probe(sdk_path, l2_bytes):
    from quactlize.runtime.native import SDK, checked
    from tools.run_kpack_pack_gate import device_identity
    from dev.gemv_ppu.run import query_l2_attribute
    sdk = SDK(sdk_path)
    result = device_identity(sdk)
    f = sdk.lib.hggcDeviceGetAttribute
    f.argtypes, f.restype = [C.POINTER(C.c_int), C.c_int, C.c_int], C.c_int
    count, warp = C.c_int(), C.c_int()
    checked(f(C.byref(count), 16, result["ordinal"]), "CU attribute")
    checked(f(C.byref(warp), 10, result["ordinal"]), "warp attribute")
    if count.value != 72 or warp.value != 32:
        raise ValueError("requires PPU-ZW810 72 CU / 32 lanes")
    l2 = query_l2_attribute(sdk.lib)
    if l2["bytes"] and l2_bytes and l2["bytes"] != l2_bytes:
        raise ValueError("verified L2 override disagrees with SDK attribute")
    return result | dict(compute_units=count.value, warp=warp.value, l2_attribute=l2, l2_override=l2_bytes)


def build(args, plan):
    from quactlize.decode.compiler import DecodeCompiler
    from quactlize.decode.grouped_compiler import GroupedComputeCompiler
    compilers = {}
    for spec in plan["modules"].values():
        key = (spec["compute"], spec["parent"]["route"].endswith("grouped"))
        if key not in compilers:
            cls = GroupedComputeCompiler if key[1] else DecodeCompiler
            compilers[key] = cls(args.sdk, args.output / "build-cache", compute_type=key[0])
    work = list(plan["modules"].items())
    # Memory admission reduces simultaneous compiler processes, never cancels
    # completed work or a running build because CPU utilization is low.
    mem = next(int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
    jobs = min(args.jobs, len(work), max(1, (mem - 4 * 2**30) // (2 * 2**30)))
    print(f"SMALLM_CLOSURE_BUILD modules={len(work)} requested_jobs={args.jobs} active_slots={jobs} memory_bytes={mem}", flush=True)
    results, done = {}, queue.Queue()
    def one(item):
        mid, spec = item
        try:
            c = compilers[spec["compute"], spec["parent"]["route"].endswith("grouped")]
            r = c.build(spec["parent"]) | dict(status="PASS", compute=spec["compute"])
        except Exception as exc:
            r = dict(status="FAIL", error=str(exc), parent=spec["parent"], compute=spec["compute"])
        write(args.output / "results/build" / (mid + ".json"), r)
        done.put((mid, r))
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(one, item) for item in work]
        while len(results) < len(work):
            try:
                mid, row = done.get(timeout=30)
                results[mid] = row
            except queue.Empty:
                if all(f.done() for f in futures):
                    for future in futures:
                        future.result()
                    raise RuntimeError("compiler workers ended without all result receipts")
            age = time.monotonic() - start
            remaining = age * (len(work) - len(results)) / max(1, len(results)) / 60
            print(f"SMALLM_CLOSURE_BUILD_PROGRESS completed={len(results)}/{len(work)} failed={sum(r['status']!='PASS' for r in results.values())} elapsed_minutes={age/60:.1f} remaining_minutes={remaining:.1f} eta=OBSERVED_BUILD_AVERAGE", flush=True)
    write(args.output / "results/modules.json", results)
    return results


def child(args, plan):
    from dev.smallm_closure.bench import Context, WeightRing, Simt, TensorCore, Unsupported, correctness, measure
    from quactlize.runtime.native import SDK
    from tools.run_kpack_grouped_device_gate import graph_bind
    from quactlize.runtime.compiler import LIBRARIES
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    dev = probe(args.sdk, args.l2_bytes)
    expected = json.loads((args.output / "results/devices.json").read_text())[os.environ["CUDA_VISIBLE_DEVICES"]]
    if dev != expected:
        raise ValueError("physical device differs from assigned worker")
    auth = authority(args, plan) | dict(device=dev,
        runtime={name: sha(args.sdk / "lib" / f"lib{name}.so") for name in LIBRARIES})
    groups = json.loads((args.output / "results/groups.json").read_text())
    points = [p for p in plan["points"] if p["id"] in groups[args.child]]
    modules = json.loads((args.output / "results/modules.json").read_text())
    print(f"SMALLM_CLOSURE_FIXTURE_START family={args.child} q={points[0]['q']} n={points[0]['n']} k={points[0]['k']} experts={points[0]['experts']}", flush=True)
    ring = WeightRing(sdk, points[0], args.l2_bytes)
    print(f"SMALLM_CLOSURE_FIXTURE family={args.child} points={len(points)} copies={ring.copies}", flush=True)
    try:
        # Second pass cross-confirms routed-profile winners. Screen receipts
        # are reused; no fixture regeneration and no full resweep.
        for index, p in enumerate(points + points):
            point_dir = args.output / "results/cells" / p["id"]
            context = Context(sdk, ring, p)
            tc_modules = {cid: modules[digest((p["compute"], plan["candidates"][cid]["parent"]))]
                          for cid in p["candidates"] if plan["candidates"][cid]["kind"] == "tc"}
            local_auth = auth | dict(point=p["id"], fixture=ring.receipt,
                tc_modules_sha256=digest({cid: (r.get("key"), r.get("sha256"), r["status"]) for cid, r in tc_modules.items()}))
            screen, confirmed = {}, {}
            def execute(cid, phase):
                path = point_dir / f"{phase}-{cid}.json"
                cached = read(path, local_auth)
                if cached and cached["status"] in ("MEASURED", "STRUCTURAL"):
                    return cached
                c = plan["candidates"][cid]
                obj = None
                started = time.monotonic()
                try:
                    context.update(0)
                    cls = TensorCore if c["kind"] == "tc" else Simt
                    # Cleanup partial allocations even when a query rejects.
                    obj = cls.__new__(cls)
                    if c["kind"] == "tc":
                        record = modules[digest((p["compute"], c["parent"]))]
                        if record["status"] != "PASS":
                            raise ValueError("TC_COMPILE_FAILED: " + record["error"])
                        obj.__init__(context, record, c)
                    else:
                        obj.__init__(context, args.bundle / "libquactlize_ppu_execution.so", c)
                    proof = correctness(obj, full=phase.endswith("-r0"))
                    row = dict(status="MEASURED", proof=proof, runtime=obj.receipt)
                    if phase == "screen":
                        row["samples_us"] = measure(obj, plan["protocol"]["screen"])
                    else:
                        row["samples_us"] = measure(obj, plan["protocol"]["samples"])
                except Unsupported as exc:
                    row = dict(status="STRUCTURAL", error=str(exc))
                except Exception as exc:
                    row = dict(status="FAIL", error=str(exc), traceback=traceback.format_exc())
                finally:
                    if obj is not None and hasattr(obj, "r"):
                        try:
                            obj.close()
                        except Exception as exc:
                            row = dict(status="FAIL", error="runtime cleanup: " + str(exc),
                                       original=row, traceback=traceback.format_exc())
                row.update(authority=local_auth, candidate=cid, phase=phase, elapsed_seconds=time.monotonic() - started)
                row = seal(row)
                write(path, row)
                return row
            def confirm(ids, epoch=0):
                rounds = {cid: [] for cid in ids}
                # All candidates rotate through alternating order in each
                # confirmation round; not four rounds of A then four of B.
                for rnd in range(plan["protocol"]["rounds"]):
                    order = list(ids)
                    random.Random(p["id"] + str(rnd // 2)).shuffle(order)
                    if rnd % 2:
                        order.reverse()
                    for cid in order:
                        rounds[cid].append(execute(cid, f"confirm-e{epoch}-r{rnd}"))
                for cid, rows in rounds.items():
                    good = all(r["status"] == "MEASURED" for r in rows)
                    confirmed[cid] = dict(status="MEASURED" if good else "FAIL",
                        rounds=[r["samples_us"] for r in rows] if good else [],
                        receipts=[r["receipt_sha256"] for r in rows])
            try:
                order = list(p["candidates"])
                random.Random(p["id"]).shuffle(order)
                for ordinal, cid in enumerate(order):
                    screen[cid] = execute(cid, "screen")
                    if ordinal % 16 == 0:
                        write(args.output / "results/progress" / (args.child + ".json"), dict(
                            point=p["id"], points_done=index, points_total=len(points)*2,
                            phase="screen", candidate_done=ordinal+1, candidate_total=len(order)))
                        print(f"SMALLM_CLOSURE_POINT point={p['id']} phase=screen completed={ordinal+1}/{len(order)}", flush=True)
                initial = set(shortlist(p, plan["candidates"], screen))
                selected = set(initial)
                if index >= len(points) and p["mode"]:
                    for other in points:
                        if any(other[f] != p[f] for f in ("q", "mode", "n", "k", "experts", "topk", "channels", "tokens", "compute")):
                            continue
                        path = args.output / "results/cells" / other["id"] / "result.json"
                        if path.exists():
                            result = json.loads(path.read_text())
                            if result.get("winner"):
                                cid = result["winner"]["candidate"]
                                if cid in screen and screen[cid]["status"] == "MEASURED":
                                    selected.add(cid)
                selected = sorted(selected)
                start_epoch = 2 if set(selected) - initial else 0
                confirm(selected, start_epoch)
                report = adjudicate(p, plan["candidates"], screen, confirmed, plan["protocol"])
                # Do not leave competitive screen-only cells for another box
                # request. Complete their confirmation in this invocation.
                epoch = start_epoch
                while True:
                    extra = {cid for issue in report["issues"] if issue["reason"] == "COMPETITIVE_UNCONFIRMED" for cid in issue["candidates"]} - confirmed.keys()
                    unstable = any(i["reason"] == "WINNER_UNSTABLE" for i in report["issues"])
                    if not extra and (not unstable or epoch >= start_epoch + 2):
                        break
                    # Reconfirm the incumbents in the expansion's own
                    # alternating cohort, not cross-cohort standalone times.
                    # Existing round receipts remain valid for recovery.
                    epoch += 1
                    confirm(sorted(set(confirmed) | extra), epoch)
                    report = adjudicate(p, plan["candidates"], screen, confirmed, plan["protocol"])
                report.update(authority=local_auth, point_spec=p)
                if report["winner"]:
                    active = len(set(map(int, context.data["owners"])))
                    unique = active * ring.expert_bytes
                    report["traffic"] = dict(unique_active_weight_bytes=unique, peak_gbps=2700,
                        modeled_mbu_pct=unique / (report["winner"]["median_us"] * 2700 * 1000) * 100,
                        scope="UNIQUE_PACKED_B_ONLY_NOT_ACU_COUNTERS_EXCLUDES_A_OUTPUT_AND_REPEATED_LOADS")
                    config = plan["candidates"][report["winner"]["candidate"]]
                    if config["kind"] == "simt":
                        from dev.gemv_simt.access import pattern
                        from quactlize.execution.simt_codegen import Config
                        access = pattern(p["q"], Config(**config["recipe"]), p["n"], p["k"])
                        write(point_dir / "winner-access-pattern.json", access)
                        report["access_pattern"] = "winner-access-pattern.json"
                write(point_dir / "result.json", seal(report))
                write(args.output / "results/progress" / (args.child + ".json"), dict(
                    point=p["id"], points_done=index+1, points_total=len(points)*2,
                    phase="point-complete", status=report["status"]))
                print(f"SMALLM_CLOSURE_POINT_DONE family={args.child} pass={1+index//len(points)} completed={index%len(points)+1}/{len(points)} status={report['status']}", flush=True)
            finally:
                context.close()
    finally:
        ring.close()
    return 0


def summarize(args, plan):
    rows, missing = [], []
    for p in plan["points"]:
        path = args.output / "results/cells" / p["id"] / "result.json"
        if path.exists():
            r = json.loads(path.read_text())
            if r.get("authority", {}).get("plan") != plan["plan_sha256"] or r.get("receipt_sha256") != digest({k: v for k, v in r.items() if k != "receipt_sha256"}):
                raise ValueError("stale or corrupt point summary")
            rows.append(r)
        else:
            missing.append(p["id"])
    good = sum(r["status"] == "MEASURED_POOL_CLOSED" for r in rows)
    policies = policy_review(rows)
    write(args.output / "results/policy-review.json", policies)
    write(args.output / "results/tpot-estimate.json", tpot_estimate(plan["inventory"], rows, policies))
    policy_open = [r for r in policies if r["status"] != "WITHIN_5_PERCENT"]
    report = dict(status="MEASURED_POOL_CLOSED" if good == len(plan["points"]) else "OPEN",
        expected=len(plan["points"]), completed=len(rows), closed=good, missing=missing,
        open_points=[dict(point=r["point"], issues=r["issues"]) for r in rows if r["status"] != "MEASURED_POOL_CLOSED"],
        evidence_audit=plan["evidence_audit"],
        production_admission="PENDING_POLICY_REPLAY_AND_MODEL_NUMERICAL_PERF_GATE",
        policy_5pct_admission="PASS" if not policy_open and good == len(plan["points"]) else "OPEN",
        policy_open=policy_open, scope=plan["scope"], global_optimality_proven=False)
    write(args.output / "results/coverage.json", report)
    winners = [dict(point=r["point_spec"], winner=r["winner"],
                    config=plan["candidates"][r["winner"]["candidate"]], status=r["status"])
               for r in rows if r["winner"]]
    write(args.output / "results/winners.json", winners)
    lines = ["q\tmode\tN\tK\ttokens\tchannels\tcompute\trouter\tstatus\tkind\tmedian_us\tissues"]
    for r in rows:
        p, win = r["point_spec"], r["winner"]
        kind = plan["candidates"][win["candidate"]]["kind"] if win else "NONE"
        lines.append("\t".join(map(str, [p[f] for f in ("q","mode","n","k","tokens","channels","compute","router")] +
            [r["status"], kind, win["median_us"] if win else "", ",".join(i["reason"] for i in r["issues"])])))
    (args.output / "results/summary.tsv").write_text("\n".join(lines) + "\n")
    print(f"SMALLM_CLOSURE_DONE status={report['status']} closed={good}/{len(plan['points'])} results={args.output}/results", flush=True)
    return 0 if report["status"] == "MEASURED_POOL_CLOSED" and report["policy_5pct_admission"] == "PASS" else 1


def campaign(args, plan):
    groups = {}
    for p in plan["points"]:
        key = digest(tuple(p[f] for f in ("q", "mode", "n", "k", "experts", "topk")))[:24]
        groups.setdefault(key, []).append(p["id"])
    write(args.output / "results/groups.json", groups)
    devices = {}
    for d in args.devices:
        env = os.environ | dict(CUDA_VISIBLE_DEVICES=d)
        info = subprocess.check_output([sys.executable, str(__file__), "--probe", "--sdk", str(args.sdk), "--l2-bytes", str(args.l2_bytes)], env=env, text=True)
        devices[d] = json.loads(info)
    if len({d["pci"] for d in devices.values()}) != len(devices):
        raise ValueError("workers refer to the same physical device")
    write(args.output / "results/devices.json", devices)
    assigned = {d: [] for d in args.devices}
    for i, g in enumerate(sorted(groups)):
        assigned[args.devices[i % len(args.devices)]].append(g)
    assignment_path = args.output / "results/assignment.json"
    if assignment_path.exists() and json.loads(assignment_path.read_text()) != assigned:
        raise ValueError("resume device assignment differs; do not mix timing cohorts")
    write(assignment_path, assigned)
    completed = queue.Queue()
    def worker(device):
        for g in assigned[device]:
            cmd = [sys.executable, "-u", str(__file__), "--child", g, "--output", str(args.output),
                   "--sdk", str(args.sdk), "--bundle", str(args.bundle), "--l2-bytes", str(args.l2_bytes)]
            path = args.output / "results/logs" / (g + ".log")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("a") as log:
                    rc = subprocess.run(cmd, env=os.environ | dict(CUDA_VISIBLE_DEVICES=device), stdout=log, stderr=subprocess.STDOUT).returncode
            except Exception:
                rc = 1
                traceback.print_exc()
            completed.put((g, rc))
    start, count, failed = time.monotonic(), 0, 0
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, d) for d in devices]
        while count < len(groups):
            try:
                g, rc = completed.get(timeout=30)
                count += 1
                failed += int(rc != 0)
            except queue.Empty:
                if all(f.done() for f in futures):
                    for future in futures:
                        future.result()
                    raise RuntimeError("device workers ended without all family receipts")
            age = time.monotonic() - start
            estimate = f"{age * (len(groups)-count) / count / 60:.1f}" if count else "UNKNOWN_FIRST_FAMILY"
            progress = [json.loads(path.read_text()) for path in (args.output / "results/progress").glob("*.json")]
            points_done = sum(p["points_done"] for p in progress)
            print(f"SMALLM_CLOSURE_PROGRESS families={count}/{len(groups)} passes={points_done}/{len(plan['points'])*2} failed={failed} elapsed_minutes={age/60:.1f} remaining_minutes={estimate} eta=OBSERVED_FAMILY_AVERAGE_NOT_GUARANTEE", flush=True)
    return summarize(args, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-plan", type=Path, default=ROOT / "tools/kpack_batched_int4_2048.json")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--compute", nargs="+", choices=("f16", "bf16"), default=["f16", "bf16"])
    parser.add_argument("--jobs", type=int, default=192)
    parser.add_argument("--l2-bytes", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe(args.sdk, args.l2_bytes)))
        return 0
    if args.output is None or args.jobs < 1 or len(set(args.devices)) != len(args.devices) or any(not d.isdigit() for d in args.devices):
        parser.error("output, positive jobs and distinct numeric devices required")
    args.output = args.output.resolve()
    frozen = args.output / "results/plan.json"
    if args.child:
        return child(args, validate(json.loads(frozen.read_text())))
    inventory = model_inventory(args.model_plan, args.model_root)
    plan = validate(make_plan(inventory, args.compute))
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("resume plan differs; use a fresh output directory")
    write(frozen, plan)
    print(f"SMALLM_CLOSURE_PLAN points={len(plan['points'])} modules={len(plan['modules'])} screen_cells={sum(len(p['candidates']) for p in plan['points'])} pool=EXTRACTED_NOT_CARTESIAN", flush=True)
    write(args.output / "results/evidence-audit.json", plan["evidence_audit"])
    if args.plan_only:
        return 0
    if args.bundle is None or args.l2_bytes <= 0:
        parser.error("verified bundle and L2 bytes required for measurements")
    args.bundle = args.bundle.resolve(strict=True)
    verify_execution(args.bundle, plan)
    build(args, plan)
    if args.build_only:
        return 0
    return campaign(args, plan)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
