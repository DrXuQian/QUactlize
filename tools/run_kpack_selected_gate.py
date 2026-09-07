#!/usr/bin/env python3
"""Validate fixed single-tactic dispatch using cached real-shape modules.

No candidate search. Compilation is disabled unless explicitly requested.
Kernel timing is validation-only and never changes the selected tactic.
"""

import argparse
from collections import Counter
import ctypes as C
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from tools.kpack_heuristic import Selector, RECENT
from tools.run_kpack_warmup_real import save
from quactlize.runtime.candidates import PARENT_FIELDS, from_config
from quactlize.runtime.compiler import Compiler, sha
from quactlize.runtime.dispatch import prepare_selected
from quactlize.runtime.native import NativeBackend, SDK, checked, sdk_identity
from quactlize.runtime.tuning import Request, digest

MODEL = ROOT / "policies/kpack_zw810_heuristic_v1.json"
BASE = ROOT / "policies/kpack_zw810_runtime_v1.json"


def request_of(case):
    return Request(**(case["request"] | {"rows": tuple(case["request"]["rows"])}))


def select(selector, request, identity):
    p = dict(
        qtype=request.qtype,
        n=request.n,
        k=request.k,
        group_size=32 if request.qtype in (12, 13) else 16,
    )
    if request.grouped:
        p.update(
            experts=len(request.rows), total_rows=request.m, max_rows=max(request.rows)
        )
    else:
        p["m"] = request.m
    return selector.select(
        request.route,
        p,
        list(request.rows) if request.grouped else None,
        device_name=identity["device"],
        compute_units=identity["compute_units"],
        mapping_id=f"0x{request.mapping:016x}",
        kernel_source=identity["kernel"],
        sdk_digest=identity["sdk"],
        allow_prediction=False,
    )


def make_plan(model, base):
    selector = Selector(model, base)
    cases, parents = [], {}
    for entry in model["entries"]:
        ctx = entry["context"]
        p = ctx["problem"]
        request = Request(
            ctx["route"],
            p["qtype"],
            p["n"],
            p["k"],
            p.get("m", p.get("total_rows")),
            tuple(ctx["rows"]),
        )
        chosen = select(selector, request, model["required_binding"])
        if chosen["status"] != RECENT or chosen["config_id"] != entry["config_id"]:
            raise ValueError("recent exact selection differs from calibration")
        c = chosen["config"]
        parents[c["symbol"]] = {k: c[k] for k in PARENT_FIELDS}
        cases.append(
            dict(
                id=request.exact_key,
                request=asdict(request) | {"rows": list(request.rows)},
                config_id=entry["config_id"],
                selection=chosen,
                calibration_us=entry["selected_us"],
            )
        )
    counts = Counter(c["request"]["route"] for c in cases)
    expected = {"fq-dense": 6, "sf-dense": 4, "fq-grouped": 4, "sf-grouped": 4}
    if (
        len(cases) != 90
        or len({c["id"] for c in cases}) != 90
        or counts
        != {"fq-dense": 30, "sf-dense": 20, "fq-grouped": 20, "sf-grouped": 20}
        or {c["request"]["qtype"] for c in cases} != set(range(10, 15))
        or Counter((c["request"]["qtype"], c["request"]["route"]) for c in cases)
        != {(q, route): n for q in range(10, 15) for route, n in expected.items()}
    ):
        raise ValueError(
            "selected gate denominator differs: expected five formats / 90 contexts"
        )
    value = dict(
        schema="quactlize.kpack-selected-gate-plan.v1",
        model_digest=model["model_digest"],
        cases=cases,
        parents=[parents[k] for k in sorted(parents)],
        validation=dict(positive_checks=3, samples=3, repeats=5, warmups=2),
        scope="SINGLE_SELECTED_DISPATCH_NOT_TUNING_OR_GLOBAL_PERFORMANCE_ADMISSION",
    )
    value["digest"] = digest(value)
    return value


def module_key(parent, contract):
    # Reuse the existing pure source generator without constructing a compiler
    # (which would inspect g++). Cache-only execution requires no build tools.
    generator = Compiler.__new__(Compiler)
    return digest(
        dict(identity=contract, parent=parent, source=generator.source(parent, ""))
    )


def cached_records(plan, contract, cache):
    records, missing = [], []
    for parent in plan["parents"]:
        key = module_key(parent, contract)
        folder = cache / key
        receipt, payload = folder / "manifest.json", folder / "kernel.so"
        if not receipt.is_file() or not payload.is_file():
            missing.append(parent)
            continue
        record = json.loads(receipt.read_text())
        if (
            record.get("key") != key
            or record.get("parent") != parent
            or record.get("identity") != contract
            or record.get("sha256") != sha(payload)
        ):
            raise ValueError(f"cached module identity/payload differs: {folder}")
        records.append(record | {"path": str(payload), "cache_hit": True})
    return records, missing


def device_identity(sdk):
    lib = sdk.lib

    def function(names, args):
        for name in names:
            if hasattr(lib, name):
                f = getattr(lib, name)
                f.argtypes, f.restype = args, C.c_int
                return f
        raise ValueError(f"SDK device identity query missing: {names}")

    count, ordinal = C.c_int(), C.c_int()
    checked(
        function(("hggcGetDeviceCount", "cudaGetDeviceCount"), [C.POINTER(C.c_int)])(
            C.byref(count)
        ),
        "device count",
    )
    if count.value != 1:
        raise ValueError(
            f"runtime sees {count.value} devices; set CUDA_VISIBLE_DEVICES to exactly one device"
        )
    checked(
        function(("hggcGetDevice", "cudaGetDevice"), [C.POINTER(C.c_int)])(
            C.byref(ordinal)
        ),
        "device ordinal",
    )
    pci = C.create_string_buffer(64)
    checked(
        function(
            ("hggcDeviceGetPCIBusId", "cudaDeviceGetPCIBusId"),
            [C.c_char_p, C.c_int, C.c_int],
        )(pci, len(pci), ordinal.value),
        "device PCI identity",
    )
    if not pci.value:
        raise ValueError("empty device PCI identity")
    return dict(
        host=socket.gethostname(),
        ordinal=ordinal.value,
        pci=pci.value.decode(),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


def source_identity():
    paths = [
        "tools/run_kpack_selected_gate.py",
        "tools/kpack_heuristic.py",
        "tools/kpack_runtime_policy.py",
        "tools/kpack_policy.py",
        "tools/run_kpack_warmup_real.py",
        "tools/kpack_warmup_fixture.py",
        "reference/gguf_kpack.py",
        "quactlize/runtime/__init__.py",
    ]
    paths += [
        f"quactlize/runtime/{n}.py"
        for n in ("dispatch", "native", "tuning", "candidates", "compiler")
    ]
    return {p: sha(ROOT / p) for p in paths}


def completed(output, case, authority):
    path = output / "cases" / f"{case['id']}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if (
        r.get("digest") != digest({k: v for k, v in r.items() if k != "digest"})
        or r.get("authority") != authority
        or r.get("case") != case
        or r.get("status") != "PASS"
    ):
        raise ValueError(f"stale/corrupt result: {path}")
    if (
        len(r.get("errors", [])) != 3
        or any(not math.isfinite(e) or not 0 <= e < 5e-3 for e in r["errors"])
        or not r.get("direct_raw_equal")
        or not r.get("replay_raw_equal")
        or not r.get("negative_rejected")
        or not r.get("negative_raw_equal")
        or not 5e-3 <= r.get("planted_error", 0) < float("inf")
        or r.get("selection") != case["selection"]
        or r.get("grid") != case["selection"]["grid"]
        or r.get("timing_calls_during_dispatch") != 0
        or len(r.get("samples_us", [])) != 3
        or any(not math.isfinite(t) or t <= 0 for t in r["samples_us"])
        or r.get("median_us") != statistics.median(r["samples_us"])
        or r.get("validation_repeats") != 5
    ):
        raise ValueError(f"incomplete selected-dispatch validation: {path}")
    return r


class SelectedBackend(NativeBackend):
    def measure(self, handle, repeats):
        raise RuntimeError("online timing is forbidden on selected dispatch")

    def validation_sample(self, handle):
        return super().measure(handle, 5)


def run_case(case, selector, backend, weights, planes):
    import numpy as np

    request = request_of(case)
    start = time.monotonic()
    chosen = select(selector, request, backend.identity)
    selection_us = (time.monotonic() - start) * 1e6
    if chosen != case["selection"]:
        raise ValueError("live module selection differs from the frozen single tactic")
    tactic = from_config(chosen["config"])
    a, golden, denom = weights.activation(request)
    denom = np.maximum(denom, np.finfo(np.float64).tiny)
    sdk, owned, handles = backend.sdk, [], []
    errors = []

    def upload(array):
        p = sdk.upload(array.tobytes())
        owned.append(p)
        return p

    def output_error():
        raw = sdk.download(out, output_bytes)
        got = np.frombuffer(raw, dtype="<f2").reshape(golden.shape)
        error = float(np.max(np.abs(got.astype(np.float64) - golden) / denom))
        if not np.isfinite(got).all() or not math.isfinite(error):
            raise RuntimeError(
                "nonfinite device output is not a valid numerical/negative result"
            )
        return raw, error

    def check(handle):
        sdk.fill(out, 0x7B, output_bytes)
        backend.run(handle)
        backend.synchronize()
        raw, error = output_error()
        errors.append(error)
        if error >= 5e-3:
            raise RuntimeError(
                f"selected output disagrees with official GGUF oracle: error={error:.9g}"
            )
        return raw

    try:
        output_bytes = request.m * request.n * 2
        out = sdk.allocate(output_bytes)
        owned.append(out)
        buffers = dict(
            a=upload(a),
            low=planes["low"],
            high=planes["high"],
            metadata=planes["units" if request.route.startswith("fq") else "scale"],
            zero=0 if request.route.startswith("fq") else planes["zero"],
            output=out,
        )
        if request.grouped:
            buffers.update(
                rows_device=upload(np.array(request.rows, dtype=np.int32)),
                offsets_device=upload(
                    np.array([0, *np.cumsum(request.rows)], dtype=np.int32)
                ),
            )
        backend.buffers = buffers
        start = time.monotonic()
        handle = prepare_selected(backend, request, chosen)
        prepare_us = (time.monotonic() - start) * 1e6
        handles.append(handle)
        actual_grid = backend.arguments(request, tactic)[2].grid
        if actual_grid != chosen["grid"]:
            raise ValueError("live kernel grid differs from actual-row selection")
        first = check(handle)
        replay_raw_equal = first == check(handle)
        if not replay_raw_equal:
            raise RuntimeError("reused selected handle changed raw output")
        # A direct call to the same parent is a wiring control, not a second
        # configuration. It must agree bit-for-bit, in addition to the oracle.
        direct = backend.prepare(request, tactic)
        handles.append(direct)
        direct_raw_equal = first == check(direct)
        if not direct_raw_equal:
            raise RuntimeError("dispatch differs from direct same-parent launch")
        for _ in range(2):
            backend.run(handle)
        samples = [backend.validation_sample(handle) for _ in range(3)]
        if any(not math.isfinite(t) or t <= 0 for t in samples):
            raise RuntimeError("invalid validation timing")
        # Only the low-code pointer changes; a launch error/NaN is NOT credited
        # as a detected planted fault. This is outside timing and selection.
        backend.buffers = buffers | {"low": planes["zero_low"]}
        negative = prepare_selected(backend, request, chosen)
        handles.append(negative)
        sdk.fill(out, 0x7B, output_bytes)
        backend.run(negative)
        backend.synchronize()
        negative_raw, planted_error = output_error()
        # Different poison patterns expose skipped stores on the fault path;
        # a large finite sentinel is not itself proof of a decoded negative.
        sdk.fill(out, 0x55, output_bytes)
        backend.run(negative)
        backend.synchronize()
        replay_negative, _ = output_error()
        if negative_raw != replay_negative:
            raise RuntimeError("negative output depends on previous storage contents")
        if planted_error < 5e-3:
            raise RuntimeError("zero-low fault was not detected by the oracle")
        return dict(
            status="PASS",
            selection=chosen,
            errors=errors,
            replay_raw_equal=replay_raw_equal,
            direct_raw_equal=direct_raw_equal,
            negative_rejected=True,
            negative_raw_equal=True,
            planted_error=planted_error,
            selection_us=selection_us,
            prepare_us=prepare_us,
            grid=actual_grid,
            samples_us=samples,
            median_us=statistics.median(samples),
            validation_repeats=5,
            timing_calls_during_dispatch=0,
            timing_scope="SELECTED_RESIDENT_KERNEL_VALIDATION_ONLY_NO_SAME_RUN_POOL_COMPARISON",
        )
    finally:
        backend.synchronize()
        for handle in reversed(handles):
            backend.close(handle)
        for p in reversed(owned):
            sdk.free(p)


def weight_key(case):
    r = request_of(case)
    return r.qtype, r.n, r.k, len(r.rows) if r.grouped else 1


def run_weight(args, plan, records, auth, selector, index):
    from tools.kpack_warmup_fixture import Weights

    keys = list(dict.fromkeys(weight_key(c) for c in plan["cases"]))
    key = keys[index]
    cases = [c for c in plan["cases"] if weight_key(c) == key]
    pending = [c for c in cases if completed(args.output, c, digest(auth)) is None]
    if not pending:
        # Results may have been written just before a failed cleanup or a killed
        # coordinator. Replay one case to establish a clean worker exit.
        pending = cases[-1:]
        print(
            f"KPACK_SELECTED_RECHECK weight={index} reason=no_clean_worker_receipt",
            flush=True,
        )
    sdk = SDK(args.sdk)
    if device_identity(sdk) != auth["device"] or source_identity() != auth["sources"]:
        raise ValueError("worker device/source differs from campaign authority")
    print(f"KPACK_SELECTED_WEIGHT index={index} shape={key} phase=fixture", flush=True)
    started = time.monotonic()
    weights = Weights(
        *key,
        progress=lambda n, total: print(
            f"KPACK_SELECTED_FIXTURE experts={n}/{total}", flush=True
        ),
    )
    pointers, backends = {}, {}
    try:
        for name, array in weights.planes.items():
            pointers[name] = sdk.upload(array.tobytes()) if array.size else 0
        pointers["zero_low"] = sdk.allocate(weights.planes["low"].nbytes)
        sdk.fill(pointers["zero_low"], 0, weights.planes["low"].nbytes)
        fixture_seconds = time.monotonic() - started
        catalog = {p["symbol"]: p for p in plan["parents"]}
        record_map = {r["parent"]["symbol"]: r for r in records}
        for case in pending:
            r = request_of(case)
            print(
                f"KPACK_SELECTED_CASE id={case['id']} route={r.route} m={r.m} phase=dispatch",
                flush=True,
            )
            try:
                name = case["selection"]["config"]["symbol"]
                load_seconds = 0.0
                if name not in backends:
                    load_start = time.monotonic()
                    backends[name] = SelectedBackend(
                        args.sdk, [record_map[name]], {}, None, catalog=catalog
                    )
                    load_seconds = time.monotonic() - load_start
                started = time.monotonic()
                result = run_case(case, selector, backends[name], weights, pointers)
                value = dict(
                    case=case,
                    authority=digest(auth),
                    fixture_seconds=fixture_seconds,
                    module_load_seconds=load_seconds,
                    case_seconds=time.monotonic() - started,
                    **result,
                )
                value["digest"] = digest(value)
                save(args.output / "cases" / f"{case['id']}.json", value)
                print(
                    f"KPACK_SELECTED_CASE id={case['id']} status=PASS kernel_us={result['median_us']:.3f} selection_us={result['selection_us']:.3f} prepare_us={result['prepare_us']:.3f}",
                    flush=True,
                )
            except Exception as error:
                save(
                    args.output / "failures" / f"{case['id']}.json",
                    dict(case=case, authority=digest(auth), error=str(error)),
                )
                # A runtime error may poison this device context. Keep completed
                # cases and continue other weights in fresh processes.
                raise
    finally:
        for backend in backends.values():
            backend.release()
        for p in reversed(list(pointers.values())):
            sdk.free(p)
    return 0


def summarize(plan, values):
    by_id = {r["case"]["id"]: r for r in values}
    if len(by_id) != len(values) or not set(by_id) <= {c["id"] for c in plan["cases"]}:
        raise ValueError("summary contains duplicate/foreign cases")
    return dict(
        status="PASS" if len(values) == len(plan["cases"]) else "INCOMPLETE",
        expected=len(plan["cases"]),
        completed=len(values),
        parents=len(plan["parents"]),
        route_contexts=dict(Counter(r["case"]["request"]["route"] for r in values)),
        positive_checks=sum(len(r["errors"]) for r in values),
        negatives=sum(r["negative_rejected"] for r in values),
        max_error=max((max(r["errors"]) for r in values), default=None),
        timing_scope="SELECTED_MODULE_ONLY_NO_GLOBAL_PERFORMANCE_BOUND",
        online_tuning=False,
        calibration_choices_changed=False,
    )


def weight_completed(output, index, cases, authority):
    path = output / "weights" / f"{index}.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    expected = dict(authority=authority, cases=[c["id"] for c in cases], process_rc=0)
    if value != expected:
        raise ValueError("weight completion authority differs")
    return all(completed(output, case, authority) is not None for case in cases)


def campaign_summary(plan, values, output, authority, started):
    keys = list(dict.fromkeys(weight_key(c) for c in plan["cases"]))
    result = summarize(plan, values) | dict(
        wall_seconds=time.monotonic() - started, authority=authority
    )
    clean = sum(
        weight_completed(
            output, i, [c for c in plan["cases"] if weight_key(c) == key], authority
        )
        for i, key in enumerate(keys)
    )
    result.update(clean_workers=clean, expected_workers=len(keys))
    if clean != len(keys):
        result["status"] = "INCOMPLETE"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--compile-missing",
        action="store_true",
        help="explicit opt-in; otherwise cache miss stops before device work",
    )
    p.add_argument(
        "--jobs", type=int, default=32, help="only used with --compile-missing"
    )
    p.add_argument("--worker-weight", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    model, base = json.loads(MODEL.read_text()), json.loads(BASE.read_text())
    selector, plan = Selector(model, base), make_plan(model, base)
    print(
        f"KPACK_SELECTED_PLAN contexts={len(plan['cases'])} parents={len(plan['parents'])} selected_per_case=1 online_tuning=0",
        flush=True,
    )
    if args.worker_weight is not None:
        auth = json.loads((args.output / "authority.json").read_text())
        saved_plan = json.loads((args.output / "plan.json").read_text())
        records = json.loads((args.output / "modules.json").read_text())
        if (
            saved_plan != plan
            or auth["plan"] != plan["digest"]
            or auth["modules"] != digest(records)
        ):
            raise ValueError("worker plan/modules differ")
        return run_weight(args, plan, records, auth, selector, args.worker_weight)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("output is not empty; use a fresh path or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "another selected gate owns this output directory"
            ) from error
        return run_campaign(args, plan, model, p)


def run_campaign(args, plan, model, p):
    if args.plan_only:
        if (args.output / "authority.json").exists():
            p.error("plan-only cannot replace an existing campaign")
        save(args.output / "plan.json", plan)
        return 0
    if args.sdk is None or args.cache is None:
        p.error("--sdk and --cache are required")
    args.sdk, args.cache = args.sdk.resolve(), args.cache.resolve()
    contract = model["authority"]["receipt"]["compiler"]
    pinned = model["authority"]["receipt"]["sources"]
    required_sources = [
        f"quactlize/runtime/{n}.py"
        for n in ("compiler", "native", "tuning", "candidates")
    ]
    required_sources += ["reference/gguf_kpack.py", "tools/kpack_warmup_fixture.py"]
    if any(sha(ROOT / name) != pinned[name] for name in required_sources):
        raise ValueError(
            "module consumer/generator/fixture differs from calibration source"
        )
    cache_started = time.monotonic()
    records, missing = cached_records(plan, contract, args.cache)
    cache_seconds = time.monotonic() - cache_started
    compile_seconds = 0.0
    print(
        f"KPACK_SELECTED_CACHE verified={len(records)} missing={len(missing)} compile_opt_in={int(args.compile_missing)}",
        flush=True,
    )
    if missing:
        save(args.output / "missing-modules.json", missing)
        if not args.compile_missing:
            raise ValueError(
                f"{len(missing)} selected modules are missing; restore cache or explicitly use --compile-missing --resume"
            )
        compiler = Compiler(args.sdk, args.cache, args.jobs)
        if compiler.identity != contract:
            raise ValueError("compilation identity differs from the calibrated modules")
        compile_started = time.monotonic()
        compiler.compile_only(
            missing,
            lambda done, total: print(
                f"KPACK_SELECTED_COMPILE completed={done}/{total}", flush=True
            ),
        )
        compile_seconds = time.monotonic() - compile_started
        records, missing = cached_records(plan, contract, args.cache)
        if missing:
            raise ValueError("compiled selected-module cache is incomplete")
    if sdk_identity(args.sdk) != contract["sdk"]:
        raise ValueError("installed SDK differs from selected-module SDK")
    device = device_identity(SDK(args.sdk))
    auth = dict(
        plan=plan["digest"],
        modules=digest(records),
        sources=source_identity(),
        device=device,
        compiler_contract=contract,
        sdk=str(args.sdk),
    )
    receipt = args.output / "authority.json"
    if receipt.exists() and json.loads(receipt.read_text()) != auth:
        raise ValueError(
            "resume authority differs; existing results have not been changed"
        )
    if not receipt.exists() and (args.output / "cases").exists():
        raise ValueError("case results exist without a campaign authority")
    save(args.output / "plan.json", plan)
    save(receipt, auth)
    save(args.output / "modules.json", records)
    # This invocation's overhead is separate from the fixed resume authority.
    save(
        args.output / "setup-timing.json",
        dict(
            cache_verify_seconds=cache_seconds,
            compile_seconds=compile_seconds,
            compiler_invoked=compile_seconds > 0,
        ),
    )
    # Do not terminate the campaign on a failed family: every other family has
    # a fresh device process. Only complete, validated case receipts resume.
    keys = list(dict.fromkeys(weight_key(c) for c in plan["cases"]))
    started = time.monotonic()
    (args.output / "logs").mkdir(exist_ok=True)
    for index, key in enumerate(keys):
        cases = [c for c in plan["cases"] if weight_key(c) == key]
        if weight_completed(args.output, index, cases, digest(auth)):
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--sdk",
            str(args.sdk),
            "--output",
            str(args.output),
            "--worker-weight",
            str(index),
        ]
        log = args.output / "logs" / f"weight-{index}-{time.time_ns()}.log"
        with log.open("w") as f:
            proc = subprocess.Popen(
                command,
                env=dict(os.environ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                f.write(line)
                f.flush()
                print(line, end="", flush=True)
            rc = proc.wait()
        if rc == 0 and all(
            completed(args.output, c, digest(auth)) is not None for c in cases
        ):
            save(
                args.output / "weights" / f"{index}.json",
                dict(
                    authority=digest(auth), cases=[c["id"] for c in cases], process_rc=0
                ),
            )
        values = [
            r
            for c in plan["cases"]
            if (r := completed(args.output, c, digest(auth))) is not None
        ]
        summary = campaign_summary(plan, values, args.output, digest(auth), started)
        save(args.output / "summary.json", summary)
        save(args.output / "results.json", values)
        print(
            f"KPACK_SELECTED_PROGRESS weights={index+1}/{len(keys)} completed={len(values)}/{len(plan['cases'])} worker_rc={rc} elapsed_seconds={summary['wall_seconds']:.1f}",
            flush=True,
        )
    values = [
        r
        for c in plan["cases"]
        if (r := completed(args.output, c, digest(auth))) is not None
    ]
    summary = campaign_summary(plan, values, args.output, digest(auth), started)
    save(args.output / "summary.json", summary)
    save(args.output / "results.json", values)
    print("KPACK_SELECTED_DONE " + json.dumps(summary, sort_keys=True), flush=True)
    return int(summary["status"] != "PASS")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, AssertionError, OSError) as error:
        print(f"KPACK_SELECTED_FAIL {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
