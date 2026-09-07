#!/usr/bin/env python3
"""Bounded real-shape warmup plus same-run historical-incumbent confirmation.

Each weight family runs in a fresh process. Completed case receipts survive
interruptions; resume requires identical plan/source/SDK identity. All output
elements are checked, and real output/reducer/directory work is timed.
"""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha
from quactlize.runtime.native import SDK
from quactlize.runtime.tuning import (
    Request,
    Tactic,
    Tuner,
    TuningCache,
    UnsupportedTactic,
    digest,
)
from tools.run_kpack_warmup_gate import GateBackend
from tools.kpack_warmup_real_plan import make_plan, census
from tools.kpack_warmup_fixture import Weights


class RealBackend(GateBackend):
    active_tactic = None

    def arguments(self, request, tactic):
        self.active_tactic = tactic
        return super().arguments(request, tactic)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def source_identity():
    names = [
        "tools/run_kpack_warmup_real.py",
        "tools/kpack_warmup_real_plan.py",
        "tools/kpack_warmup_fixture.py",
        "tools/run_kpack_warmup_gate.py",
        "reference/gguf_kpack.py",
    ]
    names += [
        "quactlize/runtime/" + n + ".py"
        for n in ("compiler", "native", "tuning", "candidates")
    ]
    return {n: sha(ROOT / n) for n in names}


def request_of(case):
    return Request(**(case["request"] | {"rows": tuple(case["request"]["rows"])}))


def weight_key(case):
    r = request_of(case)
    return r.qtype, r.n, r.k, len(r.rows) if r.grouped else 1


def case_path(output, case):
    return output / "cases" / f"{case['id']}.json"


def completed(output, case, authority):
    path = case_path(output, case)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (
        value.get("digest") != digest({k: v for k, v in value.items() if k != "digest"})
        or value.get("authority") != authority
        or value.get("case") != case
        or value.get("status") != "PASS"
    ):
        raise ValueError(f"stale/corrupt completed case: {path}")
    numeric = value.get("numeric", {})
    if (
        not value.get("cache_replay")
        or numeric.get("checks", 0) <= 0
        or not 0 <= numeric.get("max_error", float("inf")) < 5e-3
        or not numeric.get("zero_low_rejected")
        or not 5e-3 <= numeric.get("planted_error", 0) < float("inf")
        or value.get("warmup", {}).get("status") != "MEASURED_EXACT"
    ):
        raise ValueError(f"incomplete numeric/cache result: {path}")
    confirm = value["confirmation"]
    recomputed = analyze_confirmation(
        confirm["samples_us"],
        Tactic(**value["warmup"]["tactic"]),
        Tactic(**case["incumbent"]),
    )
    if any(
        confirm.get(k) != v for k, v in recomputed.items() if k != "hinted_vs_best_pct"
    ):
        raise ValueError(f"confirmation receipt differs: {path}")
    return value


def analyze_confirmation(timing, chosen, incumbent, hinted=None, threshold=5):
    if any(
        len(v) != 2 or any(not math.isfinite(t) or t <= 0 for t in v)
        for v in timing.values()
    ):
        raise ValueError("two finite confirmation rounds are required")
    medians = {key: statistics.median(values) for key, values in timing.items()}
    if chosen.key not in medians or incumbent.key not in medians:
        raise ValueError("chosen/incumbent missing from independent confirmation")
    best = min(medians.values())
    regret = (medians[chosen.key] / best - 1) * 100
    delta = (medians[chosen.key] / medians[incumbent.key] - 1) * 100
    spread = max((max(v) / min(v) - 1) * 100 for v in timing.values())
    verdict = "WITHIN_BOUNDED_POOL_5PCT" if regret <= threshold else "BOUNDED_POOL_GAP"
    if spread > threshold:
        verdict = "TIMING_NOISE_REVIEW"
    return dict(
        verdict=verdict,
        chosen_us=medians[chosen.key],
        incumbent_us=medians[incumbent.key],
        best_pool_us=best,
        chosen_vs_best_pct=regret,
        chosen_vs_incumbent_pct=delta,
        max_round_spread_pct=spread,
        hinted_vs_best_pct=(
            (medians[hinted.key] / best - 1) * 100
            if hinted and hinted.key in medians
            else None
        ),
        medians_us=medians,
        global_performance_bound=False,
    )


def run_case(case, backend, weights, device_weights, output, tuning_options):
    import numpy as np

    started = time.monotonic()
    r = request_of(case)
    sdk = backend.sdk
    a, golden, denom = weights.activation(r)
    owned = []

    def upload(array):
        pointer = sdk.upload(array.tobytes())
        owned.append(pointer)
        return pointer

    stats = dict(checks=0, max_error=0.0)
    tolerance = np.maximum(denom, np.finfo(np.float64).tiny)
    try:
        output_bytes = r.m * r.n * 2
        out = sdk.allocate(output_bytes)
        owned.append(out)
        buffers = dict(
            a=upload(a),
            output=out,
            low=device_weights["low"],
            high=device_weights["high"],
            metadata=device_weights["units" if r.route.startswith("fq") else "scale"],
            zero=0 if r.route.startswith("fq") else device_weights["zero"],
        )
        if r.grouped:
            buffers["rows_device"] = upload(np.array(r.rows, dtype=np.int32))
            buffers["offsets_device"] = upload(
                np.array([0] + list(np.cumsum(r.rows)), dtype=np.int32)
            )
        backend.buffers = buffers
        checking_negative = False
        last_error = 0.0

        def check(_):
            nonlocal last_error
            got = np.frombuffer(sdk.download(out, output_bytes), dtype="<f2").reshape(
                golden.shape
            )
            finite = bool(np.isfinite(got).all())
            last_error = float(
                np.max(np.abs(got.astype(np.float64) - golden) / tolerance)
            )
            if not checking_negative:
                stats["checks"] += 1
                stats["max_error"] = (
                    max(stats["max_error"], last_error) if finite else float("inf")
                )
            if not finite or not math.isfinite(last_error) or last_error >= 5e-3:
                print(
                    f"KPACK_REAL_NUMERIC id={case['id']} tactic={backend.active_tactic} negative={int(checking_negative)} finite={int(finite)} error={last_error}",
                    flush=True,
                )
            return finite and math.isfinite(last_error) and last_error < 5e-3

        backend.correctness = check
        cache = TuningCache(
            backend.identity, output / "timing-cache" / f"q{r.qtype}-{r.route}.json"
        )
        tuner = Tuner(cache, **tuning_options)
        candidates = [Tactic(**t) for t in case["candidates"]]
        incumbent = Tactic(**case["incumbent"])
        before = tuner.select(r, backend)
        hinted = before.get("tactic")
        if hinted:
            h = backend.prepare(r, hinted)
            try:
                backend.check(h)
            finally:
                backend.synchronize()
                backend.close(h)
        tuned = tuner.warmup(r, candidates, backend, force=True)
        if tuned["status"] != "MEASURED_EXACT" or incumbent.key in tuned["rejected"]:
            raise ValueError("warmup failed to measure its historical incumbent")
        chosen = tuned["tactic"]
        replay = Tuner(
            TuningCache(backend.identity, cache.path), **tuning_options
        ).select(r, backend)
        if replay.get("tactic") != chosen:
            raise ValueError("cache replay does not match warmup choice")

        # Validation work is separate from the 100-ms warmup. Check/timestamp
        # the whole bounded pool, including candidates the soft budget missed.
        confirm = list(dict.fromkeys(candidates + ([hinted] if hinted else [])))
        timing = {}
        resources = {}
        rejected = set()
        batch_repeats = {}
        for round_id, order in enumerate((confirm, list(reversed(confirm)))):
            for tactic in order:
                h = None
                try:
                    _, _, recipe, resource, _ = backend.arguments(r, tactic)
                    resources[tactic.key] = dict(
                        workspace_bytes=resource.workspace_bytes,
                        shared_bytes=resource.shared_bytes,
                        occupancy=resource.occupancy,
                        grid=recipe.grid,
                    )
                    h = backend.prepare(r, tactic)
                    backend.check(h)
                    backend.run(h)
                    backend.run(h)
                    if tactic.key not in batch_repeats:
                        probe = backend.measure(h, 1)
                        if not math.isfinite(probe) or probe <= 0:
                            raise RuntimeError("invalid confirmation probe")
                        batch_repeats[tactic.key] = max(
                            1, min(5, math.ceil(1000 / probe))
                        )
                    us = backend.measure(h, batch_repeats[tactic.key])
                    timing.setdefault(tactic.key, []).append(us)
                except UnsupportedTactic:
                    if tactic in (incumbent, chosen):
                        raise ValueError(
                            "incumbent/chosen was rejected by confirmation"
                        )
                    rejected.add(tactic.key)
                finally:
                    if h is not None:
                        backend.synchronize()
                        backend.close(h)
        if rejected.intersection(timing):
            raise ValueError("candidate admission changed between confirmation rounds")
        comparison = analyze_confirmation(timing, chosen, incumbent, hinted)

        checking_negative = True
        backend.buffers["low"] = device_weights["zero_low"]
        h = backend.prepare(r, chosen)
        negative = False
        try:
            try:
                backend.check(h)
            except RuntimeError as error:
                if str(error) != "candidate correctness check failed":
                    raise
                negative = True
        finally:
            backend.synchronize()
            backend.close(h)
            backend.buffers["low"] = device_weights["low"]
        if not negative or not math.isfinite(last_error) or last_error < 5e-3:
            raise ValueError("finite zero-low negative was not detected")
        tuned["tactic"] = asdict(chosen)
        return dict(
            status="PASS",
            case_seconds=time.monotonic() - started,
            warmup=tuned,
            cache_status_before=before["status"],
            cache_replay=True,
            confirmation=dict(
                samples_us=timing,
                resources=resources,
                rejected=sorted(rejected),
                repeats=batch_repeats,
                tactics={t.key: asdict(t) for t in confirm},
                **comparison,
            ),
            numeric=stats | dict(zero_low_rejected=True, planted_error=last_error),
            identity=backend.identity,
        )
    finally:
        backend.synchronize()
        for p in reversed(owned):
            sdk.free(p)
        backend.buffers = {}


def run_weight(args, plan, records, authority, index):
    keys = list(dict.fromkeys(weight_key(c) for c in plan["cases"]))
    key = keys[index]
    cases = [c for c in plan["cases"] if weight_key(c) == key]
    pending = [c for c in cases if completed(args.output, c, authority) is None]
    if not pending:
        return 0
    if (
        json.loads((args.output / "authority.json").read_text())["sources"]
        != source_identity()
    ):
        raise ValueError("runner sources changed during campaign")
    q, n, k, e = key
    started = time.monotonic()
    print(
        f"KPACK_REAL_WEIGHT index={index} q={q} n={n} k={k} experts={e} phase=prepare",
        flush=True,
    )
    weights = Weights(
        q,
        n,
        k,
        e,
        lambda done, total: print(
            f"KPACK_REAL_FIXTURE experts={done}/{total}", flush=True
        ),
    )
    sdk = SDK(args.sdk)
    pointers = {}
    backends = {}
    try:
        for name, array in weights.planes.items():
            pointers[name] = sdk.upload(array.tobytes()) if array.size else 0
        pointers["zero_low"] = sdk.allocate(weights.planes["low"].nbytes)
        sdk.fill(pointers["zero_low"], 0, weights.planes["low"].nbytes)
        fixture_seconds = time.monotonic() - started
        print(
            f"KPACK_REAL_WEIGHT index={index} phase=ready seconds={fixture_seconds:.3f}",
            flush=True,
        )
        # The full plan catalog binds the cache; loading a subset is not a new
        # inventory. Keep earlier-bucket parent modules available for replay.
        catalog = {p["symbol"]: p for p in plan["parents"]}
        for c in pending:
            route = c["request"]["route"]
            if route not in backends:
                names = {
                    t["parent"]
                    for other in cases
                    if other["request"]["route"] == route
                    for t in other["candidates"]
                }
                selected = [m for m in records if m["parent"]["symbol"] in names]
                backends[route] = RealBackend(
                    args.sdk, selected, {}, None, catalog=catalog
                )
                if (
                    backends[route].identity["device"] != "PPU-ZW810"
                    or backends[route].identity["compute_units"] != 72
                ):
                    raise ValueError("real-shape gate requires ZW810/72 CUs")
            r = request_of(c)
            print(
                f"KPACK_REAL_CASE id={c['id']} route={route} m={r.m} phase=warmup",
                flush=True,
            )
            try:
                result = run_case(
                    c, backends[route], weights, pointers, args.output, plan["tuning"]
                )
            except Exception as error:
                tactic = backends[route].active_tactic
                save(
                    args.output / "failures" / f"{c['id']}.json",
                    dict(
                        case=c,
                        authority=authority,
                        error=str(error),
                        active_tactic=asdict(tactic) if tactic else None,
                    ),
                )
                raise
            value = dict(
                case=c, authority=authority, fixture_seconds=fixture_seconds, **result
            )
            value["digest"] = digest(value)
            save(case_path(args.output, c), value)
            comp = result["confirmation"]
            print(
                f"KPACK_REAL_CASE id={c['id']} status=PASS tuning_ms={result['warmup']['tuning_ms']:.3f} verdict={comp['verdict']} vs_best_pct={comp['chosen_vs_best_pct']:.3f} vs_incumbent_pct={comp['chosen_vs_incumbent_pct']:.3f}",
                flush=True,
            )
    finally:
        for backend in backends.values():
            backend.release()
        for p in reversed(list(pointers.values())):
            sdk.free(p)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--formats", type=int, nargs="+", default=list(range(10, 15)))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-weight", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_weight is not None:
        plan = json.loads((args.output / "plan.json").read_text())
        auth = json.loads((args.output / "authority.json").read_text())
        records = json.loads((args.output / "modules.json").read_text())
        if (
            plan.get("digest")
            != digest({k: v for k, v in plan.items() if k != "digest"})
            or auth["plan"] != plan["digest"]
        ):
            raise ValueError("worker plan/authority differs")
        return run_weight(args, plan, records, digest(auth), args.worker_weight)
    plan = make_plan(args.formats)
    print("KPACK_REAL_PLAN " + json.dumps(census(plan), sort_keys=True), flush=True)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error(
            "output is not empty; use a new directory or --resume with unchanged inputs"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        if (args.output / "authority.json").exists():
            parser.error("plan-only cannot replace an admitted campaign plan")
        save(args.output / "plan.json", plan)
        return 0
    if args.sdk is None or args.cache is None:
        parser.error("--sdk and --cache are required")
    compiler = Compiler(args.sdk, args.cache, args.jobs)
    auth = dict(
        plan=plan["digest"],
        compiler=compiler.identity,
        sources=source_identity(),
        numpy=importlib.metadata.version("numpy"),
        gguf=importlib.metadata.version("gguf"),
    )
    receipt = args.output / "authority.json"
    if not receipt.exists() and any(
        p.name != "plan.json" for p in args.output.iterdir()
    ):
        raise ValueError("existing campaign files have no authority receipt")
    if receipt.exists() and json.loads(receipt.read_text()) != auth:
        raise ValueError(
            "resume source/plan/SDK identity differs; completed results were not modified"
        )
    save(args.output / "plan.json", plan)
    save(receipt, auth)
    started = time.monotonic()
    records = compiler.compile_only(
        plan["parents"],
        lambda done, total: print(
            f"KPACK_REAL_COMPILE completed={done}/{total}", flush=True
        ),
    )
    build_seconds = time.monotonic() - started
    save(args.output / "modules.json", records)
    print(f"KPACK_REAL_COMPILE status=PASS seconds={build_seconds:.3f}", flush=True)
    if args.compile_only:
        return 0
    keys = list(dict.fromkeys(weight_key(c) for c in plan["cases"]))
    for index, key in enumerate(keys):
        group = [c for c in plan["cases"] if weight_key(c) == key]
        if all(completed(args.output, c, digest(auth)) is not None for c in group):
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--sdk",
            str(args.sdk),
            "--worker-weight",
            str(index),
        ]
        rc = subprocess.run(command, env=dict(os.environ)).returncode
        count = sum(
            completed(args.output, c, digest(auth)) is not None for c in plan["cases"]
        )
        print(
            f'KPACK_REAL_PROGRESS weights={index+1}/{len(keys)} completed={count}/{len(plan["cases"])} worker_rc={rc} elapsed_seconds={time.monotonic()-started:.1f}',
            flush=True,
        )
        # A failed worker does not contaminate the next family's device context.
    values = [
        v
        for c in plan["cases"]
        if (v := completed(args.output, c, digest(auth))) is not None
    ]
    verdicts = {
        k: sum(v["confirmation"]["verdict"] == k for v in values)
        for k in ("WITHIN_BOUNDED_POOL_5PCT", "BOUNDED_POOL_GAP", "TIMING_NOISE_REVIEW")
    }
    summary = dict(
        status=(
            "NUMERICS_PASS_PERFORMANCE_REVIEW_REQUIRED"
            if len(values) == len(plan["cases"])
            else "INCOMPLETE"
        ),
        expected=len(plan["cases"]),
        completed=len(values),
        parents=len(records),
        verdicts=verdicts,
        build_seconds=build_seconds,
        wall_seconds=time.monotonic() - started,
        scope=plan["scope"],
        authority=digest(auth),
        global_performance_bound=False,
    )
    save(args.output / "summary.json", summary)
    save(args.output / "results.json", values)
    print("KPACK_REAL_DONE " + json.dumps(summary, sort_keys=True), flush=True)
    return int(len(values) != len(plan["cases"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, AssertionError, OSError) as error:
        print(f"KPACK_REAL_FAIL {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
