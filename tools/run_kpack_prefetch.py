#!/usr/bin/env python3
"""Separate cache benefit from concurrent-prefetch cost, without profiler replay.

Synthetic next weights have independent addresses and permuted expert contents.
Their active experts are assumed known: this is not a next-layer MoE router.
The measured endpoint includes GPU directory/metadata and any Split-K reducer.
"""

import argparse
import copy
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import (
    SDK,
    Module,
    Call,
    Recipe,
    Resources as Query,
    checked,
)
from quactlize.execution.native import arrangement
from tools.build_kpack_prefetch import (
    SCHEMA,
    subjects,
    sdk_files,
    validate_sdk_files,
    runtime_sdk_report,
)
from tools.kpack_execution_fixture import IndexedWeights
from tools.run_kpack_decode_sweep import bind_group, admit
from tools.run_kpack_gemv_gate import Resources, fixture
from tools.run_kpack_grouped_device_gate import DeviceCall, graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity


class Range(C.Structure):
    _fields_ = [("pointer", C.c_uint64), ("bytes", C.c_uint64)]


# PPU SDK include/driver_types.h. External RECORD nodes supply timestamps;
# cross-stream dependencies still use separate, ordinary capture events.
EVENT_RECORD_EXTERNAL = 0x01
GRAPH_EVENT_RECORD = 0x07
PAIR_KINDS = ("pair", "primed-pair")
CASES = ("q12", "q13", "q4-to-q5")


def select_subject(manifest, case):
    rows = {s["case"]: s for s in manifest["subjects"]}
    if case == "q4-to-q5":
        current, target = rows["q12"], rows["q13"]
        if (current["n"], current["k"]) != (target["k"], target["n"]):
            raise ValueError("gate/up and down projection dimensions do not match")
        return target | dict(case=case, current_subject=current)
    return rows[case]


def timing_bind(sdk):
    for name, args in {
        "hggcEventRecordWithFlags": [C.c_void_p, C.c_void_p, C.c_uint],
        "hggcEventQuery": [C.c_void_p],
        "hggcStreamWaitEvent": [C.c_void_p, C.c_void_p, C.c_uint],
        "hggcGraphGetNodes": [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
        "hggcGraphNodeGetType": [C.c_void_p, C.POINTER(C.c_int)],
        "hggcGraphEventRecordNodeGetEvent": [C.c_void_p, C.POINTER(C.c_void_p)],
    }.items():
        fn = getattr(sdk.lib, name)
        fn.argtypes, fn.restype = args, C.c_int


def validate_timing_nodes(sdk, graph, events):
    count = C.c_size_t()
    checked(sdk.lib.hggcGraphGetNodes(graph, None, C.byref(count)), "graph node count")
    nodes = (C.c_void_p * count.value)()
    checked(sdk.lib.hggcGraphGetNodes(graph, nodes, C.byref(count)), "graph nodes")
    recorded = []
    for node in nodes:
        kind = C.c_int()
        checked(sdk.lib.hggcGraphNodeGetType(node, C.byref(kind)), "graph node type")
        if kind.value == GRAPH_EVENT_RECORD:
            event = C.c_void_p()
            checked(
                sdk.lib.hggcGraphEventRecordNodeGetEvent(node, C.byref(event)),
                "graph timing event",
            )
            recorded.append(event.value)
    missing = [
        name for name, event in events.items() if recorded.count(event.value) != 1
    ]
    if missing:
        raise ValueError(
            "captured timing events require exactly one record node: "
            + ",".join(missing)
        )
    return len(events)


def weight_ranges(planes, sizes, experts, active):
    if (
        len(active) != len(set(active))
        or not active
        or any(e < 0 or e >= experts for e in active)
    ):
        raise ValueError("invalid active experts")
    result = []
    for name in ("low", "high", "units"):
        size = sizes[name]
        if not size:
            continue
        if (
            size % experts
            or size // experts % 32
            or not planes[name]
            or planes[name] % 32
        ):
            raise ValueError("unaligned/non-concatenated weight plane")
        per_expert = size // experts
        result.extend((planes[name] + e * per_expert, per_expert) for e in active)
    if not result or len(result) > 24:
        raise ValueError("unexpected prefetch range count")
    return result


def disjoint(first, second):
    if any(a < b + nb and b < a + na for a, na in first for b, nb in second):
        raise ValueError("current/next weight allocations alias")


def check_receipt(sdk, sink, blocks, target, mode):
    words = np.frombuffer(sdk.download(sink, blocks * 128 * 8), dtype="<u8")
    touches = int((words >> np.uint64(32)).sum())
    checksum = int(np.bitwise_xor.reduce(words & np.uint64(0xFFFFFFFF)))
    if touches != sum(size // 32 for _, size in target.ranges) or checksum != (
        target.checksum if mode == "load" else 0
    ):
        raise ValueError(
            f"prefetch receipt differs: mode={mode} touches={touches} checksum={checksum}"
        )


def intervals(times, pair, prefetch):
    if any(not math.isfinite(v) or v < 0 for v in times.values()):
        raise ValueError("invalid event timestamps")

    def duration(a, b):
        value = times[b] - times[a]
        if value <= 0:
            raise ValueError("nonpositive event interval: " + a)
        return value

    out = dict(target_us=duration("next_start", "next_end"), total_us=times["next_end"])
    if pair:
        out["current_us"] = duration("current_start", "current_end")
        if times["next_start"] < times["current_end"]:
            raise ValueError("target precedes current completion")
    if prefetch:
        out["prefetch_us"] = duration("prefetch_start", "prefetch_end")
        if times["next_start"] < times["prefetch_end"]:
            raise ValueError("target precedes prefetch join")
        if pair:
            out["envelope_overlap_us"] = max(
                0.0,
                min(times["current_end"], times["prefetch_end"])
                - max(times["current_start"], times["prefetch_start"]),
            )
    return out


def verify(bundle):
    bundle = bundle.resolve(strict=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("production_selection_changed") is not False
    ):
        raise ValueError("not the cache experiment")
    validate_sdk_files(manifest)
    if manifest.get("subjects") != subjects():
        raise ValueError("experiment differs from the two measured FQ anchors")
    library = (bundle / manifest["library"]).resolve(strict=True)
    if not library.is_relative_to(bundle) or sha(library) != manifest["sha256"]:
        raise ValueError("prefetch helper path/hash differs")
    for relative, expected in manifest["source_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != expected:
            raise ValueError("probe source differs: " + relative)
    for subject in manifest["subjects"]:
        record = subject["module"]
        path = (ROOT / record["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != record["sha256"]:
            raise ValueError("measured GEMM parent path/hash differs")
    return manifest, library


def admit_runtime_sdk(args, manifest):
    report = runtime_sdk_report(
        manifest,
        sdk_files(args.sdk, optional_tools=True),
        allow_unverified=args.allow_unverified_sdk,
    )
    (args.output / (args.case + ".sdk.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "KPACK_PREFETCH_SDK "
        + json.dumps(
            dict(
                case=args.case,
                status=report["status"],
                allowed=report["allowed"],
                differences=report["differences"],
            )
        ),
        flush=True,
    )
    if not report["allowed"]:
        changed = ", ".join(row["path"] for row in report["differences"])
        raise ValueError(
            f"SDK runtime libraries differ ({changed}); use matching libraries or "
            "--allow-unverified-sdk for an explicitly unverified experiment. "
            "GPU launch and numerical checks remain mandatory."
        )
    return report


class Prepared:
    def __init__(self, sdk, r, subject, w, shift):
        self.sdk, self.r = sdk, r
        record = subject["module"]
        self.module = Module(record | {"path": str(ROOT / record["path"])})
        self.handle = C.c_void_p()
        view = copy.copy(w)
        view.sums = np.roll(w.sums, shift, axis=0)
        view.abs_sums = np.roll(w.abs_sums, shift, axis=0)
        self.data = fixture(
            view, dict(mode=2, rows=8, topk=8, channels=1 if w.q == 12 else 8)
        )
        self.planes, self.sizes, self.checksum = {}, {}, 0
        self.active = [int(e) for e in self.data["expert"]]
        self.hashes = {}
        for name in ("low", "high", "units"):
            data = np.ascontiguousarray(np.roll(w.planes[name], shift, axis=0))
            self.sizes[name] = data.nbytes
            self.planes[name] = r.upload(data) if data.size else None
            self.hashes[name] = hashlib.sha256(data.tobytes()).hexdigest()
            if data.size:
                words = data.reshape(w.experts, -1).view("<u4")[:, ::8]
                self.checksum ^= int(
                    np.bitwise_xor.reduce(words[self.active].reshape(-1))
                )
        self.ranges = weight_ranges(self.planes, self.sizes, w.experts, self.active)
        rows = np.bincount(self.active, minlength=w.experts).astype("i4")
        offsets = np.r_[0, rows.cumsum()].astype("i4")
        self.order = np.argsort(self.data["expert"], kind="stable")
        self.output_bytes = 8 * w.n * 2
        self.output = r.alloc(self.output_bytes + 32)
        self.expected_bits = None
        arr = arrangement(w.q)
        dev = self.module.device_identity()
        call = Call(
            version=1,
            size=C.sizeof(Call),
            m=8,
            n=w.n,
            k=w.k,
            experts=w.experts,
            group_size=arr.group_size,
            device=dev["ordinal"],
            compute_units=dev["compute_units"],
            mapping_id=arr.mapping_id,
            a=r.upload(self.data["a"][self.data["arows"][self.order]]),
            low=self.planes["low"],
            high=self.planes["high"],
            metadata=self.planes["units"],
            output=self.output + 16,
            offsets_device=r.upload(offsets),
            stream=r.stream.value,
        )
        self.recipe = Recipe(1, C.sizeof(Recipe), 0, subject["split"], 0)
        self.query = Query()
        device_call = DeviceCall(2, C.sizeof(DeviceCall), call, 1, 0)
        query, prepare = bind_group(self.module)
        checked(
            query(C.byref(device_call), C.byref(self.recipe), C.byref(self.query)),
            "compact query",
        )
        self.workspace = r.alloc(self.query.workspace_bytes + 32)
        r.fill(self.workspace, 0xA5, self.query.workspace_bytes + 32)
        device_call.call.workspace = self.workspace + 16
        device_call.call.workspace_bytes = self.query.workspace_bytes
        sdk.synchronize(None)
        sdk.synchronize(r.stream)
        checked(
            prepare(C.byref(device_call), C.byref(self.recipe), C.byref(self.handle)),
            "compact prepare",
        )
        sdk.synchronize(r.stream)

    def launch(self):
        return self.module.run(self.handle, self.r.stream)

    def poison(self):
        self.r.fill(self.output, 0xA5, self.output_bytes + 32)
        self.r.fill(self.output + 16, 0xFF, self.output_bytes)

    def check(self):
        raw = self.sdk.download(self.output, self.output_bytes + 32)
        if (
            raw[:16] != b"\xa5" * 16
            or raw[-16:] != b"\xa5" * 16
            or self.sdk.download(self.workspace, 16) != b"\xa5" * 16
            or self.sdk.download(self.workspace + 16 + self.query.workspace_bytes, 16)
            != b"\xa5" * 16
        ):
            raise ValueError("output/workspace guard changed")
        got = np.frombuffer(raw[16:-16], dtype="<f2").reshape(8, -1)[
            np.argsort(self.order)
        ]
        error = admit(got, self.data, "prefetch experiment output")
        if self.expected_bits is None:
            self.expected_bits = got.tobytes()
        elif self.expected_bits != got.tobytes():
            raise ValueError("prefetch changed GEMM output bits")
        return error

    def close(self):
        if self.handle:
            self.module.destroy(self.handle)
            self.handle = C.c_void_p()


class Probe:
    def __init__(self, library, sdk):
        self.sdk = sdk
        self.lib = C.CDLL(str(library), mode=C.RTLD_LOCAL)
        self.prefetch = self.lib.kpack_prefetch_v1
        self.prefetch.argtypes, self.prefetch.restype = [
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
        ], C.c_int
        self.pressure = self.lib.kpack_cache_pressure_v1
        self.pressure.argtypes, self.pressure.restype = [
            C.c_void_p,
            C.c_uint64,
            C.c_void_p,
            C.c_void_p,
        ], C.c_int
        query = self.lib.kpack_prefetch_device_v1
        query.argtypes, query.restype = [C.POINTER(C.c_int64), C.c_int], C.c_int
        values = (C.c_int64 * 8)()
        checked(query(values, 8), "prefetch device query")
        self.device = dict(
            zip(
                (
                    "compute_units",
                    "warp_size",
                    "l2_bytes",
                    "shared_bytes_per_cu",
                    "registers_per_cu",
                    "threads_per_cu",
                    "concurrent_kernels",
                    "ordinal",
                ),
                values,
            )
        )
        if self.device["compute_units"] != 72 or self.device["warp_size"] != 32:
            raise ValueError("experiment requires the measured 72-CU device")


class Experiment:
    def __init__(
        self,
        sdk,
        r,
        pf,
        probe,
        current,
        target,
        pressure,
        pressure_bytes,
        pressure_sink,
        ranges,
        sink,
        kind,
        mode,
        blocks,
    ):
        self.sdk, self.r, self.pf = sdk, r, pf
        self.events, self.fences = {}, {}
        self.kind = kind
        self.prefetching = mode != "none" and kind != "primed-pair"
        if kind == "primed-pair" and mode != "load":
            raise ValueError("primed-pair requires the completed-load control")
        self.label = f"kind={kind} mode={mode} blocks={blocks}"
        self.graph = None
        names = ["origin", "next_start", "next_end"]
        if kind in PAIR_KINDS:
            names += ["current_start", "current_end"]
        if self.prefetching:
            names += ["prefetch_start", "prefetch_end"]
        try:
            for destination, event_names in (
                (self.events, names),
                (
                    self.fences,
                    ["fork", "join"] if kind in PAIR_KINDS and self.prefetching else [],
                ),
            ):
                for name in event_names:
                    event = C.c_void_p()
                    checked(
                        sdk.lib.hggcEventCreate(C.byref(event)), "event create " + name
                    )
                    destination[name] = event
        except BaseException:
            self.close()
            raise

        def record(name, stream):
            checked(
                sdk.lib.hggcEventRecordWithFlags(
                    self.events[name], stream, EVENT_RECORD_EXTERNAL
                ),
                "capture timing record " + name,
            )

        def launch_prefetch(stream):
            checked(
                probe.prefetch(
                    ranges,
                    len(target.ranges),
                    int(mode == "hint"),
                    blocks,
                    sink,
                    stream,
                ),
                "prefetch",
            )

        def prefetch(stream):
            record("prefetch_start", stream)
            launch_prefetch(stream)
            record("prefetch_end", stream)

        def capture():
            checked(
                probe.pressure(pressure, pressure_bytes, pressure_sink, r.stream),
                "cache pressure",
            )
            if kind == "primed-pair":
                # An optimistic control, not a net-speedup measurement: the
                # same weight reads finish before either current or target is
                # timed. No full target GEMM is used to prime its other state.
                launch_prefetch(r.stream)
            record("origin", r.stream)
            if kind in PAIR_KINDS:
                if self.prefetching:
                    # Keep dependency events separate from timestamp nodes.
                    # No external wait and no dependency on uncaptured work.
                    checked(
                        sdk.lib.hggcEventRecord(self.fences["fork"], r.stream),
                        "capture fork record",
                    )
                    checked(
                        sdk.lib.hggcStreamWaitEvent(pf.stream, self.fences["fork"], 0),
                        "capture fork",
                    )
                    prefetch(pf.stream)
                    checked(
                        sdk.lib.hggcEventRecord(self.fences["join"], pf.stream),
                        "capture join record",
                    )
                record("current_start", r.stream)
                checked(current.launch(), "current selected call")
                record("current_end", r.stream)
                if self.prefetching:
                    checked(
                        sdk.lib.hggcStreamWaitEvent(r.stream, self.fences["join"], 0),
                        "capture join",
                    )
            elif self.prefetching:
                prefetch(r.stream)
            elif kind == "warm":
                checked(target.launch(), "warm-weight control")
            record("next_start", r.stream)
            checked(target.launch(), "target selected call")
            record("next_end", r.stream)
            return 0

        try:
            self.graph = Replay(sdk, r.stream, capture, 1)
            self.timing_nodes = validate_timing_nodes(
                sdk, self.graph.graph, self.events
            )
        except BaseException:
            self.close()
            raise

    def sample(self):
        checked(self.graph(), "experimental graph replay")
        self.sdk.synchronize(self.r.stream)
        result = {}
        for name, event in self.events.items():
            if name == "origin":
                continue
            elapsed = C.c_float()
            status = self.sdk.lib.hggcEventElapsedTime(
                C.byref(elapsed), self.events["origin"], event
            )
            if status:
                begin_query = self.sdk.lib.hggcEventQuery(self.events["origin"])
                end_query = self.sdk.lib.hggcEventQuery(event)
                raise RuntimeError(
                    f"event interval {self.label} origin->{name} failed: status={status} "
                    f"event_query=[{begin_query},{end_query}] timing_nodes={self.timing_nodes}"
                )
            result[name] = float(elapsed.value) * 1000
        return intervals(result, self.kind in PAIR_KINDS, self.prefetching)

    def close(self):
        if self.graph:
            self.graph.close()
            self.graph = None
        for event in (*self.events.values(), *self.fences.values()):
            checked(self.sdk.lib.hggcEventDestroy(event), "destroy graph event")
        self.events.clear()
        self.fences.clear()


def timing_self_test(sdk, output):
    """Exercise the actual timeline with small memsets, before GGUF fixtures.

    This admits event-node timestamps and capture fork/join only. Its times
    are not kernel performance results or proof of concurrent kernel execution.
    """
    r, pf = Resources(sdk), Resources(sdk)
    result = dict(status="FAIL", timing="EXPLICIT_GRAPH_RECORD_NODES", arms=[])
    size = 4 * 1024**2
    try:
        print("KPACK_PREFETCH_TIMING_BEGIN before_weight_fixture=1", flush=True)
        pressure, target, prefetched = (r.alloc(size) for _ in range(3))

        def fill(pointer, value, stream):
            return sdk.lib.hggcMemsetAsync(pointer, value, size, stream)

        current = SimpleNamespace(launch=lambda: fill(target, 0x22, r.stream))
        next_call = SimpleNamespace(
            launch=lambda: fill(target, 0x33, r.stream), ranges=[(prefetched, size)]
        )
        probe = SimpleNamespace(
            pressure=lambda *args: fill(pressure, 0x11, r.stream),
            prefetch=lambda *args: fill(prefetched, 0x44, args[-1]),
        )
        for key in (
            ("cold", "none", 0),
            ("warm", "none", 0),
            ("pair", "none", 0),
            ("sequential", "load", 1),
            ("pair", "load", 1),
            ("primed-pair", "load", 1),
        ):
            result["current_arm"] = dict(kind=key[0], mode=key[1])
            r.fill(target, 0xA5, size)
            r.fill(prefetched, 0xA5, size)
            sdk.synchronize(r.stream)
            graph = Experiment(
                sdk, r, pf, probe, current, next_call, pressure, size, 0, 0, 0, *key
            )
            try:
                values = [graph.sample() for _ in range(3)]
                if sdk.download(target, size) != b"\x33" * size:
                    raise ValueError("timing self-test target write differs")
                expected = 0x44 if key[1] != "none" else 0xA5
                if sdk.download(prefetched, size) != bytes([expected]) * size:
                    raise ValueError("timing self-test fork/join write differs")
                result["arms"].append(
                    dict(
                        kind=key[0],
                        mode=key[1],
                        timing_nodes=graph.timing_nodes,
                        samples=values,
                    )
                )
            finally:
                graph.close()
        result["status"] = "PASS"
        result.pop("current_arm")
        print(
            "KPACK_PREFETCH_TIMING PASS arms=6 graph_replays=18 before_weight_fixture=1",
            flush=True,
        )
        return result
    finally:
        output.write_text(json.dumps(result, indent=2) + "\n")
        pf.close()
        r.close()


def configurations(blocks, *, include_primed=False):
    result = [("cold", "none", 0), ("warm", "none", 0), ("pair", "none", 0)] + [
        (kind, mode, count)
        for kind in ("sequential", "pair")
        for mode in ("hint", "load")
        for count in blocks
    ]
    if include_primed:
        result.append(("primed-pair", "load", max(blocks)))
    return result


def summary(cells):
    grouped = {}
    for c in cells:
        grouped.setdefault((c["kind"], c["mode"], c["blocks"]), []).extend(c["samples"])
    result = []
    for (kind, mode, blocks), values in grouped.items():
        row = dict(kind=kind, mode=mode, blocks=blocks, samples=len(values))
        row.update(
            {name: statistics.median(v[name] for v in values) for name in values[0]}
        )
        result.append(row)
    cold = next(r for r in result if r["kind"] == "cold")
    pair = next(r for r in result if r["kind"] == "pair" and r["mode"] == "none")
    for row in result:
        base = pair if row["kind"] in PAIR_KINDS else cold
        row["target_delta_pct"] = 100 * (row["target_us"] / base["target_us"] - 1)
        row["total_delta_pct"] = 100 * (row["total_us"] / base["total_us"] - 1)
        if row["kind"] in PAIR_KINDS:
            row["current_delta_pct"] = 100 * (
                row["current_us"] / base["current_us"] - 1
            )
        if row["kind"] == "pair" and row["mode"] != "none":
            row["overlap_scope"] = "CALL_ENVELOPES_NOT_PRODUCER_CONCURRENCY_PROOF"
        if row["kind"] == "primed-pair":
            row["preload_cost_excluded"] = True
            row["metric_scope"] = "OPTIMISTIC_CURRENT_PLUS_TARGET_NOT_NET_LATENCY"
    return result


def run(args, subject, manifest, library, runtime_sdk):
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    timing_bind(sdk)
    timing_probe = timing_self_test(sdk, args.output / (args.case + ".timing.json"))
    current_subject = subject.get("current_subject", subject)
    cross_projection = subject["case"] == "q4-to-q5"
    probe = Probe(library, sdk)
    if probe.device["l2_bytes"] <= 0 and not args.pressure_mib:
        raise ValueError("SDK reports no L2 size; explicitly choose --pressure-mib")
    pressure_bytes = (
        args.pressure_mib * 1024**2
        if args.pressure_mib
        else max(256 * 1024**2, 4 * probe.device["l2_bytes"])
    )
    if pressure_bytes > 1024**3 or pressure_bytes < 4 * probe.device["l2_bytes"]:
        raise ValueError(
            "cache pressure must be at least 4x reported L2 and at most 1 GiB"
        )
    result = dict(
        status="FAIL",
        case=subject["case"],
        subject=subject,
        current_subject=current_subject,
        device=device_identity(sdk),
        device_resources=probe.device,
        cells=[],
        manifest_sha256=sha(args.bundle / "manifest.json"),
        runner_sha256=sha(__file__),
        runtime_sdk=runtime_sdk,
        timing_probe=timing_probe,
        cache_initialization="PRESSURE_EVICTION_NOT_PROVEN_FLUSH",
        pressure_bytes=pressure_bytes,
        timing="UNPROFILED_EXPLICIT_GRAPH_EVENTS_FULL_COMPACT_CALL",
        instrumentation="TIMESTAMP_NODES_PRESENT_IN_ALL_ARMS_NOT_ZERO_OVERHEAD",
        known_next_experts=(
            "SAME_PRECOMPUTED_TOP8_IDS" if cross_projection else "ASSUMED_NOT_PREDICTED"
        ),
        activation_dependency="SYNTHETIC_DOWN_INPUT_NOT_A_FULL_MLP",
        current_projection_calls=1,
        hint_completion="ISSUANCE_ONLY_LOAD_CONTROL_WAITS_FOR_READS",
        excluded="LLAMA_ADAPTERS_ROUTER_PREDICTION_AND_ACTIVATION_DEPENDENCY",
    )
    r, pf = Resources(sdk), Resources(sdk)
    current = target = None
    graphs = {}
    path = args.output / (subject["case"] + ".json")
    try:
        print(
            f"KPACK_PREFETCH_FIXTURE case={subject['case']} E=256 active=8", flush=True
        )

        def weights(spec, role):
            return IndexedWeights(
                spec["q"],
                spec["n"],
                spec["k"],
                256,
                progress=lambda done, total: print(
                    f"KPACK_PREFETCH_FIXTURE role={role} q={spec['q']} experts={done}/{total}",
                    flush=True,
                ),
            )

        w = weights(subject, "target")
        current_w = weights(current_subject, "current") if cross_projection else w
        current = Prepared(sdk, r, current_subject, current_w, 0)
        target = Prepared(sdk, r, subject, w, 0 if cross_projection else 1)
        if current.active != target.active or len(target.active) != 8:
            raise ValueError("current and down must use the same eight expert IDs")
        disjoint(
            [(p, current.sizes[n]) for n, p in current.planes.items() if p],
            [(p, target.sizes[n]) for n, p in target.planes.items() if p],
        )
        for item in (current, target):
            item.poison()
            checked(item.launch(), "selected parent eager warmup")
            sdk.synchronize(r.stream)
            item.check()
        ranges = pf.upload(np.array(target.ranges, dtype="<u8"))
        sink_bytes = max(args.blocks) * 128 * 8
        sink = pf.alloc(sink_bytes + 32)
        pf.fill(sink, 0xA5, sink_bytes + 32)
        pressure = r.alloc(pressure_bytes)
        # A nonconstant setup buffer avoids making the cache-pressure control
        # depend on compression of a memset pattern. Setup is never timed.
        rng = np.random.default_rng(81941)
        pressure_checksum = 0
        for offset in range(0, pressure_bytes, 8 * 1024**2):
            chunk = rng.integers(
                0, 256, min(8 * 1024**2, pressure_bytes - offset), dtype="u1"
            )
            pressure_checksum ^= int(np.bitwise_xor.reduce(chunk.view("<u4")[::8]))
            checked(
                sdk.lib.hggcMemcpy(
                    pressure + offset, chunk.ctypes.data, chunk.nbytes, 1
                ),
                "pressure H2D",
            )
        pressure_sink = r.alloc(288 * 256 * 4)
        sdk.synchronize(None)
        sdk.synchronize(pf.stream)
        sdk.synchronize(r.stream)
        # Admit both primitives before graph capture; never discover an invalid
        # device image inside the measured graph.
        for mode in ("hint", "load"):
            checked(
                probe.prefetch(
                    ranges,
                    len(target.ranges),
                    int(mode == "hint"),
                    args.blocks[0],
                    sink + 16,
                    pf.stream,
                ),
                "prefetch eager",
            )
            sdk.synchronize(pf.stream)
            check_receipt(sdk, sink + 16, args.blocks[0], target, mode)
        checked(
            probe.pressure(pressure, pressure_bytes, pressure_sink, r.stream),
            "pressure eager",
        )
        sdk.synchronize(r.stream)
        got_pressure = np.frombuffer(
            sdk.download(pressure_sink, 288 * 256 * 4), dtype="<u4"
        )
        if int(np.bitwise_xor.reduce(got_pressure)) != pressure_checksum:
            raise ValueError("cache-pressure load checksum differs")
        result.update(
            shared_bytes=current.query.shared_bytes,
            residency_ctas_per_cu=current.query.occupancy,
            weight_bytes=sum(size for _, size in target.ranges),
            current_hashes=current.hashes,
            next_hashes=target.hashes,
            weight_address_sets_disjoint=True,
            active_experts=target.active,
            target_shared_bytes=target.query.shared_bytes,
            target_residency_ctas_per_cu=target.query.occupancy,
        )
        config = configurations(args.blocks, include_primed=cross_projection)
        for key in config:
            print(
                f"KPACK_PREFETCH_GRAPH case={subject['case']} kind={key[0]} mode={key[1]} blocks={key[2]}",
                flush=True,
            )
            graphs[key] = Experiment(
                sdk,
                r,
                pf,
                probe,
                current,
                target,
                pressure,
                pressure_bytes,
                pressure_sink,
                ranges,
                sink + 16,
                *key,
            )
            graphs[key].sample()  # graph upload/first launch excluded
        for round_ in range(args.rounds):
            for key in config[:: 1 if round_ % 2 == 0 else -1]:
                kind, mode, blocks = key
                print(
                    f"KPACK_PREFETCH_PROGRESS case={subject['case']} round={round_+1}/{args.rounds} kind={kind} mode={mode} blocks={blocks}",
                    flush=True,
                )
                values = []
                for _ in range(args.samples):
                    target.poison()
                    if kind in PAIR_KINDS:
                        current.poison()
                    values.append(graphs[key].sample())
                error = target.check()
                if kind in PAIR_KINDS:
                    error = max(error, current.check())
                if mode != "none":
                    check_receipt(sdk, sink + 16, blocks, target, mode)
                if (
                    sdk.download(sink, 16) != b"\xa5" * 16
                    or sdk.download(sink + 16 + sink_bytes, 16) != b"\xa5" * 16
                ):
                    raise ValueError("prefetch receipt guard changed")
                result["cells"].append(
                    dict(
                        round=round_,
                        kind=kind,
                        mode=mode,
                        blocks=blocks,
                        error=error,
                        samples=values,
                    )
                )
        result["summary"] = summary(result["cells"])
        result["status"] = "PASS"
        for row in result["summary"]:
            print(
                "KPACK_PREFETCH_RESULT "
                + json.dumps(dict(case=subject["case"], **row)),
                flush=True,
            )
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")
        sdk.synchronize(r.stream)
        sdk.synchronize(pf.stream)
        for graph in graphs.values():
            graph.close()
        if current:
            current.close()
        if target:
            target.close()
        pf.close()
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--bundle", type=Path, default=ROOT / "prebuilt/ppu0010/kpack-prefetch-v1"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=[4, 16, 36])
    p.add_argument("--samples", type=int, default=11)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pressure-mib", type=int, default=0)
    p.add_argument(
        "--allow-unverified-sdk",
        action="store_true",
        help="allow different runtime-library hashes, record unverified SDK; no numerical check bypass",
    )
    args = p.parse_args()
    if (
        args.samples < 3
        or args.rounds < 2
        or len(args.blocks) != len(set(args.blocks))
        or any(b < 1 or b > 72 for b in args.blocks)
        or args.pressure_mib < 0
    ):
        p.error("invalid bounded sample/grid settings")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / (args.case + ".json")).exists():
        p.error("result already exists; use a fresh output directory")
    manifest, library = verify(args.bundle)
    runtime_sdk = admit_runtime_sdk(args, manifest)
    subject = select_subject(manifest, args.case)
    run(args, subject, manifest, library, runtime_sdk)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
