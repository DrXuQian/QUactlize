#!/usr/bin/env python3
"""Matched resident GEMV/FQ/SF endpoints and separate native ACU reports.

No device compilation, model routing change, or online tactic tuning occurs
in inference. This diagnostic scans only 24 recipes per SIMT reader. GEMM
choices come from the actual native selector, including their Split-K.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import Call as VecCall, Config, Sizes, arrangement, bind
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, Call, checked
from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command
from tools.run_kpack_decode_sweep import pair_bind, check_partials, upload_into
from tools.run_kpack_gemv_gate import Resources, compare, fixture
from tools.run_kpack_grouped_decode_probe import Replay, timing_summary
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_postops import resolve_acu
from tools.run_kpack_native_gate import check_prepass
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify as verify_native

CASES = {
    "q4-up": (12, 512, 2048, 256, 8, 1),
    "q5-down": (13, 2048, 512, 256, 8, 8),
    "q4-dense-matched": (12, 4096, 2048, 1, 1, 1),
    "q4-dense-wide": (12, 8192, 5120, 1, 1, 1),
    "q4-dense-long-k": (12, 5120, 25600, 1, 1, 1),
}
READERS = ("pair", "affine")
PROFILES = (*READERS, "fq", "sf", "prepass")
COUNTS = dict(rounds=4, samples=11, graph_calls=16, correctness_repeats=3)


def configs():
    return [(c, w, s) for c in (16, 32) for w in (2, 4, 8) for s in (1, 2, 4, 8)]


def verify_affine(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    if (
        m.get("schema") != "quactlize.kpack-execution-build.v1"
        or m.get("variant") != "fp32-affine"
        or m.get("comparison_adapters") is not True
        or m.get("library") != "libquactlize_ppu_execution.so"
        or sha(bundle / m["library"]) != m["sha256"]
    ):
        raise ValueError("affine PPU package differs or is not materialized by Git LFS")
    return m


def authority(args, sdk):
    affine = verify_affine(args.affine_bundle)
    verify_native(args.native_bundle)
    for name, digest in affine["runtime"].items():
        if sha(args.sdk / "lib" / name) != digest:
            raise ValueError("SDK runtime differs: " + name)
    paths = [
        Path(__file__),
        ROOT / "tools/kpack_execution_fixture.py",
        ROOT / "tools/kpack_warmup_fixture.py",
        ROOT / "tools/run_kpack_gemv_gate.py",
        ROOT / "tools/run_kpack_decode_sweep.py",
        ROOT / "tools/run_kpack_native_gate.py",
        ROOT / "tools/run_kpack_grouped_decode_probe.py",
        ROOT / "tools/profile_kpack_gpu_compact.py",
        ROOT / "reference/gguf_kpack.py",
        ROOT / "quactlize/runtime/native.py",
        ROOT / "quactlize/execution/native.py",
        ROOT / "quactlize/dispatch/native.py",
    ]
    return dict(
        native_manifest=sha(args.native_bundle / "manifest.json"),
        affine_manifest=sha(args.affine_bundle / "manifest.json"),
        device=device_identity(sdk),
        counts=COUNTS,
        source={str(p.relative_to(ROOT)): sha(p) for p in paths},
    )


class Prepared:
    def __init__(self, name, launch, check, selection, core=None):
        self.name, self.launch, self.check = name, launch, check
        self.selection, self.core = selection, core


class Comparison:
    def __init__(self, args, sdk, case):
        self.args, self.sdk, self.case = args, sdk, case
        self.q, self.n, self.k, self.e, self.m, self.channels = CASES[case]
        self.grouped = self.e > 1
        self.arr = arrangement(self.q)
        self.r = Resources(sdk)
        self.dispatch = Dispatch(args.native_bundle)
        self.graphs = []
        self.guard_allocations = []
        self.choices = {}
        manifest = verify_native(args.native_bundle)
        self.parents = {r["key"]: r["parent"] for r in manifest["modules"]}
        self.old = C.CDLL(
            str(args.native_bundle / "libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL
        )
        self.new = C.CDLL(
            str(args.affine_bundle / "libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL
        )
        self.functions = {
            "pair": pair_bind(self.old)["pair"],
            "affine": pair_bind(self.new)["pair"],
        }
        self.adapter = self.new.qkg_comparison_adapter_v1
        self.adapter.argtypes = [
            C.c_int,
            C.c_void_p,
            C.c_void_p,
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        self.adapter.restype = C.c_int
        specs = []
        for arm, route in (
            ("fq", 2 if self.grouped else 0),
            ("sf", 3 if self.grouped else 1),
        ):
            choice = self.dispatch.query(
                self.q, route, self.m, self.n, self.k, self.e, 1, self.arr.mapping_id
            )
            if choice is None:
                raise ValueError("current native policy misses " + case + "/" + arm)
            self.choices[arm] = choice
            parent = self.parents[choice.build_key.decode()]
            if choice.split > 1:
                specs.append((parent["tk"], choice.split))
        selected = np.arange(self.m) * 17 if self.grouped else [0]
        print(
            f"GEMV_FQ_SF_FIXTURE case={case} shape={self.m}x{self.n}x{self.k} E={self.e}",
            flush=True,
        )
        self.w = IndexedWeights(
            self.q,
            self.n,
            self.k,
            self.e,
            partial_specs=set(specs),
            partial_experts=selected,
            include_contiguous_partials=not self.grouped,
            progress=lambda d, t: print(
                f"GEMV_FQ_SF_FIXTURE experts={d}/{t}", flush=True
            ),
        )
        if not self.grouped:
            self.w.partial_sums = self.w.contiguous_partial_sums
            self.w.partial_schedule = "contiguous"
        mode = dict(
            mode=2 if self.grouped else 0,
            rows=self.m,
            channels=self.channels,
            topk=self.m,
        )
        self.data = fixture(self.w, mode)
        self.data["values"] = activation_values(np.arange(len(self.data["a"])))
        self.initial_experts = self.data["expert"].copy()
        self.planes = {
            p: self.r.upload(self.w.planes[p]) if self.w.planes[p].size else None
            for p in ("low", "high", "units")
        }
        self.a = self.r.upload(self.data["a"].astype("<f4"))
        self.ids = (
            self.r.upload(self.initial_experts.astype("i4")) if self.grouped else None
        )
        self.order_ptr = self.r.alloc(self.m * 4)
        self.bounds = self.r.alloc((self.e + 1) * 4) if self.grouped else None
        self.gathered = self.allocate(self.m * self.k * 2)
        self.set_router(self.initial_experts)
        size = self.e * (self.k // self.arr.group_size) * self.n * 2
        self.scale, self.zero = self.allocate(size), self.allocate(size)
        _, _, sf = bind(self.old)
        self.prepass = lambda: sf(
            self.q,
            self.n,
            self.k,
            self.e,
            self.planes["units"],
            self.w.planes["units"].nbytes,
            self.scale,
            self.zero,
            size,
            C.byref(self.arr),
            self.r.stream,
        )
        sdk.synchronize(None)
        self.r.fill(self.scale, 0x7E, size)
        self.r.fill(self.zero, 0x7E, size)
        self.first_prepass_us = self.r.samples(self.prepass, 1)[0]
        self.prepass_proof = check_prepass(sdk, self.w, self.scale, self.zero)
        self.prepass_bytes = size * 2 + self.w.planes["units"].nbytes

    def allocate(self, size, alignment=16):
        base = self.r.alloc(size + alignment + 16)
        self.r.fill(base, 0xA5, size + alignment + 16)
        ptr = base + alignment
        self.guard_allocations.append((base, ptr, size, alignment))
        return ptr

    def guards(self):
        for base, ptr, size, align in self.guard_allocations:
            if (
                self.sdk.download(base, align) != b"\xa5" * align
                or self.sdk.download(ptr + size, 16) != b"\xa5" * 16
            ):
                raise ValueError("output/workspace/metadata allocation guard changed")

    def set_router(self, experts):
        self.data["expert"] = np.asarray(experts, dtype="i4")
        self.order = np.argsort(self.data["expert"], kind="stable")
        rows = np.bincount(self.data["expert"], minlength=self.e).astype("i4")
        upload_into(self.sdk, self.order_ptr, self.order.astype("i4"))
        if self.grouped:
            upload_into(self.sdk, self.ids, self.data["expert"])
            upload_into(self.sdk, self.bounds, np.r_[0, rows.cumsum()].astype("i4"))
        v = self.data["values"]
        self.data["golden"] = np.stack(
            [v[self.data["arows"][r]] @ self.w.sums[e] for r, e in enumerate(experts)]
        )
        self.data["denom"] = np.stack(
            [
                abs(v[self.data["arows"][r]]) @ self.w.abs_sums[e]
                for r, e in enumerate(experts)
            ]
        )

    def gather(self):
        return self.adapter(
            0,
            self.a,
            self.gathered,
            self.order_ptr,
            self.m,
            self.k,
            self.channels,
            self.r.stream,
        )

    def prepare_gemv(self, reader, config):
        query, run = self.functions[reader]
        cfg = Config(*config)
        out = self.allocate(self.m * self.n * 4)
        c = VecCall(
            version=1,
            size=C.sizeof(VecCall),
            qtype=self.q,
            n=self.n,
            k=self.k,
            experts=self.e,
            rows=self.m,
            mode=2 if self.grouped else 0,
            input_type=1,
            channels=self.channels,
            topk=self.m,
            a_row_stride=self.k,
            a_token_stride=self.channels * self.k,
            ids_stride=self.m,
            out_row_stride=self.n,
            a=self.a,
            low=self.planes["low"],
            high=self.planes["high"],
            units=self.planes["units"],
            ids=self.ids,
            output=out,
            stream=self.r.stream.value,
        )
        sizes = Sizes()
        checked(
            query(C.byref(c), C.byref(cfg), C.byref(self.arr), C.byref(sizes)),
            "SIMT query",
        )
        c.workspace = self.allocate(sizes.workspace_bytes)
        c.workspace_bytes = sizes.workspace_bytes
        launch = lambda: run(C.byref(c), C.byref(cfg), C.byref(self.arr))

        def check():
            self.sdk.synchronize(self.r.stream)
            got = np.frombuffer(
                self.sdk.download(out, self.m * self.n * 4), dtype="<f4"
            ).reshape(self.m, self.n)
            err = compare(got, self.data)
            if cfg.split > 1:
                parts = np.frombuffer(
                    self.sdk.download(c.workspace, sizes.workspace_bytes), dtype="<f4"
                ).reshape(self.m, cfg.split, self.n)
                total = np.zeros_like(got)
                for s in range(cfg.split):
                    np.add(total, parts[:, s], out=total)
                if not np.array_equal(total.view("u4"), got.view("u4")):
                    raise ValueError(
                        "SIMT fixed-order reducer differs from its FP32 partials"
                    )
            self.guards()
            return err

        def poison():
            self.r.fill(out, 0xFF, self.m * self.n * 4)
            if sizes.workspace_bytes:
                self.r.fill(c.workspace, 0xFF, sizes.workspace_bytes)

        p = Prepared(
            reader,
            launch,
            check,
            dict(
                reader=reader,
                columns=cfg.columns,
                warps=cfg.warps,
                split=cfg.split,
                workspace_bytes=sizes.workspace_bytes,
                input="F32",
                output="F32",
            ),
        )
        p.poison = poison
        return p

    def prepare_gemm(self, arm):
        choice = self.choices[arm]
        halfout = self.allocate(self.m * self.n * 2)
        output = self.allocate(self.m * self.n * 4)
        workspace = self.allocate(
            choice.workspace_bytes, 128 if not self.grouped else 16
        )
        sf = arm == "sf"
        c = Call(
            version=1,
            size=C.sizeof(Call),
            m=self.m,
            n=self.n,
            k=self.k,
            experts=self.e,
            group_size=self.arr.group_size,
            device=choice.device,
            compute_units=choice.compute_units,
            mapping_id=self.arr.mapping_id,
            a=self.gathered,
            low=self.planes["low"],
            high=self.planes["high"],
            metadata=self.scale if sf else self.planes["units"],
            zero=self.zero if sf else None,
            output=halfout,
            offsets_device=self.bounds,
            workspace=workspace,
            workspace_bytes=choice.workspace_bytes,
            stream=self.r.stream.value,
        )
        core = self.dispatch.prepare(choice, c)

        def launch():
            rc = self.gather()
            if rc:
                return rc
            rc = core()
            if rc:
                return rc
            return self.adapter(
                1,
                halfout,
                output,
                self.order_ptr,
                self.m,
                self.n,
                self.channels,
                self.r.stream,
            )

        def check():
            self.sdk.synchronize(self.r.stream)
            got = np.frombuffer(
                self.sdk.download(output, self.m * self.n * 4), dtype="<f4"
            ).reshape(self.m, self.n)
            err = compare(got, self.data)
            h = np.frombuffer(
                self.sdk.download(halfout, self.m * self.n * 2), dtype="<f2"
            ).reshape(self.m, self.n)
            if not np.array_equal(
                h[np.argsort(self.order)].astype("f4").view("u4"), got.view("u4")
            ):
                raise ValueError("common F32 output differs from the FP16 GEMM result")
            if choice.split > 1:
                size = choice.split * self.m * self.n * 4
                parent = self.parents[choice.build_key.decode()]
                check_partials(
                    self.sdk.download(workspace + choice.workspace_bytes - size, size),
                    h,
                    self.w,
                    self.data,
                    parent["tk"],
                    choice.split,
                    self.order,
                )
            self.guards()
            return err

        def poison():
            self.r.fill(output, 0xFF, self.m * self.n * 4)
            self.r.fill(halfout, 0xFF, self.m * self.n * 2)
            if choice.split > 1:
                size = choice.split * self.m * self.n * 4
                self.r.fill(workspace + choice.workspace_bytes - size, 0xFF, size)

        p = Prepared(arm, launch, check, receipt(choice), core)
        p.poison = poison
        return p

    def graph(self, launch, repeats):
        g = Replay(self.sdk, self.r.stream, launch, repeats)
        self.graphs.append(g)
        return g

    def validate(self, p, launch):
        p.poison()
        checked(launch(), p.name + " correctness")
        return p.check()

    def measure(self, launch, samples, repeats):
        g = self.graph(launch, repeats)
        try:
            for _ in range(3):
                checked(g(), "warmup")
            return self.r.samples(g, samples)
        finally:
            g.close()
            self.graphs.remove(g)

    def close(self):
        self.sdk.synchronize(self.r.stream)
        for graph in self.graphs:
            graph.close()
        self.dispatch.close()
        self.r.close()


def run_case(args, sdk):
    result = dict(
        status="FAIL",
        case=args.case,
        authority=authority(args, sdk),
        screen=[],
        winners={},
    )
    bench = None
    try:
        bench = Comparison(args, sdk, args.case)
        if (
            np.max(abs(bench.data["golden"]) / np.maximum(bench.data["denom"], 1e-30))
            <= 0.005
        ):
            raise ValueError("zero-output negative is not discriminating")
        weight_bytes = sum(
            bench.w.planes[k].nbytes // bench.e * bench.m
            for k in ("low", "high", "units")
        )
        result.update(
            shape=[bench.m, bench.n, bench.k],
            experts=bench.e,
            channels=bench.channels,
            weight_bytes=weight_bytes,
            plane_sha256={
                p: hashlib.sha256(bench.w.planes[p].tobytes()).hexdigest()
                for p in ("low", "high", "units")
            },
            first_prepass_us=bench.first_prepass_us,
            prepass_proof=bench.prepass_proof,
            scope="GPU_READY_ROUTING_COMMON_F32_INPUT_OUTPUT_NO_IDS_GENERATION_NO_HOST_PREPARATION",
            accuracy="OFFICIAL_GGUF_CONDITIONED_DOT_LT_0.005_NOT_BIT_IDENTICAL_ALGORITHMS",
        )
        prepared = {}
        if args.profile:
            previous = json.loads(args.measured.read_text())
            if not case_complete(previous, args.case, result["authority"]):
                raise ValueError(
                    "ACU requires this case's complete, same-authority timing result"
                )
            for reader in READERS:
                cfg = previous["winners"][reader]["selection"]
                prepared[reader] = bench.prepare_gemv(
                    reader, tuple(cfg[x] for x in ("columns", "warps", "split"))
                )
        else:
            for reader in READERS:
                for index, config in enumerate(configs()):
                    p = bench.prepare_gemv(reader, config)
                    error = bench.validate(p, p.launch)
                    times = bench.measure(p.launch, 3, 8)
                    row = dict(
                        reader=reader,
                        config=list(config),
                        error=error,
                        samples_us=[x / 8 for x in times],
                        median_us=statistics.median(times) / 8,
                    )
                    result["screen"].append(row)
                    if index % 6 == 5:
                        print(
                            f"GEMV_FQ_SF_SCREEN case={args.case} reader={reader} done={index+1}/24",
                            flush=True,
                        )
                winner = min(
                    (r for r in result["screen"] if r["reader"] == reader),
                    key=lambda r: r["median_us"],
                )
                prepared[reader] = bench.prepare_gemv(reader, tuple(winner["config"]))
        prepared.update({arm: bench.prepare_gemm(arm) for arm in ("fq", "sf")})

        def with_prepass():
            rc = bench.prepass()
            return rc if rc else prepared["sf"].launch()

        combined = Prepared(
            "sf_with_prepass",
            with_prepass,
            prepared["sf"].check,
            prepared["sf"].selection,
        )
        combined.poison = prepared["sf"].poison
        prepared["sf_with_prepass"] = combined
        errors = {name: [] for name in prepared}
        graphs = {
            name: bench.graph(p.launch, 1 if args.profile else COUNTS["graph_calls"])
            for name, p in prepared.items()
        }
        for name, p in prepared.items():
            for repeat in range(COUNTS["correctness_repeats"]):
                experts = (
                    np.roll(bench.initial_experts, repeat)
                    if bench.grouped
                    else bench.initial_experts
                )
                bench.set_router(experts)
                errors[name].append(
                    bench.validate(p, p.launch if repeat == 0 else graphs[name])
                )
        bench.set_router(bench.initial_experts)
        if args.profile:
            fn = (
                bench.prepass
                if args.profile == "prepass"
                else prepared[args.profile].launch
            )
            for _ in range(3):
                checked(fn(), "profile warmup")
            bench.sdk.synchronize(bench.r.stream)
            graph = bench.graph(fn, 1)
            with AcuRange(sdk):
                checked(graph(), "ACU selected graph")
                sdk.synchronize(bench.r.stream)
            if args.profile == "prepass":
                check_prepass(sdk, bench.w, bench.scale, bench.zero)
            else:
                prepared[args.profile].check()
            result.update(
                status="PASS",
                profile=args.profile,
                selection=(
                    None
                    if args.profile == "prepass"
                    else prepared[args.profile].selection
                ),
            )
        else:
            samples = {name: [] for name in prepared}
            for round_ in range(COUNTS["rounds"]):
                names = list(prepared)[:: 1 if round_ % 2 == 0 else -1]
                for name in names:
                    p = prepared[name]
                    bench.validate(p, p.launch)
                    for _ in range(3):
                        checked(graphs[name](), "confirmation warmup")
                    samples[name].append(
                        bench.r.samples(graphs[name], COUNTS["samples"])
                    )
                    p.check()
                print(
                    f"GEMV_FQ_SF_CONFIRM case={args.case} round={round_+1}/{COUNTS['rounds']}",
                    flush=True,
                )
            for name, p in prepared.items():
                sfbytes = sum(
                    bench.w.planes[k].nbytes // bench.e * bench.m
                    for k in ("low", "high")
                )
                sfbytes += bench.m * bench.n * bench.k // bench.arr.group_size * 4
                active_bytes = sfbytes if name.startswith("sf") else weight_bytes
                row = dict(
                    selection=p.selection,
                    error=max(errors[name]),
                    status="PASS",
                    correctness_checks=len(errors[name]),
                    active_weight_bytes=active_bytes,
                    **timing_summary(
                        samples[name], COUNTS["graph_calls"], active_bytes
                    ),
                )
                if p.core:
                    checked(bench.gather(), "resident core A preparation")
                    core_times = bench.measure(
                        p.core, COUNTS["samples"], COUNTS["graph_calls"]
                    )
                    row["core_median_us"] = (
                        statistics.median(core_times) / COUNTS["graph_calls"]
                    )
                    row["core_graph_samples_us"] = core_times
                    row["core_scope"] = (
                        "FP16_INPUT_OUTPUT_METADATA_DIRECTORY_PRODUCER_REDUCER_NO_ADAPTERS"
                    )
                    p.check()
                result["winners"][name] = row
            prepass_times = bench.measure(
                bench.prepass, COUNTS["samples"], COUNTS["graph_calls"]
            )
            result["prepass"] = dict(
                median_us=statistics.median(prepass_times) / COUNTS["graph_calls"],
                graph_samples_us=prepass_times,
                all_experts=bench.e,
                logical_read_write_bytes=bench.prepass_bytes,
            )
            check_prepass(sdk, bench.w, bench.scale, bench.zero)
            result["status"] = "PASS"
    except BaseException as error:
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        try:
            if bench:
                bench.close()
        except Exception as error:
            result.update(status="FAIL", cleanup_error=str(error))
            traceback.print_exc()
        finally:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result["status"] == "PASS"


def case_complete(result, case, expected_authority):
    if (
        result.get("status") != "PASS"
        or result.get("case") != case
        or result.get("authority") != expected_authority
        or len(result.get("screen", [])) != 48
        or set(result.get("winners", {}))
        != {"pair", "affine", "fq", "sf", "sf_with_prepass"}
    ):
        return False

    def positive(x):
        return isinstance(x, (int, float)) and math.isfinite(x) and x > 0

    if (
        not positive(result.get("first_prepass_us"))
        or result.get("prepass_proof", {}).get("status") != "PASS"
        or not positive(result.get("prepass", {}).get("median_us"))
    ):
        return False
    keys = {(r.get("reader"), tuple(r.get("config", []))) for r in result["screen"]}
    if keys != {(reader, cfg) for reader in READERS for cfg in configs()}:
        return False
    if any(
        not 0 <= r.get("error", 1) < 0.005
        or len(r.get("samples_us", [])) != 3
        or not all(positive(x) for x in r["samples_us"])
        for r in result["screen"]
    ):
        return False
    for row in result["winners"].values():
        samples = row.get("graph_elapsed_samples_us", [])
        if (
            row.get("status") != "PASS"
            or not 0 <= row.get("error", 1) < 0.005
            or row.get("correctness_checks") != COUNTS["correctness_repeats"]
            or row.get("calls_per_graph") != COUNTS["graph_calls"]
            or len(samples) != COUNTS["rounds"]
            or any(len(s) != COUNTS["samples"] for s in samples)
            or not all(
                isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                for s in samples
                for x in s
            )
        ):
            return False
    return True


def child_command(args, case, output):
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--sdk",
        str(args.sdk),
        "--native-bundle",
        str(args.native_bundle),
        "--affine-bundle",
        str(args.affine_bundle),
        "--case",
        case,
        "--output",
        str(output),
    ]


def execute(command, log, label):
    print(f"GEMV_FQ_SF_START {label} log={log}", flush=True)
    started = time.monotonic()
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                return process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(
                    f"GEMV_FQ_SF_ALIVE {label} elapsed_seconds={time.monotonic()-started:.0f}",
                    flush=True,
                )


def collect(args):
    sdk = SDK(args.sdk)
    expected = authority(args, sdk)
    acu = None if args.skip_acu else resolve_acu(args.sdk, args.acu)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "acu").mkdir(exist_ok=True)
    summary = []
    for case in args.only_cases or list(CASES):
        target = args.output / f"{case}.json"
        old = json.loads(target.read_text()) if target.exists() else {}
        valid = case_complete(old, case, expected)
        if args.resume and old and old.get("authority") != expected:
            raise ValueError("resume authority differs; use a fresh results directory")
        if not (args.resume and valid):
            if target.exists():
                target.rename(target.with_suffix(f".failed.{time.time_ns()}.json"))
            log = args.output / f"{case}.{time.time_ns()}.log"
            rc = execute(
                child_command(args, case, target), log, f"case={case} phase=timing"
            )
            old = json.loads(target.read_text()) if target.exists() else {}
            valid = rc == 0 and case_complete(old, case, expected)
        row = dict(
            case=case,
            status="PASS" if valid else "FAIL",
            timing_status="PASS" if valid else "FAIL",
            reports=[],
        )
        if valid:
            for arm, w in old["winners"].items():
                print(
                    f"GEMV_FQ_SF_RESULT case={case} arm={arm} median_us={w['median_us']:.6f} selection={json.dumps(w['selection'],sort_keys=True)}",
                    flush=True,
                )
        if valid and acu:
            for arm in PROFILES:
                prefix = args.output / "acu" / f"{case}-{arm}"
                capture = Path(str(prefix) + ".json")
                stamp = Path(str(prefix) + ".capture.json")
                reports = [
                    p
                    for p in (prefix, Path(str(prefix) + ".acurep"))
                    if p.is_file() and p.stat().st_size
                ]
                prior = json.loads(capture.read_text()) if capture.exists() else {}
                mark = json.loads(stamp.read_text()) if stamp.exists() else {}
                if (
                    args.resume
                    and prior.get("authority") == expected
                    and prior.get("status") == "PASS"
                    and prior.get("profile") == arm
                    and len(reports) == 1
                    and mark.get("report_sha256") == sha(reports[0])
                    and mark.get("receipt_sha256") == sha(capture)
                ):
                    row["reports"].append(
                        dict(
                            arm=arm,
                            status="REUSED",
                            report=str(reports[0]),
                            sha256=sha(reports[0]),
                        )
                    )
                    continue
                if capture.exists():
                    capture.rename(Path(str(capture) + f".failed.{time.time_ns()}"))
                for report in reports:
                    report.rename(Path(str(report) + f".failed.{time.time_ns()}"))
                cmd = acu_launch_command(
                    acu,
                    prefix,
                    child_command(args, case, capture)
                    + ["--profile", arm, "--measured", str(target)],
                )
                log = Path(str(prefix) + f".{time.time_ns()}.log")
                rc = execute(cmd, log, f"case={case} phase=acu arm={arm}")
                captures = [
                    p
                    for p in (prefix, Path(str(prefix) + ".acurep"))
                    if p.is_file() and p.stat().st_size
                ]
                proof = json.loads(capture.read_text()) if capture.exists() else {}
                good = (
                    rc == 0
                    and len(captures) == 1
                    and proof.get("status") == "PASS"
                    and proof.get("authority") == expected
                    and proof.get("profile") == arm
                )
                good = (
                    good
                    and "no kernels were profiled"
                    not in log.read_text(errors="replace").lower()
                )
                row["reports"].append(
                    dict(
                        arm=arm,
                        status="PASS" if good else "FAIL",
                        report=str(captures[0]) if captures else None,
                        sha256=sha(captures[0]) if captures else None,
                    )
                )
                if not good:
                    row["status"] = "FAIL"
                else:
                    stamp.write_text(
                        json.dumps(
                            dict(
                                report_sha256=sha(captures[0]),
                                receipt_sha256=sha(capture),
                            )
                        )
                        + "\n"
                    )
        summary.append(row)
        (args.output / "summary.json").write_text(
            json.dumps(dict(authority=expected, cases=summary), indent=2) + "\n"
        )
    lines = [
        "case\tarm\tmedian_us\tcore_median_us\tfirst_prepass_us\tsplit\tstatus\tacu_status"
    ]
    for row in summary:
        path = args.output / f"{row['case']}.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        if data.get("status") != "PASS":
            continue
        for arm, w in data["winners"].items():
            lines.append(
                "\t".join(
                    map(
                        str,
                        [
                            row["case"],
                            arm,
                            w["median_us"],
                            w.get("core_median_us", "NA"),
                            data["first_prepass_us"] if arm.startswith("sf") else "NA",
                            w["selection"]["split"],
                            data["status"],
                            (
                                "SKIPPED"
                                if not row["reports"]
                                else (
                                    "FAIL"
                                    if any(
                                        r["status"] == "FAIL" for r in row["reports"]
                                    )
                                    else "PASS"
                                )
                            ),
                        ],
                    )
                )
            )
    (args.output / "summary.tsv").write_text("\n".join(lines) + "\n")
    success = all(row["status"] == "PASS" for row in summary)
    print(
        f"GEMV_FQ_SF_COMPLETE cases={len(summary)} status={'PASS' if success else 'FAIL'} results={args.output}",
        flush=True,
    )
    return success


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument(
        "--native-bundle", type=Path, default=ROOT / "prebuilt/ppu0010/kpack-native-v1"
    )
    p.add_argument(
        "--affine-bundle",
        type=Path,
        default=ROOT / "prebuilt/ppu0010/kpack-gemv-affine-v1",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--only-cases", nargs="+", choices=CASES)
    p.add_argument("--profile", choices=PROFILES)
    p.add_argument("--measured", type=Path)
    p.add_argument("--acu", type=Path)
    p.add_argument("--skip-acu", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    for field in ("sdk", "native_bundle", "affine_bundle", "output"):
        setattr(a, field, getattr(a, field).resolve())
    if a.profile and (a.collect or not a.measured):
        p.error("profile requires measured case result, not collect")
    if not a.collect and not a.case:
        p.error("choose --collect or --case")
    if not a.collect and a.output.exists():
        p.error("case output already exists")
    if a.only_cases and len(a.only_cases) != len(set(a.only_cases)):
        p.error("duplicate case")
    if a.collect:
        ok = collect(a)
    else:
        sdk = SDK(a.sdk)
        graph_bind(sdk)
        ok = run_case(a, sdk)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
