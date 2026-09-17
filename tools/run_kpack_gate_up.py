#!/usr/bin/env python3
"""Bounded paired-N4 device gate; each format/backend runs in a fresh process."""

import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.bf16_compute.fixture import (
    raw_weight,
    planes,
    bf16_bits,
    bf16_float,
    round_compute,
)
from dev.gemv_simt.native import Runtime, Graph, checked
from quactlize.fusion.native import Library, FusionCall, Config, Call, Sizes
from quactlize.runtime.compiler import sha, LIBRARIES

FORMATS = (8, 10, 11, 12, 13, 14)
HELPERS = (
    "tools/run_kpack_gate_up.py",
    "quactlize/fusion/native.py",
    "dev/bf16_compute/fixture.py",
    "dev/gemv_simt/native.py",
    "tools/kpack_warmup_fixture.py",
    "reference/gguf_kpack.py",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(bundle, sdk):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema"] != "quactlize.gate-up-paired-n4.v1":
        raise ValueError("wrong bundle schema")
    if manifest["library"] != "libquactlize_ppu_gate_up.so" or manifest[
        "formats"
    ] != list(FORMATS):
        raise ValueError("unexpected library or format inventory")
    library = (bundle / manifest["library"]).resolve(strict=True)
    if library.parent != bundle.resolve():
        raise ValueError("library outside bundle")
    if sha(library) != manifest["sha256"]:
        raise ValueError("library hash differs")
    runtime = {f"lib{x}.so": sha(sdk / "lib" / f"lib{x}.so") for x in LIBRARIES}
    if runtime != manifest["runtime"]:
        raise ValueError("runtime libraries differ from build receipt")
    for name, want in manifest["source_hashes"].items():
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha(path) != want:
            raise ValueError("compiled source differs: " + name)
    return library, manifest


def identity(manifest):
    return dict(
        library_sha256=manifest["sha256"],
        runtime=manifest["runtime"],
        helpers={name: sha(ROOT / name) for name in HELPERS},
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def device_identity(rt):
    count, current = C.c_int(), C.c_int()
    for name in ("hggcGetDeviceCount", "hggcGetDevice"):
        fn = getattr(rt.lib, name)
        fn.argtypes = [C.POINTER(C.c_int)]
        fn.restype = C.c_int
    fn = rt.lib.hggcDeviceGetPCIBusId
    fn.argtypes = [C.c_char_p, C.c_int, C.c_int]
    fn.restype = C.c_int
    checked(rt.lib.hggcGetDeviceCount(C.byref(count)), "visible device count")
    checked(rt.lib.hggcGetDevice(C.byref(current)), "current device")
    if count.value != 1 or current.value != 0:
        raise ValueError("select exactly one visible PPU")
    pci = C.create_string_buffer(64)
    checked(fn(pci, len(pci), current.value), "physical device PCI identity")
    return dict(
        ordinal=current.value, pci=pci.value.decode(), visible_devices=count.value
    )


def cases(backend):
    result = [(0, m, 1, "dense") for m in range(1, 9)]
    result += [
        (1, m, 1, profile) for m in (9, 17, 33) for profile in ("spread", "single")
    ]
    result += [(2, m, channels, "indexed") for m in range(1, 9) for channels in (1, 8)]
    if backend == "tc":
        result += [(0, m, 1, "dense") for m in (65, 129)]
        result += [
            (1, m, 1, profile) for m in (65, 129) for profile in ("spread", "single")
        ]
    return result


def configurations(backend, split):
    if backend == "simt":
        return [Config(0, split, 0, w) for w in (4, 8)]
    return [Config(1, split, tm, 0) for tm in (8, 16)]


def arithmetic():
    for compute in (0, 1):
        for storage, output, rounding in (
            (1, 1, 0),
            (1, 1, 1),
            (2 if compute else 0, 2 if compute else 0, 1),
        ):
            yield compute, storage, output, rounding


RECORD_KEYS = (
    "mode",
    "m",
    "channels",
    "profile",
    "compute",
    "storage",
    "output",
    "round_projection",
    "split",
    "tile_m",
    "warps",
)


def complete_part(part, q, backend, receipt):
    if (
        part.get("status") != "PASS"
        or part.get("identity") != receipt
        or part.get("q") != q
        or part.get("backend") != backend
    ):
        return False
    expected = {
        (*case, *types, split, config.tile_m, config.warps)
        for case in cases(backend)
        for types in arithmetic()
        for split in (1, 2, 4, 8)
        for config in configurations(backend, split)
    }
    records = part.get("records", [])
    try:
        actual = [tuple(record[key] for key in RECORD_KEYS) for record in records]
        return (
            len(actual) == len(expected)
            and set(actual) == expected
            and all(
                math.isfinite(record["error"]) and 0 <= record["error"] < 0.005
                for record in records
            )
            and part.get("replays", 0) >= 80
            and part.get("negative_controls", 0) >= 32
        )
    except (KeyError, TypeError):
        return False


class Weights:
    def __init__(self, rt, lib, q, n, k, experts):
        self.rt, self.q, self.n, self.k, self.experts = rt, q, n, k, experts
        self.layout = lib.arrangement(q)
        sources, original = [], []
        for side in (0, 1):
            raw, golden = zip(
                *(raw_weight(q, n, k, 17 + side * 419 + e * 73) for e in range(experts))
            )
            sources.append(np.stack(raw))
            original.append(np.stack(golden))
        self.gate, self.up = original
        # Independent physical rows, then the existing byte-checked CPU packer.
        physical = np.stack(
            [x.reshape(experts, n // 4, 4, *x.shape[2:]) for x in sources], axis=2
        )
        physical = physical.reshape(experts, 2 * n, *sources[0].shape[2:])
        packed = [planes(raw, q) for raw in physical]
        self.buffers = {}
        self.guards = []
        for name in ("low", "high", "units"):
            expected = np.stack([p[name] for p in packed]).view("u1").reshape(-1)
            if not expected.size:
                self.buffers[name] = None
                continue
            allocation = rt.allocate(expected.size + 32)
            rt.fill(allocation, expected.size + 32)
            self.buffers[name] = allocation + 16
            self.guards.append((allocation, expected))
        raw_ptrs = [rt.upload(x) for x in sources]
        checked(
            lib.pack(
                *raw_ptrs,
                *[self.buffers[x] for x in ("low", "high", "units")],
                n,
                k,
                experts,
                q,
                C.byref(self.layout),
                rt.stream,
            ),
            "paired device pack",
        )
        rt.sync()
        for allocation, expected in self.guards:
            got = rt.download(allocation, expected.size + 32)
            if (
                not np.all(got[:16] == 0xA5)
                or not np.all(got[-16:] == 0xA5)
                or not np.array_equal(got[16:-16], expected)
            ):
                raise ValueError(
                    "paired pack differs from independent physical-row oracle"
                )


class Bench:
    def __init__(
        self,
        rt,
        lib,
        w,
        mode,
        m,
        compute,
        storage,
        output_type,
        rounding,
        channels=1,
        profile="spread",
    ):
        self.rt, self.lib, self.w = rt, lib, w
        self.start = len(rt.allocations)
        self.compute, self.storage, self.output_type = compute, storage, output_type
        self.mode, self.m, self.rounding = mode, m, rounding
        self.channels, self.profile = channels, profile
        self.rows = m * 8 if mode == 2 else m
        self.a_count = m * channels
        self.a = rt.allocate(self.a_count * (w.k + 8) * (4 if storage == 1 else 2))
        self.ids_array = (
            np.full((m, 11), -1, dtype="<i4") if mode == 2 else np.empty(0, dtype="<i4")
        )
        self.ids = rt.upload(self.ids_array)
        if mode == 1:
            self.owners = (
                np.arange(m) * w.experts // m
                if profile == "spread"
                else np.full(m, w.experts - 1)
            )
            counts = np.bincount(self.owners, minlength=w.experts)
            self.offset_array = np.r_[0, counts.cumsum()].astype("<i4")
        else:
            self.offset_array = np.empty(0, dtype="<i4")
        self.offsets = rt.upload(self.offset_array)
        width = 4 if output_type == 1 else 2
        self.output_bytes = self.rows * (w.n + 8) * width + 32
        self.output = rt.allocate(self.output_bytes)
        self.work_bytes = self.rows * 8 * (2 * w.n) * 4
        self.work = rt.allocate(self.work_bytes + 32)
        c = Call(
            version=1,
            size=C.sizeof(Call),
            qtype=w.q,
            n=w.n,
            k=w.k,
            experts=w.experts,
            rows=self.rows,
            mode=mode,
            input_type=storage,
            channels=channels,
            topk=8 if mode == 2 else 1,
            a_row_stride=w.k + 8,
            a_token_stride=(w.k + 8) * channels,
            ids_stride=11,
            out_row_stride=w.n + 8,
            a=self.a,
            ids=self.ids,
            offsets=self.offsets,
            output=self.output + 16,
            workspace=self.work + 16,
            workspace_bytes=self.work_bytes,
            stream=rt.stream.value,
            **w.buffers,
        )
        self.call = FusionCall(c, compute, output_type, rounding)
        self.update(0)

    def update(self, repeat, large=False):
        w = self.w
        rng = np.random.default_rng(132 + repeat)
        self.host_a = rng.integers(-7, 8, (self.a_count, w.k + 8)).astype("<f4") / 32
        if large:
            if not self.compute:
                raise ValueError("large activation requires BF16 compute")
            self.host_a[:, 0] = np.float32(243383.484)
        if self.storage == 0:
            stored = self.host_a.astype("<f2")
        elif self.storage == 2:
            stored = bf16_bits(self.host_a)
        else:
            stored = self.host_a
        self.rt.copy(self.a, stored)
        if self.mode == 2:
            self.ids_array[:, :8] = (
                np.arange(8)[None, :] * 3 + np.arange(self.m)[:, None] * 5 + repeat
            ) % w.experts
            self.rt.copy(self.ids, self.ids_array)
            self.owners = self.ids_array[:, :8].reshape(-1)
            a_rows = (
                np.arange(self.rows) // 8 * self.channels
                + np.arange(self.rows) % 8 % self.channels
            )
        else:
            if self.mode == 0:
                self.owners = np.zeros(self.rows, dtype=int)
            a_rows = np.arange(self.rows)
        activation = round_compute(
            self.host_a[:, : w.k], "bf16" if self.compute else "f16"
        )
        self.dot = []
        for weight in (w.gate, w.up):
            self.dot.append(
                np.stack(
                    [
                        activation[a_rows[r]].astype("f8") @ weight[e].astype("f8").T
                        for r, e in enumerate(self.owners)
                    ]
                ).astype("<f4")
            )
        g, u = self.dot
        if self.rounding:
            g, u = [round_compute(x, "bf16" if self.compute else "f16") for x in (g, u)]
        with np.errstate(over="ignore"):
            self.gold = (g / (np.float32(1) + np.exp(-g))) * u
        if self.output_type == 0:
            self.gold = self.gold.astype("<f2").astype("<f4")
        if self.output_type == 2:
            self.gold = bf16_float(bf16_bits(self.gold))
        if not np.isfinite(self.gold).all() or not np.any(self.gold):
            raise ValueError("degenerate oracle")

    def poison(self):
        self.rt.fill(self.output, self.output_bytes)
        self.rt.fill(self.work, self.work_bytes + 32)

    def check(self, split):
        raw = self.rt.download(self.output, self.output_bytes)
        if not np.all(raw[:16] == 0xA5) or not np.all(raw[-16:] == 0xA5):
            raise ValueError("output guard overwritten")
        body = (
            raw[16:-16]
            .view("<u4" if self.output_type == 1 else "<u2")
            .reshape(self.rows, self.w.n + 8)
        )
        if not np.all(
            body[:, self.w.n :] == (0xA5A5A5A5 if self.output_type == 1 else 0xA5A5)
        ):
            raise ValueError("row stride guard overwritten")
        values = body[:, : self.w.n].copy()
        if self.output_type == 1:
            got = values.view("<f4")
        elif self.output_type == 0:
            got = values.view("<f2").astype("<f4")
        else:
            got = bf16_float(values)
        work = self.rt.download(self.work, self.work_bytes + 32)
        if not np.all(work[:16] == 0xA5) or not np.all(work[-16:] == 0xA5):
            raise ValueError("partial guard overwritten")
        used = 0 if split == 1 else self.rows * split * 2 * self.w.n * 4
        if not np.all(work[16 + used : -16] == 0xA5):
            raise ValueError("unused workspace overwritten")
        if not np.isfinite(got).all():
            raise ValueError("nonfinite fused output")
        error = float(
            np.max(np.abs(got.astype("f8") - self.gold))
            / max(1e-20, float(np.max(np.abs(self.gold))))
        )
        if error >= 0.005:
            raise ValueError(f"independent GGUF + SwiGLU oracle error={error}")
        return error

    def invoke(self, config):
        return lambda: self.lib.run(
            C.byref(self.call), C.byref(config), C.byref(self.w.layout)
        )

    def close(self):
        self.rt.release_after(self.start)


def replay_checks(bench, backend):
    if bench.m not in (1, 8, 17) or bench.storage != 1 or bench.rounding != 1:
        return 0, 0
    rt = bench.rt
    replays = 0
    negatives = 0
    for split in (1, 8):
        config = configurations(backend, split)[0]
        graph = Graph(rt, [bench.invoke(config)])
        try:
            for repeat in (1, 2):
                bench.update(repeat)
                bench.poison()
                checked(rt.GraphLaunch(graph.instance, rt.stream), "paired replay")
                rt.sync()
                bench.check(split)
                replays += 1
            if bench.compute:
                bench.update(3, large=True)
                bench.poison()
                checked(rt.GraphLaunch(graph.instance, rt.stream), "large BF16 replay")
                rt.sync()
                bench.check(split)
                replays += 1
            rt.fill(bench.a, bench.a_count * (bench.w.k + 8) * 4, 0)
            checked(rt.GraphLaunch(graph.instance, rt.stream), "zero-A negative")
            rt.sync()
            try:
                bench.check(split)
            except ValueError as e:
                if not str(e).startswith("independent GGUF"):
                    raise
            else:
                raise ValueError("zero-A negative was not detected")
            negatives += 1
        finally:
            graph.close()
        # A negative must not contaminate the next graph's first launch.
        bench.update(0)
    return replays, negatives


def child(args):
    path, manifest = verify(args.bundle, args.sdk)
    rt = Runtime(args.sdk, "ppu")
    records = []
    replays = 0
    negative_controls = 0
    device = None
    receipt = identity(manifest)
    try:
        device = device_identity(rt)
        receipt["device"] = device
        lib = Library(path)
        for experts in (1, 8):
            mark = len(rt.allocations)
            w = Weights(rt, lib, args.q, 256, 2048, experts)
            for mode, m, channels, profile in cases(args.backend):
                if (mode == 0) != (experts == 1):
                    continue
                for compute, storage, out_type, rounding in arithmetic():
                    bench = Bench(
                        rt,
                        lib,
                        w,
                        mode,
                        m,
                        compute,
                        storage,
                        out_type,
                        rounding,
                        channels,
                        profile,
                    )
                    try:
                        for split in (1, 2, 4, 8):
                            for config in configurations(args.backend, split):
                                size = Sizes()
                                checked(
                                    lib.query(
                                        C.byref(bench.call),
                                        C.byref(config),
                                        C.byref(w.layout),
                                        C.byref(size),
                                    ),
                                    "paired query",
                                )
                                bench.poison()
                                checked(bench.invoke(config)(), "fused gate/up")
                                rt.sync()
                                error = bench.check(split)
                                records.append(
                                    dict(
                                        mode=mode,
                                        m=m,
                                        channels=channels,
                                        profile=profile,
                                        compute=compute,
                                        storage=storage,
                                        output=out_type,
                                        round_projection=rounding,
                                        split=split,
                                        tile_m=config.tile_m,
                                        warps=config.warps,
                                        error=error,
                                    )
                                )
                        new_replays, new_negatives = replay_checks(bench, args.backend)
                        replays += new_replays
                        negative_controls += new_negatives
                    finally:
                        bench.close()
                print(
                    f"GATE_UP_PROGRESS q={args.q} backend={args.backend} mode={mode} m={m} channels={channels} profile={profile} cells={len(records)}",
                    flush=True,
                )
            rt.release_after(mark)
        expected = len(cases(args.backend)) * 48
        if len(records) != expected:
            raise ValueError(f"incomplete cells: {len(records)}/{expected}")
        save(
            args.output,
            dict(
                status="PASS",
                q=args.q,
                backend=args.backend,
                records=records,
                identity=receipt,
                device=device,
                expected=expected,
                replays=replays,
                negative_controls=negative_controls,
                timing="NOT_MEASURED",
            ),
        )
        return 0
    except Exception as e:
        traceback.print_exc()
        save(
            args.output,
            dict(
                status="FAIL",
                q=args.q,
                backend=args.backend,
                error=str(e),
                records=records,
                identity=receipt,
                device=device,
            ),
        )
        return 1
    finally:
        rt.close()


def collect(args):
    _, manifest = verify(args.bundle, args.sdk)
    receipt = identity(manifest)
    probe = Runtime(args.sdk, "ppu")
    try:
        receipt["device"] = device_identity(probe)
    finally:
        probe.close()
    if args.resume:
        if json.loads((args.output / "identity.json").read_text()) != receipt:
            raise ValueError("resume identity differs; use a new output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "identity.json", receipt)
    rows = []
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["LD_LIBRARY_PATH"] = (
        str(args.sdk / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    for q in FORMATS:
        for backend in ("simt", "tc"):
            out = args.output / f"q{q}-{backend}.json"
            if args.resume and out.is_file():
                previous = json.loads(out.read_text())
                if complete_part(previous, q, backend, receipt):
                    rows.append(
                        dict(q=q, backend=backend, rc=0, result=out.name, reused=True)
                    )
                    print(
                        f"GATE_UP_PART q={q} backend={backend} status=REUSED_PASS",
                        flush=True,
                    )
                    continue
            # Preserve failed/partial evidence before retrying the part.
            if out.exists() or out.with_suffix(".log").exists():
                import time

                suffix = ".previous." + str(time.time_ns())
                for previous in (out, out.with_suffix(".log")):
                    if previous.exists():
                        previous.rename(previous.with_name(previous.name + suffix))
            print(
                f'GATE_UP_PART q={q} backend={backend} status=START expected={len(cases(backend))*48} log={out.with_suffix(".log")}',
                flush=True,
            )
            command = [
                sys.executable,
                __file__,
                "--bundle",
                str(args.bundle),
                "--sdk",
                str(args.sdk),
                "--output",
                str(out),
                "--q",
                str(q),
                "--backend",
                backend,
            ]
            with out.with_suffix(".log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("GATE_UP_PROGRESS"):
                        print(line.rstrip(), flush=True)
                rc = process.wait()
            if rc == 0 and (
                not out.is_file()
                or not complete_part(json.loads(out.read_text()), q, backend, receipt)
            ):
                rc = 1
                print(
                    f"GATE_UP_PART q={q} backend={backend} error=incomplete_receipt",
                    flush=True,
                )
            rows.append(dict(q=q, backend=backend, rc=rc, result=out.name))
            print(
                f"GATE_UP_PART q={q} backend={backend} rc={rc} remaining_continue=1",
                flush=True,
            )
    save(
        args.output / "summary.json",
        dict(
            status="PASS" if all(r["rc"] == 0 for r in rows) else "FAIL",
            parts=rows,
            identity=receipt,
            timing="NOT_MEASURED",
            production_selection="UNCHANGED",
        ),
    )
    return int(any(r["rc"] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "bundle", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--q", type=int, choices=(8, 10, 11, 12, 13, 14))
    p.add_argument("--backend", choices=("simt", "tc"))
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete matching format/backend parts",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="check source/library/runtime bytes without GPU work",
    )
    a = p.parse_args()
    if (a.q is None) != (a.backend is None):
        p.error("--q and --backend must be used together")
    if a.verify_only:
        library, manifest = verify(a.bundle, a.sdk)
        print(
            "GATE_UP_VERIFIED sha256="
            + manifest["sha256"]
            + " device_admission=PENDING"
        )
    else:
        raise SystemExit(child(a) if a.q is not None else collect(a))
