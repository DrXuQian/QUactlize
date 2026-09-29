#!/usr/bin/env python3
"""Inspect the two large-BF16 failures using the unchanged candidate DSO.

This does not change admission thresholds or turn a failed numerical gate
green. Save outputs and, for Split-K, pre-activation projections for review.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_kpack_gate_up import (
    Bench,
    Weights,
    Runtime,
    Graph,
    Library,
    Config,
    checked,
    device_identity,
    identity,
    verify,
    save,
    round_compute,
)
from quactlize.runtime.compiler import sha


def activate(gate, up, rounding):
    if rounding:
        gate, up = [round_compute(x, "bf16") for x in (gate, up)]
    with np.errstate(over="ignore", under="ignore"):
        return gate / (np.float32(1) + np.exp(-gate)) * up


def serial_dot(a, weight):
    """One legal F32 order, NOT an emulation of PPU MMA accumulation."""
    result = np.zeros((len(a), len(weight)), dtype="<f4")
    for k in range(a.shape[1]):
        result += a[:, k, None] * weight[None, :, k]
    return result


def install_sparse(bench):
    bench.host_a[:, 1:] = 0
    bench.rt.copy(bench.a, bench.host_a)
    a = round_compute(bench.host_a[:, : bench.w.k], "bf16")
    bench.dot = [
        (a.astype("f8") @ weight[0].astype("f8").T).astype("<f4")
        for weight in (bench.w.gate, bench.w.up)
    ]
    bench.gold = activate(*bench.dot, bench.rounding)


def normalized_error(got, want):
    if not np.isfinite(got).all():
        return None
    return float(
        np.max(np.abs(got.astype("f8") - want))
        / max(1e-20, float(np.max(np.abs(want))))
    )


def observe(bench, split, destination):
    rt, w = bench.rt, bench.w
    raw = rt.download(bench.output, bench.output_bytes)
    got = raw[16:-16].view("<f4").reshape(bench.rows, w.n + 8)[:, : w.n].copy()
    try:
        bench.check(split)
        check_error = None
    except ValueError as error:
        check_error = str(error)
    error = normalized_error(got, bench.gold)
    difference = np.abs(got.astype("f8") - bench.gold)
    at = tuple(map(int, np.unravel_index(np.argmax(difference), difference.shape)))
    finite = lambda x: float(x) if np.isfinite(x) else None
    record = dict(
        original_check="PASS" if check_error is None else "FAIL",
        check_error=check_error,
        error=error,
        nonfinite=int(np.count_nonzero(~np.isfinite(got))),
        worst=dict(
            index=at,
            got=finite(got[at]),
            want=finite(bench.gold[at]),
            gate_gold=float(bench.dot[0][at]),
            up_gold=float(bench.dot[1][at]),
        ),
    )
    arrays = dict(
        got=got,
        want=bench.gold,
        input=bench.host_a,
        gate_gold=bench.dot[0],
        up_gold=bench.dot[1],
    )
    if split > 1:
        partial = rt.download(bench.work + 16, bench.rows * split * 2 * w.n * 4)
        partial = partial.view("<f4").reshape(bench.rows, split, w.n // 4, 2, 4)
        total = np.zeros_like(partial[:, 0])
        for part in range(split):
            total += partial[:, part]
        gate, up = [total[:, :, side].reshape(bench.rows, w.n) for side in (0, 1)]
        record["projections"] = dict(
            gate_error=normalized_error(gate, bench.dot[0]),
            up_error=normalized_error(up, bench.dot[1]),
            activation_of_device_projections_error=normalized_error(
                got, activate(gate, up, bench.rounding)
            ),
        )
        arrays.update(partial=partial, gate_device=gate, up_device=up)
    if bench.rounding:
        a = round_compute(bench.host_a[:, : w.k], "bf16")
        serial = [serial_dot(a, weight[0]) for weight in (w.gate, w.up)]
        serial_output = activate(*serial, 1)
        record["cpu_order_control"] = dict(
            scope="LEGAL_F32_SERIAL_ORDER_NOT_PPU_EMULATION",
            versus_wide_dot_error=normalized_error(serial_output, bench.gold),
            device_versus_serial_error=normalized_error(got, serial_output),
        )
        arrays.update(
            gate_serial=serial[0], up_serial=serial[1], serial_output=serial_output
        )
    np.savez_compressed(destination, **arrays)
    record["arrays_sha256"] = sha(destination)
    return record


def child(args):
    library, manifest = verify(args.bundle, args.sdk)
    rt = Runtime(args.sdk, "ppu")
    rows = []
    device = None
    try:
        device = device_identity(rt)
        lib = Library(library)
        w = Weights(rt, lib, args.q, 256, 2048, 1)
        m = 8 if args.q == 11 else 1
        for tm in (8, 16):
            for split in (1, 8):
                for rounding in (1, 0):
                    bench = Bench(rt, lib, w, 0, m, 1, 1, 1, rounding)
                    try:
                        config = Config(1, split, tm, 0)
                        graph = Graph(rt, [bench.invoke(config)])
                        try:
                            # Reproduce the original preceding A changes on one graph.
                            for repeat in (1, 2):
                                bench.update(repeat)
                                checked(
                                    rt.GraphLaunch(graph.instance, rt.stream),
                                    "preceding replay",
                                )
                                rt.sync()
                            for case in ("mixed-large", "sparse-large"):
                                bench.update(3, large=True)
                                if case == "sparse-large":
                                    install_sparse(bench)
                                bench.poison()
                                checked(
                                    rt.GraphLaunch(graph.instance, rt.stream),
                                    "diagnostic replay",
                                )
                                rt.sync()
                                name = f"q{args.q}-tm{tm}-s{split}-r{rounding}-{case}"
                                record = observe(
                                    bench, split, args.output / (name + ".npz")
                                )
                                record.update(
                                    q=args.q,
                                    m=m,
                                    tile_m=tm,
                                    split=split,
                                    round_projection=rounding,
                                    case=case,
                                    arrays=name + ".npz",
                                )
                                rows.append(record)
                                print(
                                    "GATE_UP_ROUNDING "
                                    + json.dumps(record, allow_nan=False),
                                    flush=True,
                                )
                        finally:
                            graph.close()
                    finally:
                        bench.close()
        save(
            args.output / f"q{args.q}.json",
            dict(
                status="DIAGNOSTIC_COMPLETE",
                device=device,
                identity=identity(manifest),
                records=rows,
                diagnostic_sha256=sha(Path(__file__)),
                timing="NOT_MEASURED",
                numerical_admission="UNCHANGED_PENDING_REVIEW",
            ),
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        save(
            args.output / f"q{args.q}.json",
            dict(status="INFRASTRUCTURE_FAIL", error=str(error), records=rows),
        )
        return 1
    finally:
        rt.close()


def collect(args):
    verify(args.bundle, args.sdk)
    args.output.mkdir(parents=True, exist_ok=False)
    parts = []
    for q in (11, 14):
        command = [
            sys.executable,
            __file__,
            "--sdk",
            str(args.sdk),
            "--bundle",
            str(args.bundle),
            "--output",
            str(args.output),
            "--q",
            str(q),
        ]
        with (args.output / f"q{q}.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
            rc = process.wait()
        parts.append(dict(q=q, rc=rc))
    save(
        args.output / "summary.json",
        dict(
            status=(
                "DIAGNOSTIC_COMPLETE"
                if all(p["rc"] == 0 for p in parts)
                else "INFRASTRUCTURE_FAIL"
            ),
            parts=parts,
            timing="NOT_MEASURED",
            numerical_admission="UNCHANGED_PENDING_REVIEW",
        ),
    )
    return int(any(p["rc"] for p in parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "bundle", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--q", type=int, choices=(11, 14))
    args = parser.parse_args()
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    raise SystemExit(child(args) if args.q else collect(args))
