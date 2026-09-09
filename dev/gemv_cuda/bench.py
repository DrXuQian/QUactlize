#!/usr/bin/env python3
"""Real NVIDIA numerical/performance experiment; never PPU admission."""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quactlize.execution.native import Call, Config, Sizes, arrangement


def checked(rc):
    if rc:
        raise RuntimeError(f"CUDA diagnostic returned {rc}")


def measure(fn, stream, repeats=32):
    # Capture ordinary, real CUDA kernels; Python overhead is outside events.
    with torch.cuda.stream(stream):
        for _ in range(3):
            checked(fn())
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            checked(fn())
    values = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph.replay()
        for _ in range(11):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            values.append(begin.elapsed_time(end) * 1000 / repeats)
    return values


def error(got, golden, denom):
    if not np.isfinite(got).all():
        raise ValueError("output retains poison or is nonfinite")
    err = float(np.max(np.abs(got.astype("f8") - golden) / np.maximum(denom, 1e-30)))
    if err >= 0.005:
        index = np.unravel_index(
            np.argmax(np.abs(got.astype("f8") - golden) / np.maximum(denom, 1e-30)),
            got.shape,
        )
        raise ValueError(
            f"independent official GGUF dot mismatch {err:.7g} index={index} "
            f"got={got[index]} want={golden[index]} denom={denom[index]} "
            f"first_got={got[:,0].tolist()} first_want={golden[:,0].tolist()}"
        )
    return err


def bind(lib, kind):
    suffix = "pair_" if kind == "pair" else ""
    query = getattr(lib, f"quactlize_kpack_gemv_{suffix}query_v1")
    run = getattr(lib, f"quactlize_kpack_gemv_{suffix}run_v1")
    from quactlize.execution.native import Arrangement

    arguments = [C.POINTER(Call), C.POINTER(Config), C.POINTER(Arrangement)]
    query.argtypes, query.restype = arguments + [C.POINTER(Sizes)], C.c_int
    run.argtypes, run.restype = arguments, C.c_int
    return query, run


def gemv(lib, fixture_path, *, profile=None, quick=False):
    fixture = np.load(fixture_path, allow_pickle=False)
    q, n, k, experts, channels = [
        int(fixture[x]) for x in ("q", "n", "k", "experts", "channels")
    ]
    selected = fixture["ids"].tolist()
    rows = len(selected)
    mode = 0 if experts == 1 else 2
    storage = {}
    for name in ("low", "high", "units"):
        source = fixture[name]
        if not source.size:
            storage[name] = None
            continue
        source = source.view("u1").reshape(rows, -1)
        dest = torch.zeros((experts, source.shape[1]), dtype=torch.uint8, device="cuda")
        for slot, e in enumerate(selected):
            dest[e].copy_(torch.from_numpy(source[slot]))
        storage[name] = dest
    # NumPy's column indexing may produce Fortran-contiguous channels x K.
    # The C ABI below explicitly names row-major strides.
    a = torch.from_numpy(np.ascontiguousarray(fixture["a"])).cuda()
    assert a.is_contiguous() and a.stride(-1) == 1
    ids = (
        torch.tensor(selected, dtype=torch.int32, device="cuda") if mode == 2 else None
    )
    output = torch.full((rows * n + 8,), float("nan"), device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    arr = arrangement(q)
    results = []
    configs = [
        (kind, c, w, s)
        for kind in ("scalar", "pair")
        for c in (16, 32)
        for w in ((4, 8) if kind == "scalar" else (2, 4, 8))
        for s in (
            ((1,) if quick else (1, 4))
            if kind == "scalar"
            else ((1, 2) if quick else (1, 2, 4, 8))
        )
    ]
    if profile:
        kind, c, w, s = profile.split(",")
        configs = [(kind, int(c), int(w), int(s))]
    for kind, columns, warps, split in configs:
        query, run = bind(lib, kind)
        cfg = Config(columns, warps, split)
        c = Call(
            version=1,
            size=C.sizeof(Call),
            qtype=q,
            n=n,
            k=k,
            experts=experts,
            rows=rows,
            mode=mode,
            input_type=1,
            channels=channels,
            topk=rows if mode else 1,
            a_row_stride=k,
            a_token_stride=k * channels,
            ids_stride=rows,
            out_row_stride=n,
            a=a.data_ptr(),
            low=storage["low"].data_ptr(),
            high=storage["high"].data_ptr() if storage["high"] is not None else None,
            units=storage["units"].data_ptr(),
            ids=ids.data_ptr() if ids is not None else None,
            output=output.data_ptr() + 16,
            stream=stream.cuda_stream,
        )
        sizes = Sizes()
        checked(query(C.byref(c), C.byref(cfg), C.byref(arr), C.byref(sizes)))
        workspace = torch.full(
            (sizes.workspace_bytes // 4 + 8,), float("nan"), device="cuda"
        )
        c.workspace = workspace.data_ptr() + 16 if sizes.workspace_bytes else None
        c.workspace_bytes = sizes.workspace_bytes
        torch.cuda.synchronize()
        fn = lambda: run(C.byref(c), C.byref(cfg), C.byref(arr))
        checked(fn())
        stream.synchronize()
        if profile:
            print(
                f"CUDA_GEMV_PROFILE case={fixture_path.name} kind={kind} c={columns} w={warps} s={split}",
                flush=True,
            )

        def check():
            out = output.cpu().numpy()
            if not np.isnan(out[:4]).all() or not np.isnan(out[-4:]).all():
                raise ValueError("output guard changed")
            got = out[4:-4].reshape(rows, n)
            err = error(got, fixture["golden"], fixture["denom"])
            ws = workspace.cpu().numpy()
            if not np.isnan(ws[:4]).all() or not np.isnan(ws[-4:]).all():
                raise ValueError("workspace guard changed")
            if split > 1:
                parts = ws[4:-4].reshape(rows, split, n)
                total = np.zeros((rows, n), dtype="f4")
                for s in range(split):
                    np.add(total, parts[:, s], out=total)
                if not np.array_equal(total.view("u4"), got.view("u4")):
                    raise ValueError("SIMT ordered reducer differs from FP32 partials")
            return err

        err = check()
        if profile:
            torch.cuda.profiler.start()
            checked(fn())
            stream.synchronize()
            torch.cuda.profiler.stop()
            samples = []
        else:
            samples = measure(fn, stream)
        check()
        row = dict(
            case=fixture_path.name,
            q=q,
            shape=[rows, n, k],
            experts=experts,
            kind=kind,
            columns=columns,
            warps=warps,
            split=split,
            error=err,
            samples_us=samples,
            median_us=statistics.median(samples) if samples else None,
            grid=rows * split * (n // columns),
            status="PASS",
            scope="INDEXED_F32_INPUT_OUTPUT_NO_GATHER_SCATTER",
        )
        print("CUDA_GEMV " + json.dumps(row), flush=True)
        results.append(row)
        if len(results) == 1:
            # A zeroed low-code plane must not retain the correct output.
            saved = storage["low"]
            damaged = torch.zeros_like(saved)
            torch.cuda.synchronize()
            c.low = damaged.data_ptr()
            checked(fn())
            stream.synchronize()
            try:
                error(
                    output[4:-4].cpu().numpy().reshape(rows, n),
                    fixture["golden"],
                    fixture["denom"],
                )
            except ValueError:
                pass
            else:
                raise ValueError("zero-code negative not detected")
            c.low = saved.data_ptr()
    return results


def reducers(lib):
    fn = lib.qkg_cuda_reduce
    fn.argtypes, fn.restype = [C.c_void_p, C.c_void_p] + [C.c_int] * 5 + [
        C.c_void_p
    ], C.c_int
    stream = torch.cuda.Stream()
    results = []
    fast = lib.qkg_cuda_reduce_fast
    fast.argtypes, fast.restype = [C.c_void_p, C.c_void_p] + [C.c_int] * 4, C.c_int
    for m, n in ((8, 512), (8, 2048), (1, 4096), (9, 66)):
        for s in (2, 4, 8):
            rng = np.random.default_rng(95173 + m + n + s)
            parts = rng.uniform(-100, 100, (s, m, n)).astype("f4")
            # Cancellation-sensitive order control, not only positive sums.
            if s >= 4:
                parts[:4, 0, 0] = [2**24, 1, -(2**24), 1]
            expected = np.zeros((m, n), dtype="f4")
            for part in parts:
                np.add(expected, part, out=expected)
            expected = expected.astype("f2")
            for offset in (16, 128):
                workspace = torch.full(
                    (parts.size + offset // 4 + 4,), float("nan"), device="cuda"
                )
                workspace[offset // 4 : -4].copy_(torch.from_numpy(parts.reshape(-1)))
                output = torch.full(
                    (m * n + 16,), float("nan"), dtype=torch.float16, device="cuda"
                )
                torch.cuda.synchronize()
                assert fast(
                    workspace.data_ptr() + offset, output.data_ptr() + 16, m, n, n, s
                ) == int((m * n) % 64 == 0)
                assert (
                    fast(
                        workspace.data_ptr() + offset,
                        output.data_ptr() + 16,
                        m,
                        n,
                        n + 1,
                        s,
                    )
                    == 0
                )
                assert (
                    fast(
                        workspace.data_ptr() + offset,
                        output.data_ptr() + 18,
                        m,
                        n,
                        n,
                        s,
                    )
                    == 0
                )
                for arm in (0, 1, 2, 4, 8):
                    launch = lambda: fn(
                        workspace.data_ptr() + offset,
                        output.data_ptr() + 16,
                        m,
                        n,
                        n,
                        s,
                        arm,
                        stream.cuda_stream,
                    )
                    rc = launch()
                    if rc and arm in (1, 2, 4) and (m * n) % (32 * arm):
                        results.append(
                            dict(
                                shape=[m, n],
                                split=s,
                                arm=arm,
                                status="UNSUPPORTED_TAIL",
                            )
                        )
                        continue
                    checked(rc)
                    stream.synchronize()
                    got = output[8:-8].cpu().numpy()
                    if not np.array_equal(
                        got.view("u2"), expected.reshape(-1).view("u2")
                    ):
                        raise ValueError(f"reducer raw-bit mismatch {m,n,s,arm,offset}")
                    samples = measure(launch, stream, 64)
                    if not np.array_equal(
                        output[8:-8].cpu().numpy().view("u2"),
                        expected.reshape(-1).view("u2"),
                    ):
                        raise ValueError("post-timing reducer mismatch")
                    if (
                        not torch.isnan(output[:8]).all()
                        or not torch.isnan(output[-8:]).all()
                    ):
                        raise ValueError("reducer guard changed")
                    row = dict(
                        shape=[m, n],
                        split=s,
                        arm=arm,
                        workspace_offset=offset,
                        samples_us=samples,
                        median_us=statistics.median(samples),
                        status="PASS",
                    )
                    results.append(row)
                    print("CUDA_REDUCER " + json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--profile", help="single kind,columns,warps,split")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", choices=("gemv", "reducer"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    manifest = json.loads((args.build / "manifest.json").read_text())
    so = args.build / manifest["library"]
    if hashlib.sha256(so.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise ValueError("build payload differs")
    lib = C.CDLL(str(so.resolve()))
    result = dict(
        device=str(torch.cuda.get_device_properties(0)),
        torch=torch.__version__,
        build=manifest,
        ppu_admission=False,
        gemv=[],
        reducers=[],
    )
    direct = lib.qkg_cuda_direct_test
    direct.argtypes, direct.restype = [], C.c_int
    checked(direct())
    result["direct_store"] = "PASS_216_CELLS_PPU_C_OWNERSHIP_PROJECTION"
    for record in json.loads((args.fixtures / "manifest.json").read_text()):
        if args.only == "reducer" or (args.case and args.case != record["path"]):
            continue
        path = args.fixtures / record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("fixture payload differs")
        result["gemv"] += gemv(lib, path, profile=args.profile, quick=args.quick)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.only != "gemv" and not args.profile:
        result["reducers"] = reducers(lib)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
