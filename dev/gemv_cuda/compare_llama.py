#!/usr/bin/env python3
"""Matched CUDA DMMV/MMVQ/K-pack arithmetic and graph-time comparison."""

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
from dev.gemv_cuda.bench import bind, checked, error, measure
from quactlize.execution.native import Call, Config, Sizes, arrangement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(fixture_path, kpack, reference, *, profile_kernels=False):
    f = np.load(fixture_path, allow_pickle=False)
    q, n, k, e, channels = [int(f[x]) for x in ("q", "n", "k", "experts", "channels")]
    chosen = f["ids"].tolist()
    rows = len(chosen)
    stream = torch.cuda.Stream()
    storage = {}
    for name in ("raw", "low", "high", "units"):
        values = f[name].view("u1").reshape(rows, -1)
        if not values.size:
            storage[name] = None
            continue
        dest = torch.zeros((e, values.shape[1]), dtype=torch.uint8, device="cuda")
        for slot, expert in enumerate(chosen):
            dest[expert].copy_(torch.from_numpy(values[slot]))
        storage[name] = dest
    a = torch.from_numpy(np.ascontiguousarray(f["a"])).cuda()
    ids = torch.tensor(chosen, dtype=torch.int32, device="cuda") if e > 1 else None
    output = torch.full((rows * n + 8,), float("nan"), device="cuda")
    quantized = torch.empty(channels * (k // 32) * 36, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    outptr = output.data_ptr() + 16
    idptr = ids.data_ptr() if ids is not None else None
    call = Call(
        version=1,
        size=C.sizeof(Call),
        qtype=q,
        n=n,
        k=k,
        experts=e,
        rows=rows,
        mode=2 if e > 1 else 0,
        input_type=1,
        channels=channels,
        topk=rows if e > 1 else 1,
        a_row_stride=k,
        a_token_stride=k * channels,
        ids_stride=rows,
        out_row_stride=n,
        a=a.data_ptr(),
        low=storage["low"].data_ptr(),
        high=storage["high"].data_ptr() if storage["high"] is not None else None,
        units=storage["units"].data_ptr(),
        ids=idptr,
        output=outptr,
        stream=stream.cuda_stream,
    )
    arr = arrangement(q)
    query, run = bind(kpack, "pair")

    def check_output():
        stream.synchronize()
        out = output.cpu().numpy()
        if not np.isnan(out[:4]).all() or not np.isnan(out[-4:]).all():
            raise ValueError("output guard changed")
        return error(out[4:-4].reshape(rows, n), f["golden"], f["denom"])

    def validate(fn):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
        checked(fn())
        return check_output()

    def make_pair(columns, warps, split, *, activation=a, input_type=1):
        c = Call.from_buffer_copy(call)
        c.a = activation.data_ptr()
        c.input_type = input_type
        cfg, sizes = Config(columns, warps, split), Sizes()
        checked(query(C.byref(c), C.byref(cfg), C.byref(arr), C.byref(sizes)))
        workspace = torch.full(
            (sizes.workspace_bytes // 4 + 8,), float("nan"), device="cuda"
        )
        c.workspace = workspace.data_ptr() + 16 if sizes.workspace_bytes else None
        c.workspace_bytes = sizes.workspace_bytes
        torch.cuda.synchronize()
        fn = lambda: run(C.byref(c), C.byref(cfg), C.byref(arr))

        def verify():
            err = check_output()
            partial = workspace.cpu().numpy()
            if not np.isnan(partial[:4]).all() or not np.isnan(partial[-4:]).all():
                raise ValueError("workspace guard changed")
            if split > 1:
                partial = partial[4:-4].reshape(rows, split, n)
                total = np.zeros((rows, n), dtype="f4")
                for s in range(split):
                    np.add(total, partial[:, s], out=total)
                got = output[4:-4].cpu().numpy().reshape(rows, n)
                if not np.array_equal(total.view("u4"), got.view("u4")):
                    raise ValueError("ordered K-pack reducer mismatch")
            return err

        return fn, verify

    screen = []
    for columns in (16, 32):
        for warps in (2, 4, 8):
            for split in (1, 2, 4, 8):
                fn, check = make_pair(columns, warps, split)
                validate(fn)
                samples = measure(fn, stream)
                screen.append(
                    dict(
                        columns=columns,
                        warps=warps,
                        split=split,
                        error=check(),
                        samples_us=samples,
                        median_us=statistics.median(samples),
                    )
                )
    winner = min(screen, key=lambda x: x["median_us"])
    pair, pair_check = make_pair(winner["columns"], winner["warps"], winner["split"])
    arguments = (
        q,
        n,
        k,
        rows,
        channels,
        storage["raw"].data_ptr(),
        a.data_ptr(),
        idptr,
        outptr,
        quantized.data_ptr(),
        stream.cuda_stream,
    )
    funcs = {
        "dmmv_fpA": lambda: reference(1, *arguments),
        "mmvq_q8A": lambda: reference(0, *arguments),
        "kpack_pair": pair,
    }
    errors = {name: validate(fn) for name, fn in funcs.items()}
    if ids is not None:
        with torch.cuda.stream(stream):
            ids.fill_(chosen[0])
        for name, fn in funcs.items():
            checked(fn())
            try:
                check_output()
            except ValueError:
                pass
            else:
                raise ValueError("wrong-expert negative not detected: " + name)
        with torch.cuda.stream(stream):
            ids.copy_(torch.tensor(chosen, dtype=torch.int32))
        stream.synchronize()
    rounds = {name: [] for name in funcs}
    for iteration in range(6):
        names = list(funcs) if iteration % 2 == 0 else list(reversed(funcs))
        for name in names:
            validate(funcs[name])
            rounds[name].append(measure(funcs[name], stream))
            pair_check() if name == "kpack_pair" else check_output()
    medians = {
        name: statistics.median(statistics.median(v) for v in samples)
        for name, samples in rounds.items()
    }
    alignment_controls = []
    for dtype, offset in ((torch.float32, 1), (torch.float16, 0), (torch.float16, 1)):
        activation = torch.empty(a.numel() + offset, dtype=dtype, device="cuda")[
            offset:
        ]
        activation.copy_(a.reshape(-1))
        fn, check = make_pair(
            winner["columns"],
            winner["warps"],
            winner["split"],
            activation=activation,
            input_type=1 if dtype == torch.float32 else 0,
        )
        validate(fn)
        alignment_controls.append(
            dict(
                dtype=str(dtype),
                byte_offset=offset * activation.element_size(),
                error=check(),
            )
        )
    kernel_events = {}
    if profile_kernels:
        for name, fn in funcs.items():
            validate(fn)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                checked(fn())
                stream.synchronize()
            kernel_events[name] = [
                dict(name=event.name, us=event.device_time_total)
                for event in prof.events()
                if str(event.device_type) == "DeviceType.CUDA"
            ]
            check_output()
            print(
                "CUDA_LLAMA_KERNEL_EVENTS "
                + json.dumps(
                    dict(
                        case=fixture_path.name,
                        arm=name,
                        events=kernel_events[name],
                        scope="PROFILE_NOT_WARM_GRAPH_TIMING",
                    )
                ),
                flush=True,
            )
    result = dict(
        case=fixture_path.name,
        shape=[rows, n, k],
        q=q,
        experts=e,
        channels=channels,
        status="PASS",
        oracle="OFFICIAL_GGUF_FP64_DOT",
        errors=errors,
        fixture_sha256=sha(fixture_path),
        raw_sha256=hashlib.sha256(f["raw"].tobytes()).hexdigest(),
        activation_sha256=hashlib.sha256(f["a"].tobytes()).hexdigest(),
        ids=chosen,
        kpack_recipe={x: winner[x] for x in ("columns", "warps", "split")},
        screen=screen,
        samples_us=rounds,
        median_us=medians,
        delta_vs_dmmv_pct=(medians["kpack_pair"] / medians["dmmv_fpA"] - 1) * 100,
        delta_vs_mmvq_pct=(medians["kpack_pair"] / medians["mmvq_q8A"] - 1) * 100,
        scope="SAME_RAW_WEIGHTS_F32_A_INDEXED_F32_OUTPUT_WARM_GRAPH",
        fusion=False,
        mmvq_q8_quantization_included=True,
        kpack_reducer_included=True,
        profiled_kernels=kernel_events,
        alignment_controls=alignment_controls,
        historical_indexing="GPU_WRAPPER_ORIGINAL_DOT_BODY",
        calls_per_graph=32,
    )
    print(
        "CUDA_LLAMA_COMPARISON "
        + json.dumps(
            {k: v for k, v in result.items() if k not in ("screen", "samples_us")}
        ),
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--kpack-build", type=Path, required=True)
    p.add_argument("--reference-build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-kernels", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    km = json.loads((args.kpack_build / "manifest.json").read_text())
    rm = json.loads((args.reference_build / "manifest.json").read_text())
    kl = args.kpack_build / "libkpack_gemv_cuda.so"
    rl = args.reference_build / "libllama_reference.so"
    if sha(kl) != km["library_sha256"] or sha(rl) != rm["library_sha256"]:
        raise ValueError("CUDA library identity differs")
    reference = C.CDLL(str(rl.resolve())).llama_reference_run
    reference.argtypes = [C.c_int] * 6 + [C.c_void_p] * 6
    reference.restype = C.c_int
    kpack = C.CDLL(str(kl.resolve()))
    cases = [
        "q12-n512-k2048-e256-c1.npz",
        "q13-n2048-k512-e256-c8.npz",
        "q12-n4096-k2048-e1-c1.npz",
    ]
    result = dict(
        device=torch.cuda.get_device_name(),
        kpack_build=km,
        reference_build=rm,
        torch=torch.__version__,
        results=[],
        ppu_admission=False,
        measurement_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in (Path(__file__), ROOT / "dev/gemv_cuda/bench.py")
        },
    )
    for case in cases:
        result["results"].append(
            compare(
                args.fixtures / case,
                kpack,
                reference,
                profile_kernels=args.profile_kernels,
            )
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("CUDA_LLAMA_COMPARISON_COMPLETE cases=3 status=PASS", flush=True)


if __name__ == "__main__":
    main()
