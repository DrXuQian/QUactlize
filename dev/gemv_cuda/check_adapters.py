#!/usr/bin/env python3
"""Real-CUDA check of the common-endpoint adapters, not of any PPU GEMM."""

import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import torch


def run(library):
    fn = C.CDLL(str(library.resolve())).qkg_comparison_adapter_v1
    fn.argtypes = [
        C.c_int,
        C.c_void_p,
        C.c_void_p,
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_void_p,
    ]
    fn.restype = C.c_int
    results = []
    for m, ch, n in (
        (1, 1, 512),
        (8, 1, 512),
        (8, 8, 512),
        (8, 8, 2048),
        (1, 1, 25600),
        (8, 8, 66),
    ):
        stream = torch.cuda.Stream()
        src = np.linspace(-3, 3, ch * n, dtype="f4").reshape(ch, n)
        inp = torch.from_numpy(src).cuda()
        gathered = torch.full(
            (m * n + 16,), float("nan"), dtype=torch.float16, device="cuda"
        )
        values = np.linspace(-2, 5, m * n, dtype="f4").reshape(m, n).astype("f2")
        halfout = torch.from_numpy(values).cuda()
        output = torch.full(
            (m * n + 8,), float("nan"), dtype=torch.float32, device="cuda"
        )
        order = torch.arange(m, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()

        def launch():
            assert (
                fn(
                    0,
                    inp.data_ptr(),
                    gathered.data_ptr() + 16,
                    order.data_ptr(),
                    m,
                    n,
                    ch,
                    stream.cuda_stream,
                )
                == 0
            )
            assert (
                fn(
                    1,
                    halfout.data_ptr(),
                    output.data_ptr() + 16,
                    order.data_ptr(),
                    m,
                    n,
                    ch,
                    stream.cuda_stream,
                )
                == 0
            )

        launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        for shift in (0, 1, 3):
            perm = np.roll(np.arange(m, dtype="i4"), shift)
            with torch.cuda.stream(stream):
                order.copy_(torch.from_numpy(perm))
                gathered.fill_(float("nan"))
                output.fill_(float("nan"))
                graph.replay()
            stream.synchronize()
            got = gathered.cpu().numpy()
            out = output.cpu().numpy()
            assert np.isnan(got[:8]).all() and np.isnan(got[-8:]).all()
            assert np.isnan(out[:4]).all() and np.isnan(out[-4:]).all()
            assert np.array_equal(
                got[8:-8].reshape(m, n).view("u2"),
                src[perm % ch].astype("f2").view("u2"),
            )
            expected = np.empty((m, n), dtype="f4")
            expected[perm] = values.astype("f4")
            assert np.array_equal(
                out[4:-4].reshape(m, n).view("u4"), expected.view("u4")
            )
        results.append(dict(m=m, channels=ch, extent=n, replays=3, status="PASS"))
    print(
        "GEMV_FQ_SF_ADAPTER_CUDA "
        + json.dumps(
            dict(
                status="PASS",
                cases=results,
                scope="COMMON_ENDPOINT_CAST_AND_PERMUTATION_ONLY_NOT_PPU_GEMM",
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("library", type=Path)
    run(p.parse_args().library)
