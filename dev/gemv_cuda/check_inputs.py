#!/usr/bin/env python3
"""CUDA GEMV admission for multi-token IDs, ragged groups and b16 offsets."""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.bench import bind, checked, error
from quactlize.execution.native import Call, Config, Sizes, arrangement


def cases():
    for tokens in (1, 2, 3, 4):
        for channels in (1, 8):
            yield 2, tokens, channels
    for rows in (1, 2, 4):
        yield 0, rows, 1
    yield 1, 6, 1


def q4_recipe_extension(extra, tree):
    recipes = [(4,8,1),(8,8,1),(4,4,2),(8,2,4)] if extra else []
    if tree:
        recipes += [(1,2,1),(2,4,2),(4,16,1),(8,16,8),(16,16,2),(32,16,4)]
    return recipes


def run(library, fixture, *, baseline=None, extra_q4_recipes=False, f16_aligned=False,
        q4_tree_recipes=False):
    q, n, k, experts = (int(fixture[x]) for x in ("q", "n", "k", "experts"))
    selected = fixture["ids"].tolist()
    assert len(selected) == 8 and experts == 16
    # Exported once by official gguf.quants.dequantize on the raw GGUF
    # bytes, never derived by this kernel or its K-pack reader.
    official = fixture["official"].reshape(8, n, k).astype("f8")
    storage = {}
    for name in ("low", "high", "units"):
        src = fixture[name].view("u1").reshape(8, -1)
        dest = torch.zeros((experts, src.shape[1]), dtype=torch.uint8, device="cuda")
        for slot, expert in enumerate(selected):
            dest[expert].copy_(torch.from_numpy(src[slot]))
        storage[name] = dest
    query, launch = bind(library, "pair")
    baseline_launch = bind(baseline, "pair")[1] if baseline is not None else None
    arr = arrangement(q)
    stream = torch.cuda.Stream()
    records = []
    for mode, tokens, channels in cases():
        rows = tokens * 8 if mode == 2 else tokens
        ids = np.stack([np.roll(selected, t * 3) for t in range(tokens)]).astype("i4")
        counts = np.zeros(experts, dtype="i4")
        counts[[0, 4, 6]] = [2, 3, 1]
        offsets = np.concatenate(([0], np.cumsum(counts))).astype("i4")
        owners = (ids.reshape(-1) if mode == 2 else np.repeat(np.arange(experts), counts)
                  if mode == 1 else np.zeros(rows, dtype="i4"))
        slot_of = {expert: slot for slot, expert in enumerate(selected)}
        input_cases=[(torch.float32,0),(torch.float32,1),(torch.float16,1)]
        if f16_aligned:
            input_cases.append((torch.float16,0))
        for dtype, offset in input_cases:
            a_rows = tokens * channels if mode == 2 else rows
            stride = k + (3 if offset else 4)
            rng = np.random.default_rng(99231 + q + rows + offset)
            src = rng.normal(0, .2, (a_rows, stride)).astype("f4")
            # All consumers have an FP16-A boundary, even with F32 input.
            a16 = src[:, :k].astype("f2").astype("f8")
            inp = torch.empty(a_rows * stride + offset, dtype=dtype, device="cuda")[offset:]
            inp.copy_(torch.from_numpy(src.reshape(-1)))
            expected, denom = [], []
            for row, expert in enumerate(owners):
                arow = row // 8 * channels + row % 8 % channels if mode == 2 else row
                weight = official[slot_of[int(expert)]]
                expected.append(weight @ a16[arow])
                denom.append(np.abs(weight) @ np.abs(a16[arow]))
            expected, denom = np.array(expected), np.array(denom)
            ids_dev = torch.from_numpy(ids).cuda() if mode == 2 else None
            offsets_dev = torch.from_numpy(offsets).cuda() if mode == 1 else None
            # Exercise ABI-permitted b16-but-not-b32 plane alignment. Units
            # remain at their original byte layout; this is not repacking.
            planes = {}
            for name, value in storage.items():
                if not value.numel():
                    planes[name] = value
                else:
                    shift = 2 if offset else 0
                    planes[name] = torch.empty(value.numel() + shift, dtype=torch.uint8,
                                               device="cuda")[shift:]
                    planes[name].copy_(value.reshape(-1))
            output_stride = n + 8
            output = torch.full((rows * output_stride + 8,), float("nan"), device="cuda")
            call = Call(version=1, size=C.sizeof(Call), qtype=q, n=n, k=k,
                        experts=1 if mode == 0 else experts, rows=rows, mode=mode,
                        input_type=int(dtype == torch.float32), channels=channels, topk=8 if mode == 2 else 1,
                        a_row_stride=stride, a_token_stride=stride * channels, ids_stride=8,
                        out_row_stride=output_stride, a=inp.data_ptr(), low=planes["low"].data_ptr(),
                        high=planes["high"].data_ptr() if planes["high"].numel() else None,
                        units=planes["units"].data_ptr(), ids=ids_dev.data_ptr() if mode == 2 else None,
                        offsets=offsets_dev.data_ptr() if mode == 1 else None,
                        output=output.data_ptr() + 16, stream=stream.cuda_stream)
            recipes=[(16,4,8),(32,8,1)]
            if q==12:
                recipes += q4_recipe_extension(extra_q4_recipes, q4_tree_recipes)
            for columns, warps, split in recipes:
                cfg, sizes = Config(columns, warps, split), Sizes()
                checked(query(C.byref(call), C.byref(cfg), C.byref(arr), C.byref(sizes)))
                ws = torch.full((sizes.workspace_bytes // 4 + 8,), float("nan"), device="cuda")
                call.workspace, call.workspace_bytes = ws.data_ptr() + 16, sizes.workspace_bytes
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    checked(launch(C.byref(call), C.byref(cfg), C.byref(arr)))
                worst = 0
                for replay in range(3):
                    with torch.cuda.stream(stream):
                        output.fill_(float("nan"))
                        ws.fill_(float("nan"))
                        graph.replay()
                    stream.synchronize()
                    out = output.cpu().numpy()
                    assert np.isnan(out[:4]).all() and np.isnan(out[-4:]).all()
                    matrix = out[4:-4].reshape(rows, output_stride)
                    assert np.isnan(matrix[:, n:]).all()
                    worst = max(worst, error(matrix[:, :n], expected, denom))
                    partial = ws.cpu().numpy()
                    assert np.isnan(partial[:4]).all() and np.isnan(partial[-4:]).all()
                    if split > 1:
                        p = partial[4:-4].reshape(rows, split, n)
                        total = np.zeros((rows, n), dtype="f4")
                        for s in range(split):
                            np.add(total, p[:, s], out=total)
                        assert np.array_equal(total.view("u4"), matrix[:, :n].view("u4"))
                if baseline_launch is not None:
                    # Separate output allocations are unnecessary: matrix is
                    # already a CPU copy of the candidate's last graph replay.
                    with torch.cuda.stream(stream):
                        output.fill_(float("nan"))
                        ws.fill_(float("nan"))
                    checked(baseline_launch(C.byref(call), C.byref(cfg), C.byref(arr)))
                    stream.synchronize()
                    base = output.cpu().numpy()[4:-4].reshape(rows, output_stride)
                    if not np.array_equal(base[:, :n].view("u4"), matrix[:, :n].view("u4")):
                        bad = int(np.count_nonzero(base[:, :n].view("u4") != matrix[:, :n].view("u4")))
                        raise ValueError(f"N2 baseline output bits differ: {bad}/{rows*n}")
                if mode == 2:
                    with torch.cuda.stream(stream):
                        ids_dev.fill_(selected[0])
                        graph.replay()
                    stream.synchronize()
                    try:
                        error(output[4:-4].cpu().numpy().reshape(rows, output_stride)[:, :n], expected, denom)
                    except ValueError:
                        pass
                    else:
                        raise ValueError("wrong-expert negative insensitive")
                    with torch.cuda.stream(stream):
                        ids_dev.copy_(torch.from_numpy(ids))
                    stream.synchronize()
                record = dict(q=q, mode=mode, tokens=tokens if mode == 2 else None,
                              rows=rows, channels=channels, dtype=str(dtype), a_offset=offset,
                              plane_offset=2 if offset else 0, config=[columns, warps, split],
                              repeats=3, error=worst, status="PASS",
                              baseline_bitwise="PASS" if baseline_launch is not None else "NOT_REQUESTED")
                records.append(record)
                print("GEMV_INPUT_CONTROL " + json.dumps(record), flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path,
                   help="optional same-ownership N2 library for exact output-bit comparison")
    p.add_argument("--extra-q4-recipes", action="store_true",
                   help="also validate the new C4/C8 domains; baseline must expose the same recipes")
    p.add_argument("--f16-aligned", action="store_true",
                   help="also exercise fast F16 endpoints for every indexed/grouped context")
    p.add_argument("--q4-tree-recipes", action="store_true",
                   help="also cover C1/C2 and W16; FP32 sum order differs from the serial N2 reference")
    a = p.parse_args()
    if a.q4_tree_recipes and a.baseline:
        p.error("tree reduction changes FP32 addition order; use the independent oracle, not exact N2 bits")
    if a.output.exists():
        raise ValueError("output already exists")
    records = []
    library = C.CDLL(str(a.library.resolve()))
    baseline = C.CDLL(str(a.baseline.resolve())) if a.baseline else None
    paths = sorted(a.fixtures.glob("*.npz"))
    if len(paths) != 5:
        raise ValueError("expected five format fixtures")
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            records.extend(run(library, f, baseline=baseline, extra_q4_recipes=a.extra_q4_recipes,
                               f16_aligned=a.f16_aligned,q4_tree_recipes=a.q4_tree_recipes))
    expected=12*(4 if a.f16_aligned else 3)*(10+len(q4_recipe_extension(
        a.extra_q4_recipes,a.q4_tree_recipes)))
    assert len(records) == expected
    a.output.write_text(json.dumps(dict(status="PASS", device=torch.cuda.get_device_name(),
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest() if a.baseline else None,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
        ppu_admission=False, records=records), indent=2) + "\n")
    print(f"GEMV_INPUT_CONTROLS PASS cells={len(records)} formats=5 tokens=1,2,3,4")


if __name__ == "__main__":
    main()
