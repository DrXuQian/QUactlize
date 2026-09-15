"""Device capability cases over frozen explicit-compute parents."""
import ctypes as C
from functools import lru_cache

import numpy as np

from dev.bf16_compute.fixture import Weights, bf16_bits, bf16_float, compare, digest, round_compute
from dev.bf16_compute.native import (
    Buffer, TensorCore, Resources, SimtCall, SimtCallV2, SimtConfig, Sizes,
    arrangement, bind_simt_compute, load,
)
from dev.bf16_compute.plan import grouped_rows, Parent
from quactlize.runtime.native import checked
from tools.run_kpack_grouped_decode_probe import Replay


@lru_cache(maxsize=2)
def weights(q, n, k, experts, seed=710):
    return Weights(q, n, k, experts, seed=seed)


def source(rows, k, repeat):
    return np.random.default_rng(191 + repeat).uniform(-0.5, 0.5, (rows, k)).astype("f4")


def selected_record(package, case):
    parent = case["parent"]
    if isinstance(parent, dict):
        parent = Parent(**parent)
    return package["modules"][parent.key]


def grouped(root, package, sdk, case, repeats, samples):
    rows = grouped_rows(case["profile"])
    w = weights(case["q"], 256, 2048, len(rows))
    r, kernel, graph = Resources(sdk), None, None
    try:
        kernel = TensorCore(root, selected_record(package, case), sdk, r, w, int(rows.sum()),
                            int(rows.max()), case["algorithm"], case["split"])
        graph = Replay(sdk, r.stream, kernel.run, 1)
        proofs = []
        for repeat in range(repeats):
            rows = grouped_rows(case["profile"], repeat)
            owners = np.repeat(np.arange(len(rows)), rows)
            act = source(len(owners), w.k, repeat)
            gold, denom = w.dot(act, owners, case["compute"], output_compute=True)
            kernel.upload_a(act)
            kernel.offsets.upload(np.r_[0, rows.cumsum()].astype("<i4"))
            kernel.output.poison()
            checked(graph() if repeat % 2 == 0 else kernel.run(), "grouped typed execution")
            sdk.synchronize(r.stream)
            proof = compare(kernel.read(), gold, denom)
            proof.update(input_sha256=digest(act), rows_sha256=digest(rows))
            kernel.workspace.guard()
            proofs.append(proof)
        kernel.a.upload(np.zeros((kernel.call.m, w.k), dtype="<u2"))
        checked(graph(), "grouped zero-A negative")
        sdk.synchronize(r.stream)
        zero = kernel.read()
        if np.any(zero != 0) or not np.isfinite(zero).all():
            raise ValueError("zero A did not produce zero grouped output")
        try:
            compare(zero, gold, denom)
        except ValueError:
            negative = "RED"
        else:
            raise ValueError("independent grouped oracle accepted wrong A")
        kernel.upload_a(act)
        checked(graph(), "excluded grouped warmup")
        sdk.synchronize(r.stream)
        timing = r.samples(graph, samples) if samples else []
        return dict(status="PASS", selection=kernel.receipt, proofs=proofs,
                    fixture=w.record(), zero_a_negative=negative, samples_us=timing)
    finally:
        sdk.synchronize(r.stream)
        if graph:
            graph.close()
        if kernel:
            kernel.close()
        r.close()


def simt_inputs(w, case, repeat):
    tokens, mode, channels = case["tokens"], case["mode"], case["channels"]
    topk = 8 if mode == 2 else 1
    rows = tokens * topk
    stride = w.k + 8
    act = np.full((tokens * channels if mode == 2 else rows, stride), 19, dtype="f4")
    act[:, :w.k] = source(len(act), w.k, repeat)
    if mode == 2:
        ids = np.full((tokens, 11), -17, dtype="i4")
        ids[:, :8] = (np.arange(8)[None, :] + 3 * np.arange(tokens)[:, None] + repeat * 7) % w.experts
        owners = ids[:, :8].ravel()
        from_rows = np.arange(rows) // 8 * channels + np.arange(rows) % 8 % channels
        offsets = None
    elif mode == 1:
        owners = np.arange(rows) * w.experts // rows
        counts = np.bincount(owners, minlength=w.experts)
        offsets = np.r_[0, counts.cumsum()].astype("i4")
        from_rows, ids = np.arange(rows), None
    else:
        owners, from_rows, ids, offsets = np.zeros(rows, dtype="i4"), np.arange(rows), None, None
    return act, owners, from_rows, ids, offsets


def simt(root, package, sdk, case, repeats, samples):
    w = weights(case["q"], 256, 2048, 1 if case["mode"] == 0 else 16)
    lib = load(root, package["simt"])
    query, run = bind_simt_compute(lib)
    r, graph = Resources(sdk), None
    try:
        act, owners, _, ids, offsets = simt_inputs(w, case, 0)
        dtype = "<f4" if case["storage"] == 1 else "<u2"
        inp = Buffer(r, act.size * np.dtype(dtype).itemsize)
        out = Buffer(r, len(owners) * (w.n + 8) * 4)
        ip = Buffer(r, ids.nbytes) if ids is not None else None
        op = Buffer(r, offsets.nbytes) if offsets is not None else None
        ps = {name: r.upload(value) if value.size else None for name, value in w.planes.items()}
        sdk.synchronize(None)
        arr = arrangement(w.q)
        cfg = SimtConfig(1 if w.q == 8 else 3, 4, 4, 4, case["split"])
        call = SimtCall(version=1, size=C.sizeof(SimtCall), qtype=w.q, n=w.n, k=w.k,
            experts=w.experts, rows=len(owners), mode=case["mode"], input_type=case["storage"],
            channels=case["channels"], topk=8 if case["mode"] == 2 else 1,
            a_row_stride=act.shape[1], a_token_stride=case["channels"] * act.shape[1],
            ids_stride=11, out_row_stride=w.n + 8, a=inp.ptr, low=ps["low"], high=ps["high"],
            units=ps["units"], ids=ip.ptr if ip else None, offsets=op.ptr if op else None,
            output=out.ptr, stream=r.stream.value)
        request = SimtCallV2(call, int(case["compute"] == "bf16"))
        size = Sizes()
        checked(query(C.byref(request), C.byref(cfg), C.byref(arr), C.byref(size)), "SIMT typed query")
        scratch = Buffer(r, size.workspace_bytes)
        request.call.workspace, request.call.workspace_bytes = scratch.ptr, size.workspace_bytes
        fn = lambda: run(C.byref(request), C.byref(cfg), C.byref(arr))
        graph = Replay(sdk, r.stream, fn, 1)
        proofs = []
        for repeat in range(repeats):
            act, owners, from_rows, ids, offsets = simt_inputs(w, case, repeat)
            physical = act if case["storage"] == 1 else bf16_float(bf16_bits(act))
            inp.upload(physical if case["storage"] == 1 else bf16_bits(physical))
            if ip:
                ip.upload(ids)
            if op:
                op.upload(offsets)
            gold, denom = w.dot(physical[from_rows, :w.k], owners, case["compute"])
            out.poison()
            checked(graph() if repeat % 2 == 0 else fn(), "SIMT typed execution")
            sdk.synchronize(r.stream)
            image = out.read("<u4", (len(owners), w.n + 8))
            if np.any(image[:, w.n:] != 0xa5a5a5a5):
                raise ValueError("SIMT wrote output stride padding")
            proof = compare(image[:, :w.n].copy().view("<f4"), gold, denom)
            proof.update(input_sha256=digest(physical), ids_sha256=digest(ids) if ids is not None else None)
            scratch.guard()
            proofs.append(proof)
        checked(graph(), "excluded SIMT warmup")
        sdk.synchronize(r.stream)
        timing = r.samples(graph, samples) if samples else []
        return dict(status="PASS", fixture=w.record(), config={name: getattr(cfg, name) for name in
            ("variant", "columns", "warps", "values", "split")}, proofs=proofs, samples_us=timing)
    finally:
        sdk.synchronize(r.stream)
        if graph:
            graph.close()
        r.close()


def outlier(root, package, sdk, case, repeats, samples):
    w = weights(14, 5120, 25600, 1)
    r, kernel = Resources(sdk), None
    try:
        kernel = TensorCore(root, selected_record(package, case), sdk, r, w, 1, 1, split=case["split"])
        act = source(1, w.k, 65)
        act[0, 5613] = np.float32(243383.484375)
        gold, denom = w.dot(act, [0], "bf16")
        kernel.upload_a(act)
        checked(kernel.run(), "Q6 outlier execution")
        sdk.synchronize(r.stream)
        got = kernel.read()
        if case["compute"] == "f16":
            nonfinite = int(np.count_nonzero(~np.isfinite(got)))
            if not nonfinite:
                raise ValueError("the real FP16 compute overflow negative did not reproduce")
            proof = dict(status="EXPECTED_RED", nonfinite=nonfinite, golden_finite=bool(np.isfinite(gold).all()))
        else:
            proof = compare(got, gold, denom)
        kernel.workspace.guard()
        return dict(status="PASS", selection=kernel.receipt, proof=proof,
            input_sha256=digest(act), index=5613, input_value=float(act[0, 5613]), fixture=w.record())
    finally:
        sdk.synchronize(r.stream)
        if kernel:
            kernel.close()
        r.close()
