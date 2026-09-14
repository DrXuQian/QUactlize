#!/usr/bin/env python3
"""Decode storage/graph gate. No tuning, prefill change, or model speed verdict."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dispatch.native import Dispatch, Request, receipt
from quactlize.execution.native import arrangement
from quactlize.runtime.native import SDK, Call, checked
from quactlize.runtime.compiler import sha
from tools.kpack_warmup_fixture import Weights
from tools.run_kpack_gemv_gate import Resources, metadata_oracle
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity
from tools.run_q8_kpack2_gate import fixture as q8_fixture
from tools.verify_kpack_dispatch import verify


def bf16_bits(values):
    bits = np.asarray(values, dtype="<f4").view("<u4")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")


def bf16_float(bits):
    return (np.asarray(bits, dtype="<u2").astype("<u4") << 16).view("<f4")


def fixture(q, n, k):
    if q != 8:
        w = Weights(q, n, k, 1)
        scale, zero = metadata_oracle(w.planes["units"], q, n, k, 1)
        w.planes.update(scale=scale, zero=zero)
        return w
    _, low, scale, weight = q8_fixture(n, k, 1)
    categories = np.random.default_rng(881).integers(0, 4, k)
    sums = np.stack([weight[0, :, categories == g].sum(axis=0, dtype="f8") for g in range(4)])
    abs_sums = np.stack([np.abs(weight[0, :, categories == g]).sum(axis=0, dtype="f8") for g in range(4)])
    return SimpleNamespace(q=q, n=n, k=k, categories=categories[None, :], sums=sums[None, ...],
                           abs_sums=abs_sums[None, ...], planes=dict(low=low, high=np.empty(0, "u1"),
                           units=scale, scale=scale, zero=np.empty(0, "u1")))


def output_check(got, gold, denom):
    error = float(np.max(np.abs(got.astype("f8") - gold) / np.maximum(denom, 1e-30)))
    if not np.isfinite(got).all() or not np.isfinite(error) or error >= 0.005:
        raise ValueError(f"independent GGUF dot differs: {error:.8g}")
    return error


def dense_case(sdk, bundle, w, point, storage):
    q, route, m, n, k, e, maximum = point["request"]
    arr = arrangement(q)
    request = Request(1, C.sizeof(Request), q, route, m, n, k, e, maximum, arr.mapping_id)
    d, r = Dispatch(bundle), Resources(sdk)
    graph = None
    try:
        choice = d.query_dense_io(request, storage, bool(point["decode_policy"]))
        if point["status"] != "SELECTED":
            if choice is not None:
                raise ValueError("device dispatcher invented a host-policy choice")
            return dict(status="POLICY_MISS", request=point["request"], decode_policy=point["decode_policy"])
        if choice is None or choice.parent.decode() != point["parent"] or choice.split != point["split"]:
            raise ValueError("typed endpoint changed or lost the selected parent/split")
        second = d.query_dense_io(request, storage, bool(point["decode_policy"]))
        if bytes(choice) != bytes(second):
            raise ValueError("typed cached ticket changed")
        scalar_bytes = 4 if storage == 1 else 2
        planes = {key: r.upload(w.planes[key]) if w.planes[key].size else None
                  for key in ("low", "high", "units", "scale", "zero")}
        a = r.alloc(m * k * scalar_bytes)
        out = r.alloc(m * n * scalar_bytes + 32)
        workspace = r.alloc(choice.workspace_bytes + 32)
        r.fill(workspace, 0xa5, choice.workspace_bytes + 32)
        c = Call(version=1, size=C.sizeof(Call), m=m, n=n, k=k, experts=1,
                 group_size=arr.group_size, device=choice.device, compute_units=choice.compute_units,
                 mapping_id=arr.mapping_id, a=a, low=planes["low"], high=planes["high"],
                 metadata=planes["scale" if route == 1 else "units"],
                 zero=planes["zero"] if route == 1 and q != 8 else None,
                 output=out + 16, workspace=workspace + 16 if choice.workspace_bytes else None,
                 workspace_bytes=choice.workspace_bytes, stream=r.stream.value)
        launch = d.prepare_dense_io(choice, c, storage)
        sdk.synchronize(None)
        graph = Replay(sdk, r.stream, launch, 1)
        get_nodes = sdk.lib.hggcGraphGetNodes
        get_nodes.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_size_t)]
        get_nodes.restype = C.c_int
        count = C.c_size_t()
        checked(get_nodes(graph.graph, None, C.byref(count)), "graph node count")
        expected_nodes = 1 if choice.split == 1 else 2
        if count.value != expected_nodes:
            raise ValueError(f"unexpected decode adapters: graph nodes={count.value}, expected={expected_nodes}")
        errors = []
        for replay in range(7):
            values = np.random.default_rng(4800 + replay).uniform(-.13, .13, (m, 4)).astype("f4")
            physical = values if storage == 1 else bf16_float(bf16_bits(values))
            rounded = physical.astype("f2").astype("f8")
            source = np.ascontiguousarray(physical[:, w.categories[0]])
            upload = source if storage == 1 else bf16_bits(source)
            gold = rounded @ w.sums[0]
            denom = np.abs(rounded) @ w.abs_sums[0]
            checked(sdk.lib.hggcMemcpy(a, upload.ctypes.data, upload.nbytes, 1), "typed input upload")
            sdk.synchronize(None)
            r.fill(out, 0xa5, m * n * scalar_bytes + 32)
            checked(graph() if replay % 2 == 0 else launch(), "changed-input decode")
            sdk.synchronize(r.stream)
            image = sdk.download(out, m * n * scalar_bytes + 32)
            if image[:16] != b"\xa5" * 16 or image[-16:] != b"\xa5" * 16:
                raise ValueError("typed output guard changed")
            got = np.frombuffer(image[16:-16], dtype="<f4" if storage == 1 else "<u2").reshape(m, n)
            if storage == 2:
                got = bf16_float(got)
            errors.append(output_check(got, gold, denom))
        r.fill(a, 0, m * k * scalar_bytes)
        checked(graph(), "zero-A negative")
        sdk.synchronize(r.stream)
        image = sdk.download(out + 16, m * n * scalar_bytes)
        planted = np.frombuffer(image, dtype="<f4" if storage == 1 else "<u2").reshape(m, n)
        if storage == 2:
            planted = bf16_float(planted)
        if not np.isfinite(planted).all() or np.any(planted != 0):
            raise ValueError("zero A did not produce zero output")
        try:
            output_check(planted, gold, denom)
        except ValueError:
            pass
        else:
            raise ValueError("GGUF oracle missed the planted zero A")
        checked(sdk.lib.hggcMemcpy(a, upload.ctypes.data, upload.nbytes, 1), "restore A")
        sdk.synchronize(None)
        checked(graph(), "excluded warmup")
        sdk.synchronize(r.stream)
        timing = r.samples(graph, 3)
        half_control = None
        if storage == 1 and point["decode_policy"] == 0 and point["request"] == [12, 0, 1, 1024, 5120, 1, 1]:
            old_choice = d.query(q, route, m, n, k, e, maximum, arr.mapping_id)
            if old_choice is None:
                raise ValueError("original FP16 control was not packaged")
            old = Call.from_buffer_copy(c)
            old.a = r.upload(source.astype("<f2"))
            old.output = r.alloc(m * n * 2)
            old.workspace = r.alloc(max(16, old_choice.workspace_bytes))
            old.workspace_bytes = old_choice.workspace_bytes
            old_launch = d.prepare(old_choice, old)
            sdk.synchronize(None)
            checked(old_launch(), "original FP16 ABI control")
            sdk.synchronize(r.stream)
            old_got = np.frombuffer(sdk.download(old.output, m * n * 2), dtype="<f2").reshape(m, n)
            half_control = dict(status="PASS", error=output_check(old_got, gold, denom), choice=receipt(old_choice))
        image = sdk.download(workspace, choice.workspace_bytes + 32)
        if image[:16] != b"\xa5" * 16 or image[-16:] != b"\xa5" * 16:
            raise ValueError("typed workspace guard changed")
        result = dict(status="PASS", request=point["request"], storage="F32" if storage == 1 else "BF16",
                      decode_policy=point["decode_policy"], choice=receipt(choice), errors=errors,
                      graph_nodes=count.value, standalone_adapters=0, input_output_guards="PASS",
                      original_fp16_control=half_control,
                      zero_a_negative="RED", samples_us=timing, median_us=statistics.median(timing),
                      timing_scope="WARM_FUNCTIONAL_DIAGNOSTIC_NOT_MODEL_SPEED", first_launch_excluded=True)
        print("KPACK_DECODE_IO " + json.dumps(result), flush=True)
        return result
    finally:
        sdk.synchronize(r.stream)
        if graph:
            graph.close()
        d.close()
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pack-library", type=Path, required=True)
    args = p.parse_args()
    m = verify(args.bundle, sdk=args.sdk)
    from tools.run_kpack_moe_gate import chain_case, load_pack_library
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        library, pack_identity = load_pack_library(args.pack_library)
    except (OSError, ValueError, RuntimeError) as error:
        traceback.print_exc()
        failure = dict(status='INFRASTRUCTURE_FAIL', phase='pack-library',
                       numerical_cases_started=0, error=str(error))
        (args.output / 'summary.json').write_text(json.dumps(failure, indent=2) + '\n')
        print('KPACK_DECODE_IO_COMPLETE status=INFRASTRUCTURE_FAIL phase=pack-library numerical_cases_started=0', flush=True)
        return 2
    sdk = SDK(args.sdk)
    graph_bind(sdk)
    results = dict(device=device_identity(sdk), pack_library=pack_identity,
                   dense=[], chains=[], failures=[], source=m["jit_source_contract"])
    from tools.probe_kpack_decode_device import probe
    results['device_probes'], identity_ok = probe(sdk, args.bundle, m)
    if not identity_ok:
        results.update(status='INFRASTRUCTURE_FAIL', phase='device-identity', numerical_cases_started=0)
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print('KPACK_DECODE_IO_COMPLETE status=INFRASTRUCTURE_FAIL phase=device-identity numerical_cases_started=0', flush=True)
        return 2
    fixtures = {}
    for index, point in enumerate(m["decode_io_gate"]["requests"]):
        q, _, _, n, k, _, _ = point["request"]
        key = q, n, k
        try:
            if key not in fixtures:
                fixtures.clear()
                fixtures[key] = fixture(*key)
            for storage in (1, 2):
                results["dense"].append(dense_case(sdk, args.bundle, fixtures[key], point, storage))
        except Exception as error:
            traceback.print_exc()
            results["failures"].append(dict(request=point, error=str(error)))
        print(f"KPACK_DECODE_IO_PROGRESS completed={index+1}/{len(m['decode_io_gate']['requests'])} failures={len(results['failures'])}", flush=True)
    options = SimpleNamespace(bundle=args.bundle, sdk=args.sdk, jit_cache=None, samples=3)
    for merged in (False, True):
        for router in (False, True):
            try:
                results["chains"].append(chain_case(options, sdk, library, merged, 8, router))
            except Exception as error:
                traceback.print_exc()
                results["failures"].append(dict(merged=merged, router=router, error=str(error)))
    results["status"] = "FAIL" if results["failures"] else "PASS"
    results["model_speed_admission"] = "PENDING"
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"KPACK_DECODE_IO_COMPLETE status={results['status']} model_speed_admission=PENDING", flush=True)
    return int(bool(results["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
