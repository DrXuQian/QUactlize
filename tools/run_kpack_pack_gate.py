#!/usr/bin/env python3
"""Check the GPU packer's exact bytes and event-ordered pinned backcopy."""

import argparse
import ctypes as C
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference import gguf_kpack as ref
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked


class Arrangement(C.Structure):
    _fields_ = [(n, C.c_int32) for n in (
        "version", "layout", "bits", "high_bits", "artifact_tile_k",
        "transport_tile_k", "group_size", "reserved")] + [("mapping_id", C.c_uint64)]


class Sizes(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in (
        "raw_bytes", "low_bytes", "high_bytes", "units_bytes")]


def bind(sdk, library):
    functions = {
        "hggcHostAlloc": [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint],
        "hggcFreeHost": [C.c_void_p],
        "hggcStreamCreateWithFlags": [C.POINTER(C.c_void_p), C.c_uint],
        "hggcStreamDestroy": [C.c_void_p],
        "hggcStreamWaitEvent": [C.c_void_p, C.c_void_p, C.c_uint],
        "hggcEventCreate": [C.POINTER(C.c_void_p)],
        "hggcEventRecord": [C.c_void_p, C.c_void_p],
        "hggcEventSynchronize": [C.c_void_p],
        "hggcEventDestroy": [C.c_void_p],
        "hggcEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
        "hggcMemcpyAsync": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p],
        "hggcMemsetAsync": [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p],
    }
    for name, args in functions.items():
        fn = getattr(sdk.lib, name)
        fn.argtypes, fn.restype = args, C.c_int
    query = library.quactlize_ppu_kpack_sizes_for_arrangement_v1
    query.argtypes, query.restype = [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)], C.c_int
    pack = library.quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2
    pack.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(Arrangement), C.c_void_p]
    pack.restype = C.c_int
    return query, pack


def device_identity(sdk):
    bindings = {
        "hggcGetDeviceCount": [C.POINTER(C.c_int)],
        "hggcGetDevice": [C.POINTER(C.c_int)],
        "hggcDeviceGetPCIBusId": [C.c_char_p, C.c_int, C.c_int],
    }
    for name, args in bindings.items():
        fn = getattr(sdk.lib, name)
        fn.argtypes, fn.restype = args, C.c_int
    count, ordinal = C.c_int(), C.c_int()
    checked(sdk.lib.hggcGetDeviceCount(C.byref(count)), "visible devices")
    if count.value != 1:
        raise ValueError(f"expected one visible PPU, got {count.value}; set CUDA_VISIBLE_DEVICES to one device")
    checked(sdk.lib.hggcGetDevice(C.byref(ordinal)), "current device")
    pci = C.create_string_buffer(64)
    checked(sdk.lib.hggcDeviceGetPCIBusId(pci, len(pci), ordinal.value), "device PCI identity")
    return dict(ordinal=ordinal.value, pci=pci.value.decode(),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def run_case(sdk, query, pack, q, n, k, experts):
    spec = ref.SPECS[q]
    raw = np.random.default_rng(90200 + q).integers(
        0, 256, (experts * n * (k // 256), spec.raw_bytes), dtype=np.uint8)
    started = time.monotonic()
    artifact = ref.prepare_grouped(torch.from_numpy(raw), n, k, q, experts)
    expected = b"".join(x.numpy().tobytes() for x in (artifact.low, artifact.high, artifact.units))
    reference_seconds = time.monotonic() - started
    assert len(expected) == raw.nbytes
    arr, sizes = Arrangement(**asdict(artifact.arrangement)), Sizes()
    checked(query(n, k, experts, q, C.byref(arr), C.byref(sizes)), "pack sizes")
    assert (sizes.raw_bytes, sizes.low_bytes, sizes.high_bytes, sizes.units_bytes) == (
        raw.nbytes, artifact.low.numel(), artifact.high.numel(), artifact.units.numel())
    streams, events, pinned, allocations = [], [], [], []

    def stream():
        p = C.c_void_p()
        checked(sdk.lib.hggcStreamCreateWithFlags(C.byref(p), 1), "nonblocking stream")
        streams.append(p)
        return p

    def event():
        p = C.c_void_p()
        checked(sdk.lib.hggcEventCreate(C.byref(p)), "event")
        events.append(p)
        return p

    def host(size):
        p = C.c_void_p()
        checked(sdk.lib.hggcHostAlloc(C.byref(p), size, 0), "pinned host allocation")
        pinned.append(p)
        return p

    def device(size):
        p = sdk.allocate(size)
        allocations.append(p)
        return p

    def record(e, s):
        checked(sdk.lib.hggcEventRecord(e, s), "event record")

    def elapsed(a, b):
        value = C.c_float()
        checked(sdk.lib.hggcEventElapsedTime(C.byref(value), a, b), "event elapsed")
        return value.value * 1000

    try:
        compute, copy = stream(), stream()
        upload_start, upload_done, ready, copy_start, copy_done = [event() for _ in range(5)]
        hraw, hout = host(raw.nbytes), host(len(expected) + 32)
        C.memmove(hraw, raw.ctypes.data, raw.nbytes)
        raw_dev, output = device(raw.nbytes), device(len(expected) + 32)
        checked(sdk.lib.hggcMemsetAsync(output, 0xA5, len(expected) + 32, compute),
                "output poison on pack stream")
        low = output + 16
        high = low + sizes.low_bytes if sizes.high_bytes else None
        units = low + sizes.low_bytes + sizes.high_bytes

        def launch():
            checked(pack(raw_dev, low, high, units, n, k, experts, q, C.byref(arr), compute), "device pack")

        record(upload_start, compute)
        checked(sdk.lib.hggcMemcpyAsync(raw_dev, hraw, raw.nbytes, 1, compute), "raw H2D")
        record(upload_done, compute)
        launch()
        record(ready, compute)
        # This is a copy-stream dependency, not a CPU or device-wide wait.
        checked(sdk.lib.hggcStreamWaitEvent(copy, ready, 0), "backcopy waits for packed bytes")
        record(copy_start, copy)
        checked(sdk.lib.hggcMemcpyAsync(hout, output, len(expected) + 32, 2, copy), "pinned D2H")
        record(copy_done, copy)
        checked(sdk.lib.hggcEventSynchronize(copy_done), "backcopy completion for host comparison")
        got = C.string_at(hout, len(expected) + 32)
        assert got[:16] == got[-16:] == bytes([0xA5]) * 16, "device output guard overwritten"
        bad = sum(a != b for a, b in zip(got[16:-16], expected))
        assert bad == 0, f"q={q}: {bad} packed bytes disagree with Python reference"
        # Refuse overlapping/in-place conversion before launching anything.
        assert pack(raw_dev, raw_dev, high, units, n, k, experts, q, C.byref(arr), compute) == 30
        bad_arr = Arrangement(**asdict(artifact.arrangement))
        bad_arr.mapping_id ^= 1
        assert pack(raw_dev, low, high, units, n, k, experts, q, C.byref(bad_arr), compute) == 38

        start, stop = event(), event()
        samples = []
        for _ in range(5):
            record(start, compute)
            launch()
            record(stop, compute)
            checked(sdk.lib.hggcEventSynchronize(stop), "pack timing completion")
            samples.append(elapsed(start, stop))
        # The same outputs must survive all repeated calls, not just call one.
        assert sdk.download(output, len(expected) + 32) == got
        assert all(np.isfinite(t) and t > 0 for t in samples)
        return dict(qtype=q, n=n, k=k, experts=experts, raw_bytes=raw.nbytes,
                    byte_mismatches=bad, guards_pass=True, repeated_raw_equal=True,
                    overlap_rejected=True, descriptor_rejected=True,
                    reference_seconds=reference_seconds, raw_h2d_us=elapsed(upload_start, upload_done),
                    first_pack_us=elapsed(upload_done, ready), pinned_d2h_us=elapsed(copy_start, copy_done),
                    pack_samples_us=samples, pack_median_us=statistics.median(samples),
                    backcopy_dependency="SEPARATE_NONBLOCKING_STREAM_WAITS_ON_READY_EVENT")
    finally:
        for s in streams:
            sdk.synchronize(s)
        for p in reversed(allocations):
            sdk.free(p)
        for p in reversed(pinned):
            checked(sdk.lib.hggcFreeHost(p), "free pinned host memory")
        for e in reversed(events):
            checked(sdk.lib.hggcEventDestroy(e), "destroy event")
        for s in reversed(streams):
            checked(sdk.lib.hggcStreamDestroy(s), "destroy stream")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="fresh output directory")
    p.add_argument("--real-anchor", action="store_true", help="also check five N1024 K5120 dense tensors")
    args = p.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    library = args.bundle / manifest["library"]
    if manifest["schema"] != "quactlize.kpack-device-pack-build.v1" or sha(library) != manifest["sha256"]:
        raise ValueError("packing library identity differs")
    if any(sha(ROOT / name) != value for name, value in manifest["source_hashes"].items()):
        raise ValueError("packing source identity differs")
    if any(sha(args.sdk / "lib" / name) != value for name, value in manifest["runtime"].items()):
        raise ValueError("packing runtime differs; build this small producer with the target SDK")
    args.output.mkdir(parents=True, exist_ok=False)
    sdk = SDK(args.sdk)
    identity = device_identity(sdk)
    print("KPACK_PACK_DEVICE " + json.dumps(identity), flush=True)
    loaded = C.CDLL(str(library.resolve()), mode=C.RTLD_LOCAL)
    query, pack = bind(sdk, loaded)
    cases = [(q, 256, 512, e) for q in range(10, 15) for e in (1, 3)]
    if args.real_anchor:
        cases += [(q, 1024, 5120, 1) for q in range(10, 15)]
    values = []
    started = time.monotonic()
    summary = dict(status="INCOMPLETE", expected=len(cases), completed=0,
                   library_sha256=manifest["sha256"], device=identity,
                   overlap_speedup="NOT_MEASURED", llama_cpp_validated=False)
    try:
        for q, n, k, e in cases:
            value = run_case(sdk, query, pack, q, n, k, e)
            values.append(value)
            (args.output / f"q{q}-n{n}-k{k}-e{e}.json").write_text(json.dumps(value, indent=2) + "\n")
            print(f"KPACK_PACK_CASE q={q} n={n} k={k} experts={e} bytes={value['raw_bytes']} bad=0 pack_us={value['pack_median_us']:.3f} h2d_us={value['raw_h2d_us']:.3f} d2h_us={value['pinned_d2h_us']:.3f} status=PASS", flush=True)
        summary["status"] = "PASS"
    finally:
        summary.update(completed=len(values), wall_seconds=time.monotonic() - started, cases=values)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("KPACK_PACK_GATE " + json.dumps({k: v for k, v in summary.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
