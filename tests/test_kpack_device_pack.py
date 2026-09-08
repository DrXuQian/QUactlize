"""Host execution of the device packer's exact word/metadata ownership."""

import ctypes as C
from dataclasses import asdict
from pathlib import Path
import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from tools.run_kpack_pack_gate import Arrangement, Sizes, device_identity, run_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "g++ is required for the host ownership proof"
    sdk = Path(os.environ.get("PPU_SDK", "/root/ppu-sdk/2.1.1"))
    if not (sdk / "include/hggc_runtime.h").is_file():
        pytest.skip("PPU SDK headers are required; no PPU device is needed")
    output = tmp_path_factory.mktemp("kpack-word-proof") / "host.so"
    compiled = subprocess.run([
        compiler, "-std=c++17", "-O2", "-shared", "-fPIC",
        f"-I{ROOT / 'quactlize/packing'}", f"-I{ROOT / 'quactlize/include'}",
        f"-I{ROOT / 'third_party/actlize/include'}", f"-I{sdk / 'include'}",
        str(ROOT / "tests/kpack_pack_host.cpp"),
        str(ROOT / "quactlize/packing/sizes.cpp"), "-o", str(output),
    ], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    lib = C.CDLL(str(output))
    lib.host_pack.argtypes = [C.c_int, *([C.c_void_p] * 4), C.c_int, C.c_int, C.c_int]
    lib.host_pack.restype = C.c_int
    lib.quactlize_ppu_kpack_sizes_for_arrangement_v1.argtypes = [
        C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(Arrangement), C.POINTER(Sizes)]
    lib.quactlize_ppu_kpack_sizes_for_arrangement_v1.restype = C.c_int
    return lib


def arrangement(q):
    return Arrangement(**asdict(ref.canonical_arrangement(q)))


@pytest.mark.parametrize("visible", [0, 1, 2, 8])
def test_box_gate_requires_one_physical_device(visible, monkeypatch):
    def count(out):
        out._obj.value = visible
        return 0

    def current(out):
        out._obj.value = 0
        return 0

    def pci(out, size, ordinal):
        out.value = b"0000:08:00.0"
        return 0

    sdk = SimpleNamespace(lib=SimpleNamespace(
        hggcGetDeviceCount=count, hggcGetDevice=current, hggcDeviceGetPCIBusId=pci))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    if visible == 1:
        assert device_identity(sdk) == dict(ordinal=0, pci="0000:08:00.0", visible_devices="3")
    else:
        with pytest.raises(ValueError, match="expected one visible PPU"):
            device_identity(sdk)


@pytest.mark.parametrize("q", range(10, 15))
@pytest.mark.parametrize("experts,k", [(1, 512), (3, 1024)])
def test_words_and_metadata_match_independent_reference(library, q, experts, k):
    n = 256
    spec = ref.SPECS[q]
    raw = np.random.default_rng(91200 + q).integers(
        0, 256, (experts * n * (k // 256), spec.raw_bytes), dtype=np.uint8)
    # Random bytes include signed scale codes, high planes and arbitrary FP16
    # headers. Packing must preserve those bits without doing half arithmetic.
    expected = ref.prepare_grouped(torch.from_numpy(raw), n, k, q, experts)
    sizes = Sizes()
    assert library.quactlize_ppu_kpack_sizes_for_arrangement_v1(
        n, k, experts, q, C.byref(arrangement(q)), C.byref(sizes)) == 0
    assert sizes.raw_bytes == raw.nbytes
    assert sizes.low_bytes + sizes.high_bytes + sizes.units_bytes == raw.nbytes
    guarded = [np.full(x.numel() + 32, 0xA5, dtype=np.uint8)
               for x in (expected.low, expected.high, expected.units)]
    views = [x[16:-16] for x in guarded]
    assert library.host_pack(q, raw.ctypes.data, views[0].ctypes.data,
                            views[1].ctypes.data if views[1].size else None,
                            views[2].ctypes.data, n, k, experts) == 0
    for storage, got, want in zip(guarded, views, (expected.low, expected.high, expected.units)):
        assert np.array_equal(got, want.reshape(-1).numpy())
        assert np.all(storage[:16] == 0xA5) and np.all(storage[-16:] == 0xA5)
    assert torch.equal(ref.recover_raw_blocks(expected), torch.from_numpy(raw))


@pytest.mark.parametrize("q", range(10, 15))
def test_sizes_reject_invalid_input_without_touching_output(library, q):
    query = library.quactlize_ppu_kpack_sizes_for_arrangement_v1
    good = arrangement(q)
    for n, k, e, t, arr in (
        (0, 512, 1, q, good), (255, 512, 1, q, good),
        (256, 257, 1, q, good), (256, 512, 0, q, good),
        (256, 512, 1, 99, good), (256, 512, 1, q, None),
        (2147483392, 2147483136, 2147483647, q, good),
    ):
        out = Sizes(11, 22, 33, 44)
        assert query(n, k, e, t, C.byref(arr) if arr else None, C.byref(out)) != 0
        assert bytes(out) == bytes(Sizes(11, 22, 33, 44))
    for field, _ in Arrangement._fields_:
        bad = arrangement(q)
        setattr(bad, field, getattr(bad, field) ^ 1)
        out = Sizes(11, 22, 33, 44)
        assert query(256, 512, 1, q, C.byref(bad), C.byref(out)) == 38
        assert bytes(out) == bytes(Sizes(11, 22, 33, 44))
    if q in (11, 14):
        assert query(256, 256, 1, q, C.byref(good), C.byref(Sizes())) == 24


class HostSDK:
    """Host memory/event double; never counted as device validation."""
    def __init__(self):
        self.memory, self.trace, self.serial = {}, [], 0
        self.lib = SimpleNamespace(
            hggcStreamCreateWithFlags=self.create, hggcEventCreate=self.create,
            hggcEventRecord=lambda *a: self.note("record"),
            hggcStreamWaitEvent=lambda *a: self.note("wait-ready"),
            hggcEventSynchronize=lambda *a: self.note("host-wait"),
            hggcMemcpyAsync=self.copy, hggcHostAlloc=self.host_allocate,
            hggcMemsetAsync=self.fill_async,
            hggcFreeHost=self.free, hggcEventDestroy=lambda *a: 0,
            hggcStreamDestroy=lambda *a: 0, hggcEventElapsedTime=self.elapsed,
        )

    def note(self, name):
        self.trace.append(name)
        return 0

    def create(self, out, *args):
        self.serial += 1
        out._obj.value = self.serial
        return 0

    def allocate(self, size):
        data = C.create_string_buffer(size)
        ptr = C.addressof(data)
        self.memory[ptr] = data
        return ptr

    def host_allocate(self, out, size, flags):
        out._obj.value = self.allocate(size)
        return 0

    def free(self, ptr):
        del self.memory[ptr.value if isinstance(ptr, C.c_void_p) else ptr]
        return 0

    def copy(self, dst, src, size, direction, stream):
        self.note("D2H" if direction == 2 else "H2D")
        C.memmove(dst, src, size)
        return 0

    def fill(self, ptr, value, size):
        C.memset(ptr, value, size)

    def fill_async(self, ptr, value, size, stream):
        self.note("poison-on-pack-stream")
        self.fill(ptr, value, size)
        return 0

    def download(self, ptr, size):
        return C.string_at(ptr, size)

    def synchronize(self, stream):
        self.note("stream-cleanup-wait")

    def elapsed(self, out, start, stop):
        out._obj.value = 0.1
        return 0


@pytest.mark.parametrize("fault", [None, "code", "guard", "launch"])
def test_box_gate_event_order_and_failure_controls(library, fault):
    sdk = HostSDK()
    query = library.quactlize_ppu_kpack_sizes_for_arrangement_v1

    def pack(raw, low, high, units, n, k, experts, q, arr, stream):
        rc = query(n, k, experts, q, arr, C.byref(Sizes()))
        if rc:
            return rc
        if raw == low:
            return 30
        if fault == "launch":
            return 41
        rc = library.host_pack(q, raw, low, high, units, n, k, experts)
        if fault == "code":
            C.c_uint8.from_address(low).value ^= 1
        if fault == "guard":
            C.c_uint8.from_address(low - 1).value ^= 1
        return rc

    if fault:
        with pytest.raises((AssertionError, RuntimeError)):
            run_case(sdk, query, pack, 13, 256, 512, 1)
    else:
        result = run_case(sdk, query, pack, 13, 256, 512, 1)
        assert result["byte_mismatches"] == 0 and result["repeated_raw_equal"]
        assert result["overlap_rejected"] and result["descriptor_rejected"]
        assert sdk.trace.index("poison-on-pack-stream") < sdk.trace.index("H2D")
        assert sdk.trace.index("H2D") < sdk.trace.index("wait-ready")
        assert sdk.trace.index("wait-ready") < sdk.trace.index("D2H") < sdk.trace.index("host-wait")
        assert len(result["pack_samples_us"]) == 5
    assert not sdk.memory
