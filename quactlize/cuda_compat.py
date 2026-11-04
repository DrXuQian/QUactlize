"""Prepared NVIDIA compatibility: GGUF/K-pack -> FP16 or BF16 -> cuBLAS.

Preparation and allocation precede capture. A call is an opaque mutating Torch
operator; captured graphs can replay it without Python. Keep the prepared call
and its tensors alive until all graphs using them have drained. Scratch can be
shared by calls that the graph orders sequentially.
"""
from __future__ import annotations

import ctypes as C
from functools import lru_cache
from pathlib import Path
import weakref

import torch


class Problem(C.Structure):
    _fields_ = [(s, C.c_uint32) for s in ("version", "size")] + [(s, C.c_int32) for s in (
        "qtype", "layout", "mode", "compute_type", "m", "n", "k", "experts", "topk", "channels", "device")]


class Sizes(C.Structure):
    _fields_ = [(s, C.c_uint64) for s in ("raw_bytes", "low_bytes", "high_bytes", "units_bytes",
        "input_bytes", "output_bytes", "ids_bytes", "expanded_weight_bytes", "expanded_input_bytes", "workspace_bytes")]


class Arrangement(C.Structure):
    _fields_ = [(s, C.c_int32) for s in ("version", "layout", "bits", "high_bits", "artifact_tile_k",
        "transport_tile_k", "group_size", "reserved")] + [("mapping_id", C.c_uint64)]


class Buffers(C.Structure):
    _fields_ = [(s, C.c_uint32) for s in ("version", "size")] + [(s, C.c_void_p) for s in (
        "input", "raw", "low", "high", "units", "ids", "output", "workspace")]
    _fields_ += [(s, C.c_uint64) for s in ("raw_bytes", "low_bytes", "high_bytes", "units_bytes",
        "input_bytes", "output_bytes", "ids_bytes", "workspace_bytes")]
    _fields_ += [("device_error", C.c_void_p), ("stream", C.c_void_p)]


@lru_cache(maxsize=1)
def _library():
    paths = sorted(Path(__file__).parent.glob("_cuda_compat*.so"))
    if len(paths) != 1:
        raise ImportError("Build one NVIDIA compatibility library with QUACTLIZE_WITH_CUDA=1 "
                          "pip install --no-build-isolation -e .")
    lib = C.CDLL(str(paths[0]))
    lib.quactlize_cuda_error_v1.restype = C.c_char_p
    lib.quactlize_cuda_query_v1.argtypes = [C.POINTER(Problem), C.c_void_p, C.POINTER(Sizes)]
    lib.quactlize_cuda_prepare_v1.argtypes = [C.POINTER(Problem), C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)]
    for name in ("quactlize_cuda_run_v1", "quactlize_cuda_dequant_v1"):
        getattr(lib, name).argtypes = [C.c_void_p, C.POINTER(Buffers)]
    lib.quactlize_cuda_destroy_v1.argtypes = [C.c_void_p]
    lib.quactlize_cuda_destroy_v1.restype = None
    return lib


def canonical_arrangement(qtype):
    """Query the unchanged canonical producer identity for Q8_0 or K-quant."""
    result = Arrangement()
    query = _library().quactlize_ppu_kpack_canonical_arrangement_v1
    query.argtypes = [C.c_int, C.POINTER(Arrangement)]
    if query(qtype, C.byref(result)):
        raise ValueError("no canonical K-pack arrangement for this format")
    return result


def _check(result):
    if result:
        raise RuntimeError(_library().quactlize_cuda_error_v1().decode())


def query(problem, arrangement=None):
    """Return byte requirements without allocating tensors or creating a handle."""
    sizes = Sizes()
    _check(_library().quactlize_cuda_query_v1(C.byref(problem),
        C.byref(arrangement) if arrangement is not None else None, C.byref(sizes)))
    return sizes


def _buffers(input, raw, low, high, units, ids, output, workspace, device_error):
    b = Buffers(version=1, size=C.sizeof(Buffers))
    for name, tensor in (("input", input), ("raw", raw), ("low", low), ("high", high),
                         ("units", units), ("ids", ids), ("output", output), ("workspace", workspace)):
        setattr(b, name, tensor.data_ptr() if tensor is not None else None)
        setattr(b, name + "_bytes", tensor.numel() * tensor.element_size() if tensor is not None else 0)
    b.device_error = device_error.data_ptr()
    b.stream = torch.cuda.current_stream(input.device).cuda_stream
    return b


@torch.library.custom_op("quactlize_cuda::matmul", mutates_args=("output", "workspace", "device_error"))
def _matmul(input: torch.Tensor, raw: torch.Tensor | None, low: torch.Tensor | None,
            high: torch.Tensor | None, units: torch.Tensor | None, ids: torch.Tensor | None,
            output: torch.Tensor, workspace: torch.Tensor, device_error: torch.Tensor, handle: int) -> None:
    with torch.cuda.device(input.device):
        b = _buffers(input, raw, low, high, units, ids, output, workspace, device_error)
        _check(_library().quactlize_cuda_run_v1(handle, C.byref(b)))


@_matmul.register_fake
def _matmul_fake(input, raw, low, high, units, ids, output, workspace, device_error, handle):
    pass


# Public opaque operation for callers that bind process-local handles after
# compiling a forward in another process. The owning PreparedMatmul stays live.
matmul = _matmul


def _destroy(lib, handle, device):
    with torch.cuda.device(device):
        lib.quactlize_cuda_destroy_v1(handle)


class PreparedMatmul:
    """Bind a dense or indexed matmul to caller-owned tensors.

    Dense input/output are [M,K]/[M,N]. Indexed IDs are int32 [M,topk]; input
    is [M,K] (broadcast) or [M,topk,K], output is [M,topk,N]. IDs can change on
    device between replays; duplicates work and -1 produces zero. Other invalid
    IDs set device_error to 1 and produce zero. Error checking is outside replay.

    Supply raw GGUF bytes, or low/high/units plus the producer's ctypes
    arrangement-v2 descriptor. No conversion between these layouts is implicit.
    Workspace includes expanded selected slots and cuBLAS scratch. Call once on
    the capture stream before capturing to warm library kernels. This is a
    format/graph compatibility path, not a fused quantized performance kernel.
    """
    def __init__(self, *, qtype, n, k, input, output, raw=None, low=None, high=None, units=None,
                 arrangement=None, ids=None, experts=1, workspace=None):
        if input.device.type != "cuda" or input.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("input must be CUDA FP16 or BF16")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("prepare before graph capture")
        if input.ndim not in (2, 3) or input.shape[-1] != k:
            raise ValueError("input shape must be [M,K] or [M,topk,K]")
        m = input.shape[0]
        topk = ids.shape[1] if ids is not None and ids.ndim == 2 else 1
        channels = input.shape[1] if input.ndim == 3 else 1
        expected = (m, topk, n) if ids is not None else (m, n)
        if tuple(output.shape) != expected or output.dtype != input.dtype:
            raise ValueError(f"output must have shape {expected} and the input dtype")
        if ids is not None and (tuple(ids.shape) != (m, topk) or ids.dtype != torch.int32):
            raise ValueError("ids must be int32 [M,topk]")
        if (raw is not None) == (arrangement is not None):
            raise ValueError("supply raw GGUF or a canonical K-pack arrangement")
        self.problem = Problem(1, C.sizeof(Problem), qtype, int(raw is None), int(ids is not None),
            int(input.dtype == torch.bfloat16), m, n, k, experts, topk, channels, input.device.index)
        self.sizes = query(self.problem, arrangement)
        lib = _library()
        descriptor = C.byref(arrangement) if arrangement is not None else None
        if workspace is None:
            workspace = torch.empty(self.sizes.workspace_bytes, dtype=torch.uint8, device=input.device)
        for name, tensor in (("input", input), ("output", output), ("raw", raw), ("low", low),
                             ("high", high), ("units", units), ("ids", ids), ("workspace", workspace)):
            needed = getattr(self.sizes, name + "_bytes")
            if tensor is None:
                if needed:
                    raise ValueError(f"{name} is required")
                continue
            if not needed or not tensor.is_contiguous() or tensor.device != input.device:
                raise ValueError(f"{name} must be a required contiguous tensor on {input.device}")
            if name not in ("input", "output", "ids") and tensor.dtype != torch.uint8:
                raise ValueError(f"{name} must be uint8 storage")
            if tensor.numel() * tensor.element_size() < needed:
                raise ValueError(f"{name} capacity is too small")
        self.device_error = torch.zeros((), dtype=torch.int32, device=input.device)
        self.args = (input, raw, low, high, units, ids, output, workspace, self.device_error)
        self.output, self.workspace = output, workspace
        self.handle = C.c_void_p()
        with torch.cuda.device(input.device):
            stream = torch.cuda.current_stream(input.device).cuda_stream
            _check(lib.quactlize_cuda_prepare_v1(C.byref(self.problem), descriptor, stream, C.byref(self.handle)))
        self._finalizer = weakref.finalize(self, _destroy, lib, self.handle, input.device)
        self.native_handle = self.handle.value
        self.closed = False
        slots = m * topk if ids is not None else 1
        self.expanded = workspace[:self.sizes.expanded_weight_bytes].view(input.dtype).view(slots, n, k)

    def __call__(self, input=None):
        if self.closed:
            raise RuntimeError("prepared matmul is closed")
        source = self.args[0] if input is None else input
        if (source.shape != self.args[0].shape or source.dtype != self.args[0].dtype
                or source.device != self.args[0].device or not source.is_contiguous()):
            raise ValueError("input differs from prepared geometry, dtype or device")
        _matmul(source, *self.args[1:], self.native_handle)

    def dequantize(self):
        """Expand into scratch without GEMM; qualification only, returns an alias."""
        if self.closed:
            raise RuntimeError("prepared matmul is closed")
        with torch.cuda.device(self.args[0].device):
            b = _buffers(*self.args)
            _check(_library().quactlize_cuda_dequant_v1(self.handle, C.byref(b)))
        return self.expanded

    def close(self):
        """Release the handle after the caller has drained all dependent work."""
        self._finalizer()
        self.closed = True
