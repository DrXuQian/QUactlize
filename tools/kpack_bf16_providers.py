"""Installed BF16 providers; no local GEMM implementation or timing fallback."""
import ctypes as C
import importlib
from pathlib import Path

from quactlize.runtime.compiler import sha
from quactlize.runtime.native import checked


class Cublas:
    """cuBLAS ABI from the PPU SDK (the installed PPU BLAS compatibility layer)."""
    def __init__(self, sdk, stream):
        self.path = (Path(sdk) / 'CUDA_SDK/targets/x86_64-linux/lib/libcublas.so').resolve(strict=True)
        self.lib = C.CDLL(str(self.path), mode=C.RTLD_LOCAL)
        self.handle = C.c_void_p()
        signatures = {
            'cublasCreate_v2': [C.POINTER(C.c_void_p)],
            'cublasDestroy_v2': [C.c_void_p],
            'cublasSetStream_v2': [C.c_void_p, C.c_void_p],
            'cublasSetMathMode': [C.c_void_p, C.c_int],
            'cublasGetVersion_v2': [C.c_void_p, C.POINTER(C.c_int)],
            'cublasGemmEx': [C.c_void_p] + [C.c_int]*5 + [C.c_void_p, C.c_void_p, C.c_int, C.c_int,
                C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes, fn.restype = args, C.c_int
        checked(self.lib.cublasCreate_v2(C.byref(self.handle)), 'cuBLAS create')
        try:
            checked(self.lib.cublasSetStream_v2(self.handle, stream), 'cuBLAS stream')
            # BF16 operands, FP32 compute, no reduced-precision reduction.
            checked(self.lib.cublasSetMathMode(self.handle, 16), 'cuBLAS reduction precision')
            version = C.c_int()
            checked(self.lib.cublasGetVersion_v2(self.handle, C.byref(version)), 'cuBLAS version')
        except BaseException:
            self.close(); raise
        self.alpha, self.beta = C.c_float(1), C.c_float(0)
        self.identity = dict(provider='CUBLAS_PPU_SDK', entry='cublasGemmEx', version=version.value,
            library=str(self.path), sha256=sha(self.path), a='BF16_M_K', b='BF16_N_K', output='BF16_M_N',
            compute='CUBLAS_COMPUTE_32F', math_mode=16, algorithm='CUBLAS_GEMM_DEFAULT')

    def __call__(self, a, b, out, indices=None, rows=None):
        # Row-major C = A B^T -> column-major C^T = B A^T.
        m, k = a.shape; n = b.shape[-2]
        checked(self.lib.cublasGemmEx(self.handle, 1, 0, n, m, k, C.byref(self.alpha),
            b.data_ptr(), 14, k, a.data_ptr(), 14, k, C.byref(self.beta), out.data_ptr(), 14, n, 68, -1),
            'cuBLAS BF16 GEMM')

    def close(self):
        if self.handle:
            checked(self.lib.cublasDestroy_v2(self.handle), 'cuBLAS destroy')
            self.handle = C.c_void_p()


class DeepGemm:
    def __init__(self):
        module = importlib.import_module('deep_gemm')
        name = 'm_grouped_gemm_bf16_bf16_bf16_nt_nopad'
        if not hasattr(module, name):
            raise ValueError('installed DeepGEMM lacks the PPU BF16 no-padding grouped entry: '+name)
        self.fn = getattr(module, name)
        root = Path(module.__file__).resolve().parent
        files = {str(p.relative_to(root)): sha(p) for p in root.rglob('*')
                 if p.is_file() and p.suffix in ('.py', '.so', '.hpp', '.h', '.cuh')}
        self.identity = dict(provider='DEEPGEMM_INSTALLED', entry=name, root=str(root), files=files,
            a='BF16_SORTED_ROWS_K', b='BF16_E_N_K', output='BF16_SORTED_ROWS_N',
            selection='PROVIDER_DEFAULT_NO_EXTERNAL_TACTIC_OVERRIDE',
            scope='PROVIDER_CALL_INCLUDING_INTERNAL_BLOCK_DIRECTORY_NO_EXTERNAL_ROUTING')

    def __call__(self, a, b, out, indices, rows):
        # m_rows is the resident common row-count input. Do not time a second
        # bincount; the provider's own necessary block directory stays inside.
        self.fn(a, b, out, indices, m_rows=rows)

    def close(self):
        pass


def loaded_images():
    """Bind the actual BLAS/JIT images, including dynamically loaded backends."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if path.is_file() and any(x in str(path) for x in ('cublas', 'acblas', 'deep_gemm')):
                paths.add(path)
    return {str(p): sha(p) for p in sorted(paths)}
