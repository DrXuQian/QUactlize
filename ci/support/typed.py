"""Product correctness-test support; no performance campaign."""
import ctypes as C
from quactlize.dequant.native import Call as Weight
from quactlize.execution.native import Arrangement
from quactlize.prefill.native import Call
from quactlize.runtime.native import checked
class Options(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('sdk',C.c_char_p)]+[
        (x,C.c_int32)for x in ('storage','vector_io','dequant_variant','math_mode','algorithm')]


def bind(lib,name,args,restype=C.c_int):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=restype;return fn


class DenseLibrary:
    def __init__(self, path):
        self.lib=C.CDLL(str(path),mode=C.RTLD_LOCAL)
        self.policy=bind(self.lib,'quactlize_kpack_dense_helper_policy_v1',
            [C.POINTER(Call),C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)])
        self.full_policy=bind(self.lib,'quactlize_kpack_dense_full_helper_policy_v1',
            [C.POINTER(Call),C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)])
        self.io=bind(self.lib,'quactlize_dense_io_v1',
            [C.c_int]*3+[C.c_void_p]*2+[C.c_int]*2+[C.c_int64,C.c_void_p,C.c_void_p])
        self.query=bind(self.lib,'quactlize_kpack_dense_full_workspace_v1',
            [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(C.c_uint64)])
        self.prepare=bind(self.lib,'quactlize_kpack_dense_full_prepare_v1',
            [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(Options),C.POINTER(C.c_void_p)])
        self.run=bind(self.lib,'quactlize_kpack_dense_full_run_v1',[C.c_void_p,C.POINTER(Call),C.c_int])
        self.destroy=bind(self.lib,'quactlize_kpack_dense_full_destroy_v1',[C.c_void_p],None)
        self.error=bind(self.lib,'quactlize_kpack_dense_full_error_v1',[],C.c_char_p)

    def shape(self,w):
        c=Call();c.version=1;c.size=C.sizeof(Call);c.m=c.a_rows=w.m
        c.a_stride=w.k;c.output_stride=w.n;c.device=0
        # Legacy config remains in its own namespace; typed DQ is in Options.
        c.weight=Weight(1,C.sizeof(Weight),w.qtype,w.n,w.k,1,1,5)
        return c

    def choice(self,w,storage,allow_miss=False,*,full=False):
        c=self.shape(w);dq,io=C.c_int(-1),C.c_int(-1)
        policy=self.full_policy if full else self.policy
        rc=policy(C.byref(c),storage,C.byref(dq),C.byref(io))
        if allow_miss and rc==24: return None  # QKG_SHAPE only; runtime errors still fail.
        checked(rc,
                'production dense helper policy (requires measured ZW810 shape)')
        return dict(dequant_variant=dq.value,vector_io=io.value,storage=storage)

    def workspace(self,w,arr,*,legacy=False):
        shape=self.shape(w);size=C.c_uint64()
        query=(bind(self.lib,'quactlize_kpack_prefill_query_dynamic_v1',
                    [C.POINTER(Call),C.POINTER(Arrangement),C.POINTER(C.c_uint64)]) if legacy else self.query)
        rc=query(C.byref(shape),C.byref(arr),C.byref(size))
        if rc:
            raise RuntimeError(f'{"legacy" if legacy else "typed"} full workspace failed: status={rc} '
                               f'q={w.qtype} M={w.m} N={w.n} K={w.k}; no timing admitted')
        if not size.value or size.value%256:
            raise ValueError('invalid full workspace extent')
        return size.value
