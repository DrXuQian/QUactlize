"""Complete public MoE ABI bindings for correctness tests, not a selector."""
import ctypes as C
from ci.support.dense import Arrangement, Sizes


class Params(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[(n,C.c_int32) for n in
        ('qtype','n','k','experts','tokens','topk','channels','reserved')]+[(n,C.c_int64) for n in
        ('input_row_stride','input_token_stride','ids_stride','output_stride')]+[
        ('arrangement',Arrangement),('sizes',Sizes)]


class Info(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('workspace_bytes',C.c_uint64)]+[
        (n,C.c_int32) for n in ('route','policy','split','compute_type','source_tokens',
                              'prepare_algorithm','prepare_helper','reserved')]+[('config',C.c_char*192)]


class Public:
    def __init__(self,library):
        def bind(suffix,args,result=C.c_int):
            fn=getattr(library,'quactlize_ppu_moe_matmul_'+suffix)
            fn.argtypes,fn.restype=args,result
            return fn
        self.supported=bind('supported_v1',[C.POINTER(Params)])
        self.create=bind('create_v1',[C.c_int,C.POINTER(C.c_void_p)])
        self.prepare=bind('prepare_v1',[C.c_void_p,C.POINTER(Params)]+[C.c_void_p]*3+
                          [C.POINTER(C.c_void_p),C.POINTER(Info)])
        self.run=bind('run_v1',[C.c_void_p]*5+[C.c_uint64,C.c_void_p])
        self.destroy=bind('destroy_v1',[C.c_void_p],None)
        self.error=bind('error_v1',[],C.c_char_p)

    def check(self,rc):
        if rc:raise AssertionError(f'public MoE rc={rc}: {self.error().decode(errors="replace")}')
