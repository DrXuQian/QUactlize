"""Product correctness-test support."""
import ctypes as C
class Config(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[
        (name,C.c_int32) for name in ('reader','variant','warps','values','columns')]
