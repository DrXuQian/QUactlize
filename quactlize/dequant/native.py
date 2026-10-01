"""ABI and useful byte counts for independently measured expansion kernels."""
import ctypes as C



class Call(C.Structure):
    _fields_ = [('version',C.c_uint32), ('size',C.c_uint32)] + [
        (n,C.c_int32) for n in ('qtype','n','k','experts','operation','config')] + [
        (n,C.c_void_p) for n in ('low','high','units')] + [
        (n,C.c_uint64) for n in ('low_bytes','high_bytes','unit_bytes')] + [
        ('output',C.c_void_p),('zero',C.c_void_p),('output_bytes',C.c_uint64),('stream',C.c_void_p)]


class TypedCall(C.Structure):
    _fields_ = [('version',C.c_uint32),('size',C.c_uint32),('call',Call),('metadata_type',C.c_int32)]


def traffic(q, n, k, experts, operation):
    if q not in range(10,15) or n<=0 or n%256 or k<=0 or k%(512 if q in (11,14) else 256) or experts<=0:
        raise ValueError('unsupported dequant geometry')
    if operation not in (0,1):raise ValueError('unknown dequant operation')
    count=n*k*experts
    low=count*[2,2,4,4,4][q-10]//8
    high=count*[0,1,0,1,2][q-10]//8
    units=count//256*[20,14,16,16,18][q-10]
    plane=count//[16,16,32,32,16][q-10]*2
    reads=units+(low+high if operation else 0)
    writes=count*2 if operation else plane*2
    return dict(low=low,high=high,units=units,plane=plane,
                output=count*2 if operation else plane,reads=reads,writes=writes,
                useful_bytes=reads+writes)
