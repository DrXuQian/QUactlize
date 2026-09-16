"""ABI and useful byte counts for independently measured expansion kernels."""
import ctypes as C
import json
from pathlib import Path

from quactlize.execution.native import Arrangement
from quactlize.runtime.compiler import sha


def config_ids(q, operation, generation=1):
    """Stable old config IDs; vector readers are additive experiment arms."""
    if q not in range(10, 15) or operation not in (0, 1) or generation not in (1, 2, 3, 4):
        raise ValueError('invalid dequant inventory')
    if generation == 4 and operation == 1 and q in (12, 13):
        return list(range(13))
    if generation >= 3 and operation == 1 and q in (12, 13):
        return list(range(10))
    if generation >= 2 and (operation or q in (12, 13)):
        return list(range(6))
    return list(range(3 if operation else 4))


def selected_configs(q, operation, generation, inventory='all'):
    available=config_ids(q,operation,generation)
    if inventory=='all':return available
    if inventory=='full-reader' and generation==3 and operation==1 and q in (12,13):
        return [4,5,6,7,8,9]
    if inventory=='full-packed' and generation==4 and operation==1 and q in (12,13):
        return [4,5,10,11,12]
    raise ValueError('inventory does not apply to this format/stage/package')


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


def verify(bundle, sdk):
    bundle, sdk = Path(bundle).resolve(strict=True), Path(sdk).resolve(strict=True)
    m=json.loads((bundle/'manifest.json').read_text())
    if m.get('schema')!='quactlize.dequant-only.v1' or m['timing']!='NO_GEMM':
        raise ValueError('dequant manifest schema/scope differs')
    if m['library']!='libquactlize_ppu_dequant.so' or sha(bundle/m['library'])!=m['sha256']:
        raise ValueError('dequant payload differs')
    if sha(bundle/'native.json')!=m['native_sha256']:
        raise ValueError('dequant native instruction receipt differs')
    for name, value in m['runtime'].items():
        if sha(sdk/'lib'/name)!=value:raise ValueError('dequant runtime differs: '+name)
    return m


def bind(bundle):
    lib=C.CDLL(str(Path(bundle)/'libquactlize_ppu_dequant.so'),mode=C.RTLD_LOCAL)
    fn=lib.quactlize_kpack_dequant_v1
    fn.argtypes=[C.POINTER(Call),C.POINTER(Arrangement)];fn.restype=C.c_int
    probe=lib.quactlize_kpack_dequant_probe_v1
    probe.argtypes=[C.POINTER(C.c_int)]*3;probe.restype=C.c_int
    return lib,fn,probe
