#!/usr/bin/env python3
"""Read-only identity check of the actual typed/control parent DSOs; no kernels."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.native import SDK
from quactlize.runtime.compiler import sha


def probe(sdk,bundle,manifest=None):
    m=manifest or json.loads((bundle/'manifest.json').read_text())
    attribute=sdk.lib.hggcDeviceGetAttribute
    attribute.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_int];attribute.restype=C.c_int
    rows=[]
    for typed in (False,True):
        record=next((r for r in m['modules'] if bool(r['identity'].get('endpoints'))==typed),None)
        if not record:raise ValueError('identity control module missing')
        path=(bundle/record['path']).resolve(strict=True)
        if not path.is_relative_to(bundle.resolve()) or sha(path)!=record['sha256']:
            raise ValueError('probe module bytes differ from manifest')
        lib=C.CDLL(str(path),mode=C.RTLD_LOCAL)
        name='quactlize_kpack_decode_dense_device_v1' if typed else 'quactlize_kpack_device_v1'
        fn=getattr(lib,name)
        fn.argtypes=[C.c_char_p,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_int)];fn.restype=C.c_int
        device=C.c_int(-1);count=C.c_int(-1);text=C.create_string_buffer(128);actual=C.c_int(-1)
        rc=fn(text,len(text),C.byref(device),C.byref(count))
        # SDK driver_types.h: hggcDevAttrMultiProcessorCount = 16.
        attr_rc=attribute(C.byref(actual),16,device.value) if device.value>=0 else -1
        row=dict(kind='typed' if typed else 'control',probe_rc=rc,name=text.value.decode(errors='replace'),
                 ordinal=device.value,reported_cu=count.value,attribute_rc=attr_rc,attribute_cu=actual.value,
                 parent=record['parent']['symbol'],build_key=record['key'])
        rows.append(row);print('KPACK_DECODE_DEVICE '+json.dumps(row),flush=True)
    valid=all(r['probe_rc']==0 and r['attribute_rc']==0 and r['name']=='PPU-ZW810' and
              r['reported_cu']==r['attribute_cu']==72 for r in rows)
    valid=valid and rows[0]['ordinal']==rows[1]['ordinal']
    print('KPACK_DECODE_DEVICE verdict='+('PASS' if valid else 'IDENTITY_MISMATCH_NOT_NUMERICAL'),flush=True)
    return rows,valid


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    _,ok=probe(SDK(a.sdk),a.bundle)
    raise SystemExit(0 if ok else 1)
