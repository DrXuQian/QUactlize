"""Call the existing llama CUDA DSO's Q8 F16 converter; never copy its kernel."""
import csv
import ctypes as C
import io
from pathlib import Path
import re
import subprocess

from quactlize.runtime.compiler import sha

GETTER = '_Z21ggml_get_to_fp16_cuda9ggml_type'


def describe(library):
    library=Path(library).resolve(strict=True)
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
    if not any(line.split()[-1:]==[GETTER] for line in symbols.splitlines()):
        raise ValueError('native llama DSO does not export ggml_get_to_fp16_cuda(ggml_type)')
    aliases={str(p):str(p.resolve()) for p in library.parent.glob('*.so*') if p.is_file()}
    images={Path(p) for p in aliases.values()}
    return dict(library=str(library),library_sha256=sha(library),entry=GETTER,qtype=8,
        libraries={str(p):sha(p) for p in sorted(images)},aliases=aliases,
        scope='EXISTING_LLAMA_BINARY_Q8_F16_DEQUANT_NO_REBUILD_NO_GEMM')


def verify(receipt):
    if receipt['entry']!=GETTER or receipt['qtype']!=8:
        raise ValueError('native Q8 converter identity differs')
    if sha(receipt['library'])!=receipt['library_sha256'] or any(
            sha(p)!=h for p,h in receipt['libraries'].items()):
        raise ValueError('native llama libraries changed; retain results and start a new run')
    if any(str(Path(p).resolve(strict=True))!=target for p,target in receipt['aliases'].items()):
        raise ValueError('native llama library symlink changed')


class NativeQ8:
    def __init__(self,receipt):
        verify(receipt)
        self.owner=C.CDLL(receipt['library'],mode=C.RTLD_LOCAL)
        query=self.owner[GETTER]
        query.argtypes=[C.c_int];query.restype=C.c_void_p
        address=query(8) # native device capability selection, before capture
        if not address:raise ValueError('native Q8 to F16 conversion unavailable')
        self.function=C.CFUNCTYPE(None,C.c_void_p,C.c_void_p,C.c_int64,C.c_void_p)(address)

    def launch(self,source,output,elements,stream):
        self.function(source,output,elements,stream)


def check_profile(text,n,k):
    from dev.native_components.profile import check_report
    start=text.find('"ID"')
    if start<0:raise ValueError('native ACU export has no kernel table')
    rows=[r for r in csv.DictReader(io.StringIO(text[start:])) if r.get('Kernel Name')]
    if len({(r['ID'],r.get('Kernel Mangled Name',r['Kernel Name'])) for r in rows})!=1:
        raise ValueError('native dequant profile must have exactly one launch')
    for row in rows:
        name=re.sub(r'\s+','',row['Kernel Name'])
        if not re.fullmatch(r'(?:void)?dequantize_block_q8_0_f16<false>\(.*\)',name):
            raise ValueError('native converter did not launch the Q8 F16 fast path: '+row['Kernel Name'])
    expected=dict(symbol=rows[0].get('Kernel Mangled Name',rows[0]['Kernel Name']),
                  name=rows[0]['Kernel Name'],grid=[n*k//2048,1,1],block=[32,1,1])
    if n*k%2048:raise ValueError('native comparison requires complete2048-element blocks')
    return check_report(text,expected)
