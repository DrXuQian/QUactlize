"""Static CUDA SM90 evidence; not a native PPU1.7 lowering certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def inspect(text):
    chunks=re.split(r'Function\s*:',text)[1:]
    if len(chunks)!=6:raise ValueError('expected six actual GGUF device bodies')
    result=[]
    for chunk in chunks:
        name,body=chunk.split('\n',1)
        if 'GgufPrepackedCollectiveMma' not in name:raise ValueError('wrong device body')
        bf16='bfloat16' in name
        required=['LDS.128','UTMALDG.','HGMMA.','HMUL2','HADD2']
        if bf16:required+=['HMUL2.BF16_V2','HADD2.BF16_V2','.F32.BF16']
        if any(x not in body for x in required):raise ValueError('missing live RS/TMA/typed affine instruction')
        if not re.search(r'HGMMA\.[^\n]+\.F32',body):raise ValueError('FP32 MMA accumulator absent')
        result.append({'symbol':name.strip(),'dtype':'BF16' if bf16 else 'FP16',
                       'instruction_text_sha256':hashlib.sha256(body.encode()).hexdigest(),
                       'counts':{x:body.count(x) for x in required}})
    if sum(x['dtype']=='BF16' for x in result)!=3:raise ValueError('missing dtype inventory')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--cuobjdump',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    row=manifest['builds']['adapter'];lib=a.bundle/row['file']
    if hashlib.sha256(lib.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('library differs')
    sass=subprocess.check_output([str(a.cuobjdump),'--dump-sass',str(lib)],text=True)
    bodies=inspect(sass)
    negatives=0
    for instruction in ('LDS.128','UTMALDG.','HGMMA.','HMUL2.BF16_V2','HADD2.BF16_V2'):
        try:inspect(sass.replace(instruction,'REMOVED'))
        except ValueError:negatives+=1
        else:raise ValueError('codegen negative remained green')
    result={'status':'PASS','scope':'CUDA_SM90_NOT_NATIVE_PPU17','library_sha256':row['sha256'],
            'bodies':bodies,'negatives_rejected':negatives}
    if a.output.exists():raise ValueError('fresh codegen receipt required')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('GGUF_Q4_CODEGEN PASS bodies=6 negatives='+str(negatives))

if __name__=='__main__':main()
