"""Static CUDA SM90 evidence; not a native PPU1.7 lowering certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def inspect(text,configs,modes=1,pdl=False):
    chunks=re.split(r'Function\s*:',text)[1:]
    if len(chunks)!=configs*2*modes:raise ValueError('actual GGUF device inventory differs')
    result=[]
    for chunk in chunks:
        name,body=chunk.split('\n',1)
        if 'GgufPrepackedCollectiveMma' not in name:raise ValueError('wrong device body')
        packed=re.search(r'StageCountILi\d+EEE[A-Za-z0-9_]*?_Lb([01])EEENS_8epilogue',name)
        if modes==2 and not packed:raise ValueError('missing explicit SF/FQ body identity')
        metadata='fq' if packed and packed[1]=='1' else 'sf'
        bf16='bfloat16' in name
        required=['LDS.128','UTMALDG.','HGMMA.','HMUL2','HADD2']
        if pdl:required+=['ACQBULK','PREEXIT']
        if bf16:required+=['HMUL2.BF16_V2','HADD2.BF16_V2','.F32.BF16']
        if any(x not in body for x in required):raise ValueError('missing live RS/TMA/typed affine instruction')
        if not re.search(r'HGMMA\.[^\n]+\.F32',body):raise ValueError('FP32 MMA accumulator absent')
        result.append({'symbol':name.strip(),'dtype':'BF16' if bf16 else 'FP16','metadata':metadata,
                       'instruction_text_sha256':hashlib.sha256(body.encode()).hexdigest(),
                       'counts':{x:body.count(x) for x in [*required,'LDL.','STL.']}})
    if sum(x['dtype']=='BF16' for x in result)!=configs*modes:raise ValueError('missing dtype inventory')
    if len({x['symbol'] for x in result})!=len(result):raise ValueError('duplicate device body')
    for mode in (('sf','fq') if modes==2 else ('sf',)):
        for dtype in ('FP16','BF16'):
            if sum(x['dtype']==dtype and x['metadata']==mode for x in result)!=configs:
                raise ValueError('missing per-mode dtype inventory')
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
    configs=manifest.get('config_count',3)
    modes=manifest.get('metadata_modes',1)
    pdl=manifest.get('pdl_compiled',False)
    bodies=inspect(sass,configs,modes,pdl)
    negatives=0
    instructions=['LDS.128','UTMALDG.','HGMMA.','HMUL2.BF16_V2','HADD2.BF16_V2']
    if pdl:instructions+=['ACQBULK','PREEXIT']
    for instruction in instructions:
        try:inspect(sass.replace(instruction,'REMOVED'),configs,modes,pdl)
        except ValueError:negatives+=1
        else:raise ValueError('codegen negative remained green')
    if modes==2:
        try:inspect(sass.replace('_Lb1EEENS_8epilogue','_Lb0EEENS_8epilogue'),configs,modes,pdl)
        except ValueError:negatives+=1
        else:raise ValueError('missing packed-unit body negative remained green')
    result={'status':'PASS','scope':'CUDA_SM90_NOT_NATIVE_PPU17','library_sha256':row['sha256'],
            'pdl_compiled':pdl,
            'bodies':bodies,'negatives_rejected':negatives}
    if a.output.exists():raise ValueError('fresh codegen receipt required')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('GGUF_Q4_CODEGEN PASS bodies='+str(len(bodies))+' negatives='+str(negatives))

if __name__=='__main__':main()
