"""Paired resident-SF / packed-unit-FQ screen on the same GGUF math and code map."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    args,common=p.parse_known_args()
    if args.output.exists():raise ValueError('fresh comparison directory required')
    args.output.mkdir(parents=True)
    runner=Path(__file__).with_name('run.py')
    fixture=runner.with_name('fixture.py')
    bundle=Path(common[common.index('--bundle')+1])
    backend=Path(common[common.index('--backend-source')+1])
    identity={'runner':sha(runner),'fixture':sha(fixture),'manifest':sha(bundle/'manifest.json'),
              'comparison':sha(Path(__file__)),'guard':sha(backend/'run_benchmark.py'),
              'arguments':common}
    retained={}
    if args.resume_from:
        old=json.loads(args.resume_from.read_text())
        if old['identity']!=identity:raise ValueError('comparison resume identity differs')
        for cell in old['cells']:
            if cell['status']!='PASS':continue
            path=Path(cell['summary'])
            if sha(path)!=cell['summary_sha256']:raise ValueError('retained receipt changed')
            key=(cell['dtype'],*cell['shape'],cell['metadata'])
            if key in retained:raise ValueError('duplicate comparison cell')
            retained[key]=cell
    result={'status':'INCOMPLETE','scope':'SF_RESIDENT_VS_FQ_NO_SF_PREPASS_SCREEN',
            'identity':identity,'cells':[],'pairs':[]}
    def save():
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    points=[(d,m,n,k) for d in (0,1) for n,k in ((512,2048),(4096,4096)) for m in (1,8,128)]
    for index,(d,m,n,k) in enumerate(points):
        order=['sf','fq'] if index%2==0 else ['fq','sf']
        payload={}
        configs=[0,3] if m<=8 else [2,5]
        for mode in order:
            key=(d,m,n,k,mode)
            if key in retained:
                cell={**retained[key],'retained_from':str(args.resume_from)}
            else:
                out=args.output/f'd{d}-m{m}-n{n}-k{k}-{mode}'
                command=[sys.executable,str(runner),*common,'--metadata',mode,
                    '--dtype',str(d),'--m',str(m),'--n',str(n),'--k',str(k),
                    '--configs',*map(str,configs),'--timing','--output',str(out)]
                print(f'GGUF_Q4_FQ_PROGRESS point={index+1}/12 metadata={mode} shape={m}x{n}x{k} dtype={d}',flush=True)
                rc=subprocess.run(command).returncode
                cell={'dtype':d,'shape':[m,n,k],'metadata':mode,'command':command,
                      'status':'PASS' if rc==0 else 'INVALID','summary':str(out/'summary.json'),
                      'summary_sha256':sha(out/'summary.json')}
            result['cells'].append(cell);save()
            if cell['status']!='PASS':return 1
            receipt=json.loads(Path(cell['summary']).read_text())
            if receipt['status']!='PASS':raise ValueError('missing completed guarded run')
            rows=receipt['result'][0]['rows']
            if len(rows)!=2 or {r['config'] for r in rows}!=set(configs):
                raise ValueError('comparison inventory differs')
            payload[mode]={r['config']:r for r in rows}
        for config in configs:
            sf,fq=(payload[mode][config] for mode in ('sf','fq'))
            if sf['raw_gguf_sha256']!=fq['raw_gguf_sha256']:raise ValueError('input differs')
            equal=([p['sha256'] for p in sf['proof']]==[p['sha256'] for p in fq['proof']])
            if not equal:raise ValueError('metadata delivery changed raw output; timing not admitted')
            pair={'dtype':d,'shape':[m,n,k],'config':config,'raw_equal':equal,
                  'clock_matched':sf['clocks']==fq['clocks'],
                  'sf_us':sf['median_us'],'fq_us':fq['median_us'],
                  'delta_pct':100*(fq['median_us']/sf['median_us']-1),
                  'sf_resident_bytes':sf['weight_bytes'],'fq_resident_bytes':fq['weight_bytes']}
            result['pairs'].append(pair);save()
            print('GGUF_Q4_FQ_PAIR '+json.dumps(pair),flush=True)
    result['status']='PASS';save();return 0

if __name__=='__main__':raise SystemExit(main())
