"""Summarize saved GGUF screens without promoting them to production winners."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--remote-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--memory',type=Path,help='optional completed direct sanitizer receipt')
    a=p.parse_args();a.root=a.root.resolve(strict=True)
    if a.output.exists():raise ValueError('fresh report output required')
    a.output.mkdir(parents=True)
    def resolve(remote):
        result=(a.root/Path(remote).relative_to(a.remote_root)).resolve(strict=True)
        if not result.is_relative_to(a.root):raise ValueError('receipt escaped root')
        return result
    screen=json.loads((a.root/'screen-v1/summary.json').read_text())
    stages=json.loads((a.root/'stage-screen-v2/summary.json').read_text())
    if screen['status']!='PASS' or stages['status']!='PASS':raise ValueError('incomplete screen')
    rows=[];authorities={};numeric={}
    for name in ('numeric-f16-v2','numeric-bf16-v3','stage-numeric-f16-v1','stage-numeric-bf16-v1'):
        path=a.root/name/'summary.json';receipt=json.loads(path.read_text())
        if receipt['status']!='PASS':raise ValueError('numeric gate failed')
        numeric[name]={'cells':len(receipt['result'][0]['rows']),
                       'sha256':sha(path),'maximum_relative_l2':max(c['relative_l2']
                           for r in receipt['result'][0]['rows'] for c in r['proof'])}
    for point in screen['points']:
        for arm,entry in point['arms'].items():
            path=resolve(entry['summary']);receipt=json.loads(path.read_text())
            if receipt['status']!='PASS':raise ValueError('invalid screen child')
            authorities[str(path.relative_to(a.root))]=sha(path)
            for r in receipt['result'][0]['rows']:
                rows.append({'phase':'initial-screen','dtype':r['dtype'],'m':r['shape'][0],
                    'n':r['shape'][1],'k':r['shape'][2],'arm':arm,'config':r['config'],
                    'median_us':r['median_us'],'modeled_unique_weight_GBps':r['weight_bytes']/r['median_us']/1000,
                    'clocks':r['clocks'],'admission':'SCREEN_ONLY'})
    pairs=[]
    for point in stages['points']:
        path=resolve(point['summary']);receipt=json.loads(path.read_text())
        if sha(path)!=point['summary_sha256'] or receipt['status']!='PASS':raise ValueError('stage authority differs')
        authorities[str(path.relative_to(a.root))]=sha(path)
        for pair in point['pairs']:
            if not pair['raw_equal']:raise ValueError('stage math changed')
            pairs.append({'dtype':point['dtype'],'shape':point['shape'],**pair,
                          'admission':'SCREEN_ONLY' if pair['clock_matched'] else 'CLOCK_UNMATCHED_UNRESOLVED'})
        for r in receipt['result'][0]['rows']:
            rows.append({'phase':'stage-screen','dtype':r['dtype'],'m':r['shape'][0],
                'n':r['shape'][1],'k':r['shape'][2],'arm':'candidate','config':r['config'],
                'median_us':r['median_us'],'modeled_unique_weight_GBps':r['weight_bytes']/r['median_us']/1000,
                'clocks':r['clocks'],'admission':'SCREEN_ONLY_SEE_PAIR_CLOCK_GATE'})
    with (a.output/'benchmark.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    memory={'status':'NOT_ATTACHED','limitation':'TMA_OUTPUT_NEGATIVE_NOT_DETECTED'}
    if a.memory:
        saved=json.loads(a.memory.read_text())
        if saved['status']!='PASS' or len(saved['cases'])!=19:
            raise ValueError('direct sanitizer cohort incomplete')
        for record in saved['cases']:
            if record['status']!='PASS' or sha(resolve(record['log']))!=record['log_sha256']:
                raise ValueError('sanitizer case/log differs')
        memory.update(status='PASS_SCOPED_3_CONFIGS_2_DTYPES_3_TOOLS_PLUS_SCALAR_NEGATIVE',
                      receipt_sha256=sha(a.memory),library_sha256=saved['library_sha256'])
    result={'status':'CHECKPOINT_COMPLETE_NOT_PRODUCT_ADMISSION','numeric':numeric,
            'initial_points':len(screen['points']),'stage_points':len(stages['points']),
            'stage_pairs':pairs,'authority_sha256':authorities,
            'scope':'DENSE_Q4_K_RESIDENT_SF_F16_BF16_ON_H800',
            'memory':memory,
            'not_proven':['FQ','SF_EXPANSION_COST','CANONICAL_LAYOUT_ONLY_ABLATION',
                          'MATURE_LIBRARY_GGUF_PARITY','PPU17_NATIVE','GROUPED','REAL_MODEL']}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GGUF_Q4_REPORT saved='+str(a.output)+' numeric_cells='+str(sum(x['cells'] for x in numeric.values())))

if __name__=='__main__':main()
