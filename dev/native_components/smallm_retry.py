"""Recover the dense/indexed fixture mistake without relabelling old timings."""
from collections import Counter
import json
from pathlib import Path

from dev.strategy.refresh import family
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

OLD_SCHEMA='quactlize.dense-smallm-native-ab.v1'
OLD_ENTRY='250aa1155c3aa2433768e061ee462f59400b578adfc956761886fdfbb13142a4'
OLD_FIXTURE='b480094b0a46d4d4e53ca68eb85b0ee668faf085ca8dfe4f889f7b8efa733511'


def sealed(path):
    value=json.loads(Path(path).read_bytes())
    if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('previous small-M receipt seal differs: '+str(path))
    return value


def vector_retry(current, previous, checked_record):
    previous=Path(previous).resolve(strict=True)
    old=sealed(previous/'plan.json');run=sealed(previous/'run.json')
    if (old.get('schema')!=OLD_SCHEMA or run.get('plan')!=old['sha256'] or
            run.get('harness',{}).get('sha256')!=digest(run['harness']['files']) or
            run['harness']['files'].get('tools/run_dense_smallm_ab.py')!=OLD_ENTRY or
            run['harness']['files'].get('dev/strategy/indexed.py')!=OLD_FIXTURE):
        raise ValueError('retry requires the identified dense/indexed fixture run')
    # A repair cannot silently refresh the selected recipe or lose an old key.
    projected=[{k:v for k,v in p.items() if k!='execution_contract'} for p in current['points']]
    if (projected!=old['points'] or current['protocol']!=old['protocol'] or
            current['tc']!=old['tc'] or current['vectors']!=old['vectors']):
        raise ValueError('small-M retry selection/workload/protocol changed; start a separate comparison')
    assignments={}
    if len(run['cards'])!=len(run['assignments']) or len(run['devices'])!=len(run['cards']):
        raise ValueError('previous device assignment differs')
    for i,fs in enumerate(run['assignments']):
        if not str(run['cards'][i]).isdecimal():raise ValueError('invalid previous GPU ordinal')
        for f in fs:
            if f in assignments:raise ValueError('duplicate previous family assignment')
            assignments[f]=i
    wanted={family(Workload(**p['workload'])) for p in old['points']}
    if set(assignments)!=wanted:raise ValueError('previous family coverage differs')
    retained=[];invalidated=[]
    for p in old['points']:
        i=assignments[family(Workload(**p['workload']))]
        path=previous/('gpu-'+str(run['cards'][i]))/'points'/(p['id']+'.json')
        row=sealed(path)
        authority=dict(run=run['sha256'],shard=i,device=run['devices'][i])
        if row.get('point')!=p['id'] or row.get('authority')!=authority:
            raise ValueError('previous point authority differs: '+p['id'])
        record=dict(point=p['id'],status=row['status'],path=str(path),sha256=sha(path))
        if p['arms'][0]['recipe']['kind']=='tc':
            # Recompute samples/negatives and verify all four original files.
            # Retain regressions/uncertainty too, not only previous winners.
            checked=checked_record(path,p,authority)
            if checked is None or checked!=row:
                raise ValueError('previous TC point is incomplete: '+p['id'])
            retained.append(record)
        else:
            record['reason']='INDEXED_REQUEST_IS_NOT_DENSE_EXECUTION'
            invalidated.append(record)
    points=[p for p in current['points'] if p['arms'][0]['recipe']['kind'] in ('simt','q4')]
    if not points or len(points)!=len(invalidated):raise ValueError('vector recovery coverage differs')
    modules={a['module'] for p in points for a in p['arms']}
    counts=dict(current['counts'],points=len(points),full_points=len(old['points']),
        families=len({family(Workload(**p['workload'])) for p in points}),
        selected_configs=len(points),tc_modules=0,vector_modules=len(modules),
        retained_prior_tc=len(retained))
    result={k:v for k,v in current.items() if k!='sha256'}
    result.update(points=points,tc={},vectors={k:v for k,v in current['vectors'].items() if k in modules},
        counts=counts,retry=dict(schema='quactlize.dense-smallm-mode-recovery.v1',
            previous=str(previous),run_sha256=run['sha256'],plan_sha256=old['sha256'],
            retained=retained,invalidated=invalidated,
            retained_counts=dict(Counter(r['status'] for r in retained)),
            scope='FRESH_VECTOR_ABBA_PRIOR_TC_UNCHANGED_NOT_FULL_COHORT_ADMISSION',
            historical_candidate_or_native_timings_used=False,raw_results_modified=False))
    result['sha256']=digest(result)
    return result
