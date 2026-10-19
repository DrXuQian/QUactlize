"""Reviewed failed-only continuation. Old measurements retain their old authority."""
import json
from pathlib import Path
import re
import tarfile

from dev.native_components.accounting import checked_component
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
DIAGNOSTIC_SHA='5aced49add8d19d3a60692c23a8fd0856fe1da5895cd1a1275d0ef05b3a90c9f'
PREVIOUS_FILES_SHA='0b03f09f4fb00bc8c43073d91f50ab48fe1585daae10230be98aff68dc750c8a'


def local_attempt(worker,saved):
    """Resolve only this worker's recorded attempt, including relocated reviews."""
    path=Path(saved)
    if len(path.parts)<4 or path.parts[-4]!=worker.name or path.parts[-3]!='attempts':
        raise ValueError('previous attempt path is outside its worker')
    result=(worker/Path(*path.parts[-3:])).resolve(strict=True)
    if not result.is_relative_to(worker.resolve()):raise ValueError('previous attempt escapes worker')
    return result


def inspect_previous(previous,rows):
    """Host-only audit; raw result/image verification is repeated on Box."""
    previous=previous.resolve(strict=True);by={r['id']:r for r in rows}
    files={};workers={};seen=set()
    def read(path):
        if path.is_symlink() or not path.resolve().is_relative_to(previous):
            raise ValueError('previous receipt escapes run')
        files[str(path.relative_to(previous))]=sha(path)
        return json.loads(path.read_text())
    run=read(previous/'run.json');read(previous/'summary.json')
    cards=run['devices']
    if cards!=['1','2','3','4','5','7']:raise ValueError('not the reviewed six-card run')
    for card in cards:
        worker=previous/('gpu-'+card)
        summary=read(worker/'summary.json');authority=read(worker/'authority.json')
        adapter=read(worker/'build/adapter.json');read(worker/'build/carrier.json')
        if authority['sha256']!=digest({k:v for k,v in authority.items()if k!='sha256'}):
            raise ValueError('previous authority digest differs')
        kinds={};reused=[];retry=[];deferred=[]
        for pid,record in summary['results'].items():
            if pid not in by or pid in seen:raise ValueError('previous point unknown or duplicated')
            seen.add(pid)
            if record['status']=='MEASURED':reused.append(pid);continue
            if record['status']!='FAIL':raise ValueError('unreviewed previous status')
            log=local_attempt(worker,record['log']);files[str(log.relative_to(previous))]=sha(log)
            text=log.read_text(errors='replace');w=by[pid]['workload']
            if 'native kernel params: named symbol not found'in text:
                kind='jit-params'
            elif re.search(r'independent GGUF spot dot error native=[\d.]+',text):
                if w['operation']=='indexed' and w['m']<=8 and w['qtype']in(12,13):kind='q8-activation'
                elif w['operation']=='dense' and w['m']>8 and w['qtype']in range(10,15):kind='native-fp16-deferred'
                else:raise ValueError('unreviewed numerical failure '+pid)
            else:raise ValueError('unreviewed previous failure '+pid)
            kinds[pid]=kind
            (deferred if kind=='native-fp16-deferred'else retry).append(pid)
        helpers={}
        for path in sorted((worker/'helpers').glob('*.json')):
            item=read(path)
            if item.get('authority')!=authority or item.get('record_sha256')!=digest({k:v for k,v in item.items()if k!='record_sha256'}):
                raise ValueError('previous helper authority/digest differs')
            checked_component(item['component']);helpers[path.stem]=item
        workers[worker.name]=dict(summary=summary,authority=authority,adapter=adapter,
            kinds=kinds,reused=reused,retry=retry,deferred=deferred,helpers=helpers)
    if seen!=set(by):raise ValueError('previous full workload coverage differs')
    return dict(files=files,files_sha256=digest(files),workers=workers,
        counts={k:sum(len(w[k])for w in workers.values())for k in ('reused','retry','deferred','helpers')})


def verify_previous(audit):
    if audit['files_sha256']!=PREVIOUS_FILES_SHA:
        raise ValueError('reviewed previous receipts/logs changed; old results are not relabelled')
    if audit['counts']!=dict(reused=1178,retry=268,deferred=98,helpers=36):
        raise ValueError('reviewed retry inventory differs')


def verify_diagnostic(path):
    if sha(path)!=DIAGNOSTIC_SHA:raise ValueError('not the reviewed 17-job diagnostic archive')
    with tarfile.open(path)as archive:
        def read(name):return json.load(archive.extractfile(name))
        summary=read('summary.json')
        if summary['status']!='DIAGNOSTIC_COMPLETE' or summary['expected']!=17 or len(summary['results'])!=17:
            raise ValueError('diagnostic is incomplete')
        for result in summary['results']:
            if result['rc'] or result['replay']!='PASS_TWO_INPUTS':raise ValueError('diagnostic replay failed')
            if set(result['comparisons'])!={'input0','input1'} or any(
                    c['relative_l2']!=0 or c['max_abs_difference']!=0 for c in result['comparisons'].values()):
                raise ValueError('reviewed diagnostic outputs differ')
            if result['kind']!='provider-precision' and result['arithmetic_status']!=['PASS']*4:
                raise ValueError('diagnostic arithmetic failed')
        adapter=read('build/adapter.json');carrier=read('build/carrier.json')
    # The device-admitted extraction/oracle/timer sources must still be exact.
    for name,want in adapter['identity']['test_sources'].items():
        if sha(ROOT/'dev/native_components'/name)!=want:
            raise ValueError('diagnostic-tested adapter source changed: '+name)
    for name,key in (('dev/native_components/carrier.cpp','source'),('dev/llama_ab/carrier.cpp','fixture')):
        if sha(ROOT/name)!=carrier['identity'][key]:raise ValueError('diagnostic-tested carrier source changed')
    return dict(archive_sha256=DIAGNOSTIC_SHA,adapter=adapter,
                scope='EXTRACTION_AND_NUMERIC_CONTROLS_NOT_FULL_CAMPAIGN_TIMING')


def validate_current(previous,receipt,device,probe):
    from tools.probe_native_component_failures import check_binding
    check_binding(previous['adapter'],receipt,device)
    check_binding(probe['adapter'],receipt,device)
    if previous['authority']['device']!=device:
        raise ValueError('retry physical GPU/SDK/L2 differs from its previous worker')
    if (previous['authority']['rounds'],previous['authority']['samples'])!=(3,3):
        raise ValueError('retry timing protocol differs')


def same_record(value,expected):
    if {k:v for k,v in value.items()if k!='data'}!=expected:
        raise ValueError('previous result differs from reviewed summary')


def comparison_summary(records,comparisons,previous):
    deferred=set(previous['deferred']);required=set(previous['summary']['results'])-deferred
    complete={r['id']for r in comparisons if r['status']=='HISTORICAL_SCREEN_ONLY'}
    resolved=required.issubset(complete)
    return dict(status='COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16'if resolved else'INCOMPLETE',
        expected=len(required|deferred),measured=sum(v['status']=='MEASURED'for v in records.values()),
        reused=len(previous['reused']),retry_expected=len(previous['retry']),
        retried_measured=sum(records.get(k,{}).get('status')=='MEASURED'for k in previous['retry']),
        deferred=len(deferred),deferred_ids=sorted(deferred),full_coverage=False,
        no_regression_admitted=False,model_tpot_admitted=False,production_changed=False)
