"""Recover unchanged-arm component samples omitted from the production table.

Read immutable review archives; never re-label a screen as confirmation and
never convert an absent record into zero microseconds.
"""
from collections import Counter
import hashlib
import json
import lzma
import math
import statistics
import tarfile
from pathlib import PurePosixPath

from dev.strategy.production import checked_path
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

# Sealed by the imported campaign receipts, not a guessed GPU name lookup.
HISTORICAL_COHORT = dict(
    sdk='d22a00a15fb7f378b7fd81d2ebee94275d2a686db6dc284f1fbdefef6f459772',
    compute_units=72, warp_size=32, l2_bytes=67108864,
    DriverGetVersion=13000, RuntimeGetVersion=13000)


def check_cohort(device):
    actual=dict(sdk=device['sdk'],l2_bytes=device['l2']['bytes'],
                **{k:device['device'][k] for k in HISTORICAL_COHORT if k not in ('sdk','l2_bytes')})
    changed=[k for k,v in HISTORICAL_COHORT.items() if actual[k]!=v]
    if changed:
        raise ValueError('historical comparison cohort differs: '+','.join(changed)+
                         '; do not compare old costs with this SDK/device; no timing started')
    return dict(status='SAME_DEVICE_CLASS_AND_SDK_NOT_SAME_PHYSICAL_CARD',
                expected=HISTORICAL_COHORT,actual=actual,paired=False)


def unchanged_screen(path, role, wanted):
    path=checked_path(path,role)
    proofs, components, tables={}, {}, {}
    needed=set();body=hashlib.sha256();counts=Counter();footer=False
    with lzma.open(path,'rb') as stream:
        for line in stream:
            if footer:raise ValueError('trailing historical records')
            row=json.loads(line);kind=row['type']
            if kind=='end':
                if (row['prior_lines_sha256']!=body.hexdigest() or row['counts']!=dict(counts)
                        or row['export_status']!='COMPLETE'):
                    raise ValueError('historical export footer differs')
                footer=True;continue
            body.update(line);counts[kind]+=1
            if kind=='definition' and row['table']in ('kernel','symbol','component_contract'):
                tables.setdefault(row['table'],{})[row['index']]=row['value']
            elif kind=='confirmation' and row['point'] in wanted:
                expected=wanted[row['point']]
                if row['record_sha256']!=expected['record_sha256']:continue
                if row['status']!='INCUMBENT_UNCHANGED' or row['journal_status']!='PASS':
                    raise ValueError('historical unchanged arm has no successful proof')
                proof=row['original_proof'];accuracy=proof['accuracy']
                if (accuracy.get('correctness')!='PASS' or accuracy.get('checks',0)<1 or
                        accuracy.get('bad_max',0)!=0 or accuracy.get('nonfinite_max',0)!=0 or
                        not 0<=accuracy.get('relative_l1_error_max',math.inf)<.005):
                    raise ValueError('historical unchanged arm accuracy missing')
                proofs[row['point']]=proof
                needed.update(proof['components'])
            elif kind=='component' and row['phase']=='screen' and row['id']in needed:
                if row.get('status')!='PASS' or row.get('timed_kernel_types')!=1 or not row['first_use_excluded']:
                    raise ValueError('historical screen component contract differs')
                samples=row['samples_us']
                if len(samples)<3 or any(len(r)<3 for r in samples) or any(
                        not math.isfinite(v) or v<=0 for r in samples for v in r):
                    raise ValueError('invalid historical screen samples')
                medians=[statistics.median(r)for r in samples]
                if statistics.median(medians)!=row['median_us']:
                    raise ValueError('historical median differs')
                node=dict(tables['kernel'][row['node']])
                node.update({k:tables['symbol'][node[k]]for k in ('name','symbol')if k in node})
                components[row['id']]=dict(row,node=node,contract=tables['component_contract'][row['contract']])
    if not footer:raise ValueError('truncated historical archive')
    results={}
    for pid,proof in proofs.items():
        missing=[cid for cid in proof['components']if cid not in components]
        selected=[components[cid]for cid in proof['components']if cid in components]
        results[pid]=dict(status='SCREEN_COMPONENTS_RECOVERED'if not missing else 'PRIOR_COMPONENT_IMPORT_REQUIRED',
            missing=missing,components=selected,proof=proof,
            historical_sum_us=math.fsum(c['median_us']for c in selected)if not missing else None,
            confirmation=False,source_archive_role=role)
    return results


def import_prior(path, recovered):
    if sha(path)!='fc836f44dcc15a5603b3455e30bb116b89ee225c0c91c6a62f6256f74c274c20':
        raise ValueError('unreviewed five-point decode archive')
    needed={cid for r in recovered.values() for cid in r['missing']}
    found={}
    with tarfile.open(path) as archive:
        seen=set()
        for member in archive:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..'in name.parts or member.name in seen or not member.isfile():
                raise ValueError('unsafe/duplicate prior archive member')
            seen.add(member.name)
            if name.parent!=PurePosixPath('decode/components') or name.stem not in needed:continue
            c=json.load(archive.extractfile(member))
            if (c['record_sha256']!=digest({k:v for k,v in c.items()if k!='record_sha256'}) or
                    digest(c['identity'])!=name.stem or c['status']!='PASS' or
                    c['scope']!='ISOLATED_SINGLE_KERNEL' or c['timed_kernel_types']!=1):
                raise ValueError('prior component identity/timing differs')
            samples=c['samples_us']
            if (len(samples)<3 or any(len(r)<3 for r in samples) or any(
                    not math.isfinite(x)or x<=0 for r in samples for x in r) or
                    statistics.median(map(statistics.median,samples))!=c['median_us']):
                raise ValueError('prior raw samples differ')
            found[name.stem]=dict(c,id=name.stem,node=c['identity']['node'],
                authority=c['identity']['authority'],contract=c['identity']['contract'])
    for row in recovered.values():
        row['components'] += [found[k]for k in row['missing']if k in found]
        row['missing']=[k for k in row['missing']if k not in found]
        if not row['missing']:
            row['status']='SCREEN_COMPONENTS_RECOVERED'
            row['historical_sum_us']=math.fsum(c['median_us']for c in row['components'])


def costs(report):
    """Compact immutable historical screening baseline, not new admission."""
    rows=[]
    for p in report['points']:
        evidence=p['quactlize']
        if p['cohort']=='prefill':
            rounds=evidence['component_rounds']
            parts=[dict(kind=c['kind'],median_us=statistics.median(r[i]['median_us']for r in rounds))
                   for i,c in enumerate(rounds[0])]
            origin='SIX_ROUND_CONFIRMATION'
        elif 'components'in evidence:
            parts=evidence['components'];origin='SMALLM_CONFIRMATION'
        else:
            recovered=report['recovered_unchanged'].get(p['id'],{})
            if recovered.get('status')!='SCREEN_COMPONENTS_RECOVERED':
                raise ValueError('cannot export a complete historical baseline with missing timings')
            parts=[dict(kind=c['kind'],median_us=c['median_us'])for c in recovered['components']]
            origin='UNCHANGED_ARM_SCREEN_NOT_FRESH_CONFIRMATION'
        rows.append(dict(id=p['id'],components=parts,sum_us=math.fsum(c['median_us']for c in parts),
                         origin=origin,pending=[g for g in p['pending']if not g.startswith('IMPORT_')],
                         evidence_record=evidence['record_sha256']))
    out=dict(schema='quactlize.native-comparison-history.v1',points=rows,
             source_plan=report['source_plan_sha256'],audit_sha256=report['sha256'],
             scope='HISTORICAL_SCREENING_COSTS_NOT_PAIRED_NO_REGRESSION_PROOF')
    return out|dict(sha256=digest(out))


def format_costs(data):
    """One point per line: reviewable source data, not a 30k-line raw archive."""
    header=[json.dumps(k)+': '+json.dumps(v,sort_keys=True)
            for k,v in data.items()if k!='points']
    rows=[json.dumps(row,sort_keys=True,separators=(',',':'))for row in data['points']]
    return '{\n  '+',\n  '.join(header)+',\n  "points": [\n    '+',\n    '.join(rows)+'\n  ]\n}\n'
