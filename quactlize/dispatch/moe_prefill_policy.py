"""Confirmed ordinary MoE projections; no router-dependent host selection.

This table is separate from the frozen dense/legacy prefill evidence. Exact
rows retain their own measurement identity; M buckets are labelled predictions.
"""
import json
import math
from pathlib import Path
import statistics

from quactlize.runtime.compiler import validate_parent
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / 'policies/kpack_moe_prefill_exact_v1.json'
HEADER = TABLE.with_suffix('.hpp')
BASE_FAMILIES = ((12,512,2048,1),(12,512,3072,1),(13,2048,512,8),(13,3072,512,8))
CONCAT_FAMILIES = ((12,1024,2048,1),(12,1024,3072,1))
FAMILIES = BASE_FAMILIES + CONCAT_FAMILIES
MS = (16,32,64,128,256,512,1024,2048,4096,5120)
ROUTERS = ('cluster','real','repeat2','repeat4','spread')
COMMIT = 'abafc5668827a07cabaf5b5dd91fc27652ed24f4'
ARCHIVE = '634e363e7ed0255d6c44584d20e719da94b5209597b52c4c9b81d21acf92cc6e'
REFRESH_COMMIT = '6e35abb64a96f35fb1bc45bf48a036d8f50dfca8'
REFRESH_ARCHIVE = '342200449fb4029ebf4e0ec6df3a3f5343c456b0a8294c3c859e3a0fca67f311'
CONCAT_COMMIT = '7cb0a3482d81a86353c7e48e5297145c6ca7a391'
CONCAT_ARCHIVE = 'c4a85ad4ca2b78cda06c6977378912e77722355fe0774736a2bb2b2d8b04b854'


def configuration(recipe):
    p = dict(recipe['parameters'])
    fields = {'qtype','symbol','tm','tn','tk','wm','wn','stages','ap','dn',
              'persistent','split','grid_mode','grid_b','dequant_config'}
    if recipe['kind'] != 'tc' or recipe['route'] not in ('fq','sf') or set(p) != fields:
        raise ValueError('confirmed MoE requires a complete TC recipe')
    route = 2 if recipe['route'] == 'fq' else 3
    parent = {k:p[k] for k in fields-{'split','grid_mode','grid_b','dequant_config'}}
    parent['route'] = recipe['route']+'-grouped'
    validate_parent(parent)
    if (p['qtype'] not in (12,13) or p['split'] not in (1,2,4,8) or
            p['grid_mode'] not in (0,2,3) or p['grid_b'] < 0 or
            p['dequant_config'] not in ((-1,) if route == 2 else (0,4,5))):
        raise ValueError('MoE measured launch/metadata contract differs')
    return dict(p,route=route)


def validate(data):
    if (data['sha256'] != digest({k:v for k,v in data.items() if k!='sha256'}) or
            data['source_revision']['quactlize_commit'] != COMMIT or
            data['archive_sha256'] != ARCHIVE or
            data['timing'] != 'ASYS_GPU_ACTIVITY_MEAN_COMPONENT_SUM' or
            data['prepare'] != 'PUBLIC_QUACTLIZE_PREPARE_V2' or
            data['runtime_router_readback'] or data['native_no_regression_admitted'] or
            data['model_tpot_admitted']):
        raise ValueError('MoE prefill evidence scope/identity differs')
    if [r['key'] for r in data['rows']] != [[q,n,k,ch,m] for q,n,k,ch in FAMILIES for m in MS]:
        raise ValueError('MoE prefill measured domain differs')
    concat = data['concat']
    if (concat['source_revision']['quactlize_commit'] != CONCAT_COMMIT or
            concat['archive_sha256'] != CONCAT_ARCHIVE or
            concat['routers'] != ['cluster','spread'] or
            concat['baseline'] != 'ORIGINAL_MERGED_PROJECTION' or
            concat['scope'] != 'CANONICAL_2N_PROJECTION_NOT_SWIGLU_OR_FFN_TIMING'):
        raise ValueError('MoE concat evidence identity/scope differs')
    for row in data['rows']:
        config = configuration(row['recipe'])
        refreshed = row.get('evidence')
        merged = tuple(row['key'][:4]) in CONCAT_FAMILIES
        if merged:
            if refreshed != {'cohort':'concat'}:
                raise ValueError('MoE concat row lacks its measured cohort')
        elif refreshed and (row['key'] not in ([13,2048,512,8,5120],[13,3072,512,8,5120]) or
                refreshed['quactlize_commit'] != REFRESH_COMMIT or
                refreshed['archive_sha256'] != REFRESH_ARCHIVE or
                refreshed['baseline'] != 'LLAMA_NATIVE' or
                refreshed['timing'] != data['timing']):
            raise ValueError('refreshed MoE evidence identity differs')
        routers = concat['routers'] if merged else list(ROUTERS)
        if config['qtype'] != row['key'][0] or [p['router'] for p in row['profiles']] != routers:
            raise ValueError('MoE format/router coverage differs')
        for p in row['profiles']:
            baseline = p['native_us'] if refreshed and not merged else p['original_us']
            if merged and (p['latency_provenance']['quactlize_commit'] != CONCAT_COMMIT or
                    p['authority']['run'] != concat['run_sha256'] or
                    p['latency_provenance']['timing_method'] != 'ASYS_GPU_ACTIVITY'):
                raise ValueError('MoE concat profile provenance differs')
            if (len(p['selected_samples_us']) != (12 if refreshed and not merged else 6) or any(len(r)!=5 for r in p['selected_samples_us']) or
                    any(not math.isfinite(t) or t<=0 for r in p['selected_samples_us'] for t in r) or
                    not math.isclose(statistics.fmean(t for r in p['selected_samples_us'] for t in r),
                                     p['selected_us'],rel_tol=1e-12) or
                    p['selected_us'] > baseline or not 0<=p['max_relative_l1']<.005 or
                    not p['components'] or
                    sum(c['kind']=='producer' for c in p['components']) != 1 or
                    not math.isclose(sum(c['mean_us'] for c in p['components']),p['selected_us'],rel_tol=1e-12)):
                raise ValueError('MoE prefill numerical/full-call evidence differs')


def load():
    data=json.loads(TABLE.read_bytes());validate(data);return data


def header(data):
    validate(data)
    lines=['// Generated from router-independent projection confirmations; do not edit.',
        '// Complete projection costs with public prepare v2, not native/model admission.',
        '#pragma once','#include "kpack_zw810_heuristic_v1.hpp"',
        'namespace quactlize_moe_prefill_exact_v1 {',
        'using Config = quactlize_kpack_heuristic_v1::Config;',
        'struct Row { int q,n,k,channels,tokens,dequant; Config config; };',
        f'inline constexpr char kEvidence[] = "{data["sha256"]}";',
        'inline constexpr Row kRows[] = {']
    for row in data['rows']:
        p=configuration(row['recipe']);q,n,k,ch,m=row['key']
        mapping='0x51344b5034540001' if q==12 else '0x514b504b54000001'
        fields=[p[x] for x in ('qtype','route','tm','tn','tk','wm','wn','stages','ap','dn',
                               'persistent','split','grid_mode','grid_b')]+[0]
        lines.append('    {'+','.join(map(str,(q,n,k,ch,m,p['dequant_config'])))+
            ',{"moe-confirmed",'+json.dumps(p['symbol'])+',"MEASURED",'+','.join(map(str,fields))+
            ',UINT64_C('+mapping+')}},')
    return '\n'.join(lines+['};','} // namespace quactlize_moe_prefill_exact_v1',''])
