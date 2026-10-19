from dataclasses import asdict
from pathlib import Path
import csv
import io
import json
import re
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dev.gemv_simt.q6_index import index_body, source as parent_source
from dev.gemv_simt.q6_schedule import (ROOT, ARMS, WARPS, CONFIGS, KERNEL_NAMES, EQUAL_GROUPS,
                                      source, body, load_events, validate_record)
from dev.gemv_simt.q6_schedule_access import pattern
from dev.gemv_simt.q6_schedule_cycles import comparisons
from dev.gemv_simt.q6_cycles import CYCLES, RATES, compare
from quactlize.execution.simt_codegen import Config
from tools.run_q6_metadata import workloads, check_equal_groups, experiment_contract
from tools.profile_q6_metadata import validate_prior


def record():
    return dict(kernel_names=list(KERNEL_NAMES), configs=CONFIGS, equal_groups=EQUAL_GROUPS, sha256='image')


def test_original_control_unchanged_and_only_warp_qualification_changes():
    original = index_body((ROOT/'quactlize/execution/simt_kernel.cuh').read_text())
    changed = body().replace('q6_warp_body(', 'q6_unsigned_fields_body(').replace(
        '(Warps==2 || Warps==4 || Warps==8) && P==8', 'Warps==8 && P==8')
    assert changed == original
    for compute in ('f16','bf16'):
        old, new = parent_source(compute), source(compute)
        assert new.startswith(old.split('extern "C"')[0])
        assert 'arm>5' in new and 'if(arm==0)' not in new[new.index('extern "C"'):]
        assert 'simt::query_v2(*d,cfg,a,sizes)' in new and 'simt::buffers_v2(*d,sizes)' in new
        for i,w in enumerate(WARPS):
            assert f'case {i}: simt::' in new
            assert f'<<<blocks,{w*32},0,stream>>>' in new
    with pytest.raises(ValueError):
        source('int8')


def test_pipeline_preserves_all_non_load_math_and_only_reorders_loads():
    plain, pipe = body(), body(True)
    # Strip exactly the load preamble and lookahead, not the math to be proved.
    def strip(text):
        begin = text.index('            #pragma unroll\n            for (int segment=') if 'for (int segment=' in text else text.index('            static_assert(Segments==1')
        end = text.index('            #pragma unroll\n            for (int slot=', begin)
        text = text[:begin]+text[end:]
        text = re.sub(r'                    if \(slot==0 && residue<2\) \{.*?\n                    }\n', '', text, flags=re.S)
        return text.replace('q6_pipeline_body(', 'q6_warp_body(')
    assert strip(plain) == strip(pipe)
    assert 'unsigned next_k=g*F::group+half*4+residue+2;' in pipe
    assert pipe.count('load_words<P>') == 4  # two initial and two lookahead call sites
    assert pipe.count('__syncthreads()') == plain.count('__syncthreads()') == 1


def audit_load_events(events):
    loaded, used = set(), []
    for kind, value in events:
        if kind == 'load':
            assert value in range(4) and value not in loaded
            loaded.add(value)
        else:
            slot, residue = value
            assert residue in loaded
            used.append((slot,residue))
    assert loaded == set(range(4))
    assert used == [(s,r) for s in range(2) for r in range(4)]


def test_load_lifetime_and_negative_controls():
    for pipe in (False, True):
        events = list(load_events(pipe))
        audit_load_events(events)
        for index in range(len(events)):
            with pytest.raises(AssertionError):
                audit_load_events(events[:index]+events[index+1:])
        with pytest.raises(AssertionError):
            audit_load_events([('use',(0,0))]+events)
        with pytest.raises(AssertionError):
            audit_load_events(events+[('load',4)])
    # Same uses, no reload in slot 1, and exactly four words/plane per half.
    assert [e for e in load_events(False) if e[0]=='use'] == [e for e in load_events(True) if e[0]=='use']


@pytest.mark.parametrize('w',[2,4,8])
def test_warp_group_coverage_phase_and_address_model(w):
    from dev.gemv_simt.q6_sparse_access import WORDS
    from dev.gemv_simt.access import pattern as full_pattern
    cfg = Config(3,4,w,8,1)
    phases = set()
    for n,k in ((32,256),(512,512),(512,2048),(1024,5120)):
        seen = []
        for iteration in range((k//16+w*8-1)//(w*8)+1):
            for warp in range(w):
                groups = [warp*8+lane//4+iteration*w*8 for lane in range(32)]
                active = [g for g in groups if g<k//16]
                assert len(active) in (0,32)
                if active:
                    assert len({g//8%4 for g in active}) == len({g//32 for g in active}) == 1
                    phases.add(active[0]//8%4)
                    seen.extend(active[::4])
                # Test every packet phase with a nonzero actual metadata base.
                row = pattern(14,cfg,n,k,warp=warp,iteration=iteration,bases=dict(A=0,low=0,high=0,units=36))
                old = full_pattern(14,cfg,n,k,warp=warp,iteration=iteration,bases=dict(A=0,low=0,high=0,units=36))
                assert row['lane_groups'] == groups
                assert [s for s in row['streams'] if not s['name'].startswith('metadata')] == [
                    s for s in old['streams'] if not s['name'].startswith('metadata')]
                meta = [s for s in row['streams'] if s['name'].startswith('metadata')]
                if active:
                    assert len(meta) == len(WORDS[active[0]//8%4])
                for s in row['streams']:
                    assert set(s['footprint']) == {'32','64','128'}
                    assert s['width_bytes'] == (4 if s['name'].startswith('metadata') else 16)
        assert sorted(seen) == list(range(k//16))  # no missing/duplicate scale group
    assert phases == {0,1,2,3}


def test_config_and_equality_contract_cannot_be_bypassed():
    r = record()
    assert experiment_contract(r) == (CONFIGS,EQUAL_GROUPS)
    for key in ('configs','equal_groups','kernel_names'):
        with pytest.raises(ValueError):
            validate_record(r | {key:[]})
    x = np.array([1.,2.], dtype=np.float32)
    outputs = [x+i for i in range(3)]*2
    check_equal_groups(outputs,EQUAL_GROUPS)
    outputs[5] = outputs[5]+1
    with pytest.raises(ValueError, match='same-order'):
        check_equal_groups(outputs,EQUAL_GROUPS)
    prior = dict(status='PASS', profiled=False, raw_equal=False, raw_equal_groups=EQUAL_GROUPS,
                 configs=CONFIGS, raw_equal_scope='MATCHED_WARP_PAIRS_ONLY_CROSS_WARP_INDEPENDENT_ORACLE',
                 timings=[{}]*6, authority_sha256='authority', library_sha256='image', workload=asdict(workloads()[0]))
    validate_prior(prior,{},r,workloads()[0],lambda _:'authority')
    for key,value in (('raw_equal',True),('raw_equal_groups',[]),('configs',[]),('timings',[{}]*2)):
        with pytest.raises(ValueError):
            validate_prior(prior | {key:value},{},r,workloads()[0],lambda _:'authority')


def table(arm, rnd):
    fields = {'ID':'0','Kernel Name':f'kernel-{arm}','Grid Size':'(1024,1,1)',
              'Block Size':f'({WARPS[arm]*32},1,1)',
              **{k:str(1000-10*arm+rnd) for k in CYCLES}, **{k:'1700000000' for k in RATES}}
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(fields), quoting=csv.QUOTE_ALL)
    writer.writeheader();writer.writerow(fields)
    return output.getvalue()


def point():
    return dict(status='PASS',workload=asdict(workloads()[0]),
                nodes=[dict(name='void quactlize'+KERNEL_NAMES[a]+'call)',symbol=f'kernel-{a}',
                            grid=[1024,1,1],block=[w*32,1,1]) for a,w in enumerate(WARPS)])


def test_18_alternating_single_kernel_profiles(tmp_path, monkeypatch):
    from tools import run_q6_metadata as runner, run_kpack_llama_ab as ab
    monkeypatch.setattr(ab,'device',lambda _:dict(l2=dict(bytes=67108864),device=dict(pci='host-stub')))
    monkeypatch.setattr(runner,'harness_identity',lambda _:dict(scope='HOST_ORCHESTRATION_ONLY'))
    monkeypatch.setattr(runner,'Path',lambda p:tmp_path if p=='/tmp' else Path(p))
    calls=[]

    def execute(command,log):
        if '--child' in command:
            Path(command[command.index('--result')+1]).write_text(json.dumps(point()))
        elif '--export' in command:
            arm=int(command[command.index('--arm')+1]);calls.append(arm)
            for option, value in (('--launch-count','1'),('--graph-profiling','none'),
                                  ('--cache-control','all'),('--profile-from-start','no')):
                assert command[command.index(option)+1] == value
            Path(command[command.index('--export')+1]+'.acurep').write_text(str(arm))
        else:
            assert '--import' in command
            arm=int(Path(command[command.index('--import')+1]).read_text())
            log.write_text(table(arm,(len(calls)-1)//6))
        return 0

    monkeypatch.setattr(runner,'execute',execute)
    a=NS(output=tmp_path/'out',sdk=tmp_path,acu=Path('acu'),plan_only=False,confirm_idle=True,l2_bytes=0)
    assert runner.collect(a,builder=lambda *args:None,point_indices=(0,),profile_rounds=3,
                          arms=ARMS,profile_arms=tuple(range(6))) == 0
    summary=json.loads((a.output/'summary.json').read_text())
    assert calls == [0,1,2,3,4,5,5,4,3,2,1,0,0,1,2,3,4,5]
    reports=summary['profiles']
    assert len({r['report'] for r in reports}) == 18
    result=comparisons(reports, point())
    assert len(result['comparisons']) == 5 and len(result['ranking']) == 6
    assert result['fastest_measured_arm'] == 5 and not result['production_changed']
    with pytest.raises(ValueError,match='18 unique'):
        comparisons(reports[:-1],point())
    with pytest.raises(ValueError,match='18 unique'):
        comparisons(reports[:-1]+[reports[0]],point())
    with pytest.raises(ValueError,match='geometry'):
        comparisons(reports[:-1]+[reports[-1] | dict(block=[256,1,1])],point())
    with pytest.raises(ValueError,match='identity'):
        comparisons(reports[:-1]+[reports[-1] | dict(symbol='wrong')],point())
    pair=[r for r in reports if r['arm'] in (0,1)]
    with pytest.raises(ValueError,match='geometry'): compare(pair)  # old default stays strict
    first=Path(reports[0]['raw_csv'])
    first.write_text(first.read_text().replace('1700000000','1000000000'))
    assert comparisons(reports,point())['fastest_measured_arm'] is None


def test_bounded_plan_and_shell(tmp_path):
    out=tmp_path/'plan'
    subprocess.run([sys.executable,str(ROOT/'tools/run_q6_schedule.py'),'--sdk',str(tmp_path),
                    '--output',str(out),'--plan-only'],check=True)
    plan=json.loads((out/'plan.json').read_text())
    assert len(plan['points']) == 1 and plan['arms'] == list(ARMS)
    assert plan['profile_rounds'] == 3 and plan['profile_arms'] == list(range(6))
    assert not (out/'build').exists()
    subprocess.run(['bash','-n',str(ROOT/'tools/run_q6_schedule_box.sh')],check=True)
    shell=(ROOT/'tools/run_q6_schedule_box.sh').read_text()
    assert 'git submodule update' not in shell and 'LLAMA_CI' not in shell
    assert '${POINTS:-0}' in shell and 'profiles=18' in shell


def test_native_audit_keeps_static_scope_and_control_bytes(tmp_path):
    from dev.gemv_simt.q6_schedule import KERNELS
    from dev.gemv_simt.q6_schedule_audit import audit
    sections, resources = [], []
    for arm,name in enumerate(KERNELS):
        symbol = f'_ZN9quactlize9execution4simt{len(name)}{name}E11qkg_call_v1i'
        # Synthetic native text exercises the parser, not a performance proof.
        seq = [('00 00 00 00 00 00 00 08','s.nop'),
               ('01 00 00 00 00 00 00 08','vmem.ld.b32x4\tvreg[0:3], [0x0] @vreg[4:5]'),
               ('02 00 00 00 00 00 00 08','v.fma.f32.rtte\tvreg0,vreg1,vreg2,vreg3')]
        if arm >= 3:
            seq = [seq[0],seq[2],seq[1]]
        section = f'Disassembly of section .text.kernel.{symbol}:\n'
        section += '\n'.join(f' {i*8:x}: {word}\t{op}' for i,(word,op) in enumerate(seq))+'\n'
        sections.append(section)
        resources.append(f'{symbol} RESOURCE INFO:\nvreg_number:96\nsreg_number:80\n'
                         f'shared_memory_size:{WARPS[arm]}\nSTACK SIZE:0\n.ARGUMENT:')
    (tmp_path/'bf16.isa.txt').write_text('\n'.join(sections))
    (tmp_path/'bf16.resources.txt').write_text('\n'.join(resources))
    result = audit(tmp_path,'bf16',tmp_path)
    assert result['a3_control_bytes_unchanged'] and result['no_stack']
    assert len(result['arms']) == 6 and len(result['pairs']) == 3
    assert all(p['same_math_load_shuffle_sites'] and p['native_schedule_differs'] for p in result['pairs'])
    assert result['device_admission'] == 'NOT_RUN'
    assert 'STATIC_NATIVE_SITES' in result['scope']
