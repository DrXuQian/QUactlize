import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from dev.gemv_simt.q6_cycles import CYCLES, RATES, metrics, compare
from dev.gemv_simt.q6_index import ROOT


def table(cycles,frequency=1700000000,**extra):
    values={'ID':'0','Kernel Name':'q6',CYCLES[0]:str(cycles),CYCLES[1]:str(cycles-1),
            CYCLES[2]:str(cycles-3),**{k:str(frequency) for k in RATES},**extra}
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=list(values),quoting=csv.QUOTE_ALL)
    writer.writeheader();writer.writerow(values);return output.getvalue()


def reports(tmp_path,baseline=(110,111,109),candidate=(100,101,99)):
    rows=[]
    for rnd in range(3):
        for arm in ((0,1) if rnd%2==0 else (1,0)):
            path=tmp_path/f'r{rnd}-a{arm}.csv'
            path.write_text(table((baseline,candidate)[arm][rnd],**{'ppu__time_duration.sum':'999999999'}))
            rows.append(dict(round=rnd,arm=arm,status='PASS',raw_csv=str(path),
                             grid=[1024,1,1],block=[256,1,1],symbol=f'kernel-{arm}'))
    return rows


def test_actual_cycles_score_ignores_converted_time(tmp_path):
    rows=reports(tmp_path)
    result=compare(rows)
    assert result['median']==[110,100]
    assert result['verdict']=='LOWER_CYCLES_ALL_PAIRS'
    assert result['unit']=='cycles' and result['performance_admitted'] is False
    assert result['clock_conditions_match'] is True
    assert result['median_delta_pct']==pytest.approx(-100/11)
    for row in rows:
        p=Path(row['raw_csv']);p.write_text(p.read_text().replace('999999999','1'))
    assert compare(rows)==result


def test_frequency_difference_cannot_be_a_speed_win(tmp_path):
    rows=reports(tmp_path)
    Path(rows[1]['raw_csv']).write_text(table(50,frequency=1000000000))
    result=compare(rows)
    assert not result['clock_conditions_match']
    assert result['verdict']=='CLOCK_CONDITIONS_DIFFER_NO_SPEED_ADMISSION'


@pytest.mark.parametrize('candidate,verdict',[
    ((120,121,119),'HIGHER_CYCLES_REJECT'),((100,115,99),'MIXED_CYCLES_NO_SPEED_ADMISSION')])
def test_losing_and_mixed_candidates_not_accepted(tmp_path,candidate,verdict):
    assert compare(reports(tmp_path,candidate=candidate))['verdict']==verdict


def test_missing_duplicate_failed_geometry_and_symbol_rejected(tmp_path):
    rows=reports(tmp_path)
    for damaged in (rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[rows[-1]|dict(status='FAIL')]):
        with pytest.raises(ValueError,match='inventory'):compare(damaged)
    with pytest.raises(ValueError,match='geometry'):
        compare(rows[:-1]+[rows[-1]|dict(grid=[1,1,1])])
    with pytest.raises(ValueError,match='identity'):
        compare(rows[:-1]+[rows[-1]|dict(symbol='wrong')])


def test_cycle_fields_fail_closed():
    text=table(100)
    assert metrics(text)[CYCLES[0]]==100
    for bad in ('0','-1','nan','inf','bad'):
        with pytest.raises(ValueError):metrics(table(100,**{CYCLES[0]:bad}))
    with pytest.raises(ValueError):metrics(text.replace(CYCLES[0],'missing'))
    with pytest.raises(ValueError):metrics(text+text.splitlines()[-1]+'\n')


def test_cycle_plan_is_one_point_three_alternating_pairs(tmp_path):
    out=tmp_path/'plan'
    subprocess.run([sys.executable,str(ROOT/'tools/run_q6_index.py'),'--sdk',str(tmp_path),
                    '--output',str(out),'--plan-only'],check=True)
    plan=json.loads((out/'plan.json').read_text())
    assert len(plan['points'])==1 and plan['profile_rounds']==3
    assert plan['profile_arms']==[0,1] and plan['arms']==['sparse-metadata','unsigned-index']
    assert not (out/'build').exists()
    result=subprocess.run([sys.executable,str(ROOT/'tools/run_q6_index.py'),'--sdk',str(tmp_path),
                           '--output',str(out),'--plan-only'],capture_output=True,text=True)
    assert result.returncode and '--output must be a new directory' in result.stderr


def test_box_and_profile_source_contract():
    subprocess.run(['bash','-n',str(ROOT/'tools/run_q6_index_box.sh')],check=True)
    script=(ROOT/'tools/run_q6_index_box.sh').read_text()
    assert 'git submodule update' not in script and 'LLAMA_CI' not in script
    assert 'score=CE_CYCLES pairs=3' in script
    profile=(ROOT/'tools/profile_q6_metadata.py').read_text()
    assert "manifest.get('schema')=='q6-index-v1'" in profile
    assert "import_module('dev.gemv_simt.build_q6_index').inputs" in profile


def test_collection_alternates_six_distinct_single_launch_reports(tmp_path,monkeypatch):
    from tools import run_q6_metadata as runner, run_kpack_llama_ab as ab
    monkeypatch.setattr(ab,'device',lambda _:dict(l2=dict(bytes=67108864),device=dict(pci='host-stub')))
    monkeypatch.setattr(runner,'harness_identity',lambda _:dict(scope='HOST_ORCHESTRATION_ONLY'))
    monkeypatch.setattr(runner,'Path',lambda p:tmp_path if p=='/tmp' else Path(p))
    calls=[]

    def execute(command,log):
        if '--child' in command:
            nodes=[dict(symbol=f'kernel-{a}',grid=[1024,1,1],block=[256,1,1]) for a in (0,1)]
            Path(command[command.index('--result')+1]).write_text(json.dumps(dict(status='PASS',nodes=nodes)))
        elif '--export' in command:
            arm=int(command[command.index('--arm')+1]);calls.append(arm)
            assert command[command.index('--launch-count')+1]=='1'
            assert command[command.index('--graph-profiling')+1]=='none'
            assert command[command.index('--cache-control')+1]=='all'
            Path(command[command.index('--export')+1]+'.acurep').write_text(str(arm))
        else:
            assert '--import' in command
            arm=int(Path(command[command.index('--import')+1]).read_text())
            log.write_text(table(110 if arm==0 else 100,**{'Kernel Name':f'kernel-{arm}',
                           'Grid Size':'(1024,1,1)','Block Size':'(256,1,1)'}))
        return 0

    monkeypatch.setattr(runner,'execute',execute)
    a=NS(output=tmp_path/'output',sdk=tmp_path,acu=Path('acu'),plan_only=False,confirm_idle=True,l2_bytes=0)
    assert runner.collect(a,builder=lambda *args:None,point_indices=(0,),profile_rounds=3)==0
    result=json.loads((a.output/'summary.json').read_text())
    assert calls==[0,1,1,0,0,1] and len(result['profiles'])==6
    assert len({r['report'] for r in result['profiles']})==6
    assert compare(result['profiles'])['verdict']=='LOWER_CYCLES_ALL_PAIRS'
