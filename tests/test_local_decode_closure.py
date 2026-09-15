import copy
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

import pytest

from quactlize.runtime.compiler import sha
from tools.attach_kpack_local_gates import payload_paths

ROOT=Path(__file__).resolve().parents[1]


def test_replayed_ppu_results_preserve_scope_and_midpoint_diagnosis():
    r=json.loads((ROOT/'docs/measurements/local_closure_replay_ppu_20260916.json').read_text())
    assert r['bf16']==dict(expected=746,passed=733,failed=1,not_run=12)
    assert r['q8']['numeric']=='PASS' and r['q8']['contexts']==12480
    points=r['q8']['performance']
    assert len(points)==len({x['case'] for x in points})==50
    assert sum(x['delta_pct']<0 for x in points)==48
    for point in points:
        assert point['ring_copies']*point['useful_bytes']>=2.25*point['l2_bytes']
        assert all(len(x['round_medians'])==6 for x in point['best'].values())
    replay=r['q4_rounding_replay']
    assert replay['rounded_output_sha256']=='1d95c4dfa905fe0845673a519e4cfd3109238d9edec7d484262d4f4434f0434e'
    assert replay['unrounded_reference_proof']['bad']==0 and len(replay['legacy_bad'])==8
    assert r['moe_prepare']['status']=='NEGATIVE_CONTROL_FAIL'


def test_returned_ppu_closure_keeps_failures_and_unexecuted_cases_separate():
    r=json.loads((ROOT/'docs/measurements/local_closure_ppu_20260916.json').read_text())
    b=r['bf16']
    assert (b['expected'],b['passed'],b['failed'],b['not_run'])==(746,715,2,29)
    assert sum(p['passed'] for p in b['parts'])==b['passed']
    assert sum(p['completed']-p['passed'] for p in b['parts'])==b['failed']
    assert sum(p['expected']-p['completed'] for p in b['parts'])==b['not_run']
    assert r['q8']['status']=='LOAD_FAILED_NOT_MEASURED'
    assert r['selected_q4']['denominator']['bf16_cells']==258
    assert r['selected_q4']['status']=='PASS' and not r['selected_q4']['timing_valid']
    moe=r['moe_prepare']
    assert (moe['numeric_contexts'],moe['passed'],moe['expected'])==(3840,48,48)
    assert len(moe['cases'])==48
    for row in moe['cases']:
        assert row['samples_per_arm']==60
        assert row['delta_pct']==(row['candidate_us']/row['baseline_us']-1)*100
    all_simt=[r for r in moe['cases'] if r['parameters'][2]==5]
    assert len(all_simt)==16 and all(r['delta_pct']<0 for r in all_simt)
    assert any(r['delta_pct']>0 for r in moe['cases'] if r['parameters'][2]!=5)
    assert len(r['bf16_failures'])==2


def test_local_evidence_keeps_control_faults_and_complete_ncu_units():
    r=json.loads(gzip.decompress((ROOT/'docs/measurements/local_optimizations_20260915.json.gz').read_bytes()))
    assert r['admission']=='NVIDIA_GUIDANCE_ONLY_PPU_AND_MODEL_PENDING'
    assert r['q8_numeric']['expected']==12480 and r['q8_numeric']['status']=='PASS'
    assert len(r['q8'])==50
    for point in r['q8']:
        assert point['ring_copies']*point['useful_bytes']>=2.25*point['l2_bytes']
        for best in point['best'].values():
            assert len(best['rounds'])==6 and all(len(x)==15 for x in best['rounds'])
            assert statistics.median(x for values in best['rounds'] for x in values)==best['median_us']
    assert r['moe_numeric']['contexts']==3840 and r['moe_numeric']['status']=='PASS'
    assert (r['moe']['passed'],r['moe']['expected'])==(46,48)
    assert sum(x['status']=='FAIL' for x in r['moe']['results'])==2
    for name,report in r['reports'].items():
        assert len(report['sha256'])==64
        assert report['metrics']['launch__registers_per_thread']['unit']=='register/thread'
        # DRAM fields contain scaled profiler units; never treat them as bytes.
        assert report['metrics']['dram__bytes_read.sum']['unit'] in ('Kbyte','Mbyte')
    assert r['reports']['q8-n2048-k512-vector']['metrics']['launch__registers_per_thread']['value']=='56'


def test_typed_q4_cuda_evidence_covers_every_compiled_selected_recipe():
    from dev.bf16_fastpath.gate import summarize
    with tarfile.open(ROOT/'docs/measurements/bf16_q4_fastpath_5070_20260916.tgz') as archive:
        assert set(archive.getnames())=={'compact.json','summary.json','manifest.json'}
        data={n:archive.extractfile(n).read() for n in archive.getnames()}
    compact=json.loads(data['compact.json']);result=json.loads(data['summary.json']);manifest=json.loads(data['manifest.json'])
    assert compact['manifest_sha256']==hashlib.sha256(data['manifest.json']).hexdigest()
    assert compact['summary_sha256']==hashlib.sha256(data['summary.json']).hexdigest()
    assert result['manifest_sha256']==compact['manifest_sha256']
    assert summarize(manifest,result['results'])['status']=='PASS'
    assert compact['compiled_recipes']==compact['covered_recipes']==40
    assert compact['bf16_eager_cells']==258 and compact['f16_v1_v2_controls']==129
    assert compact['ppu_device_admission']=='PENDING' and not compact['performance_admitted']


def test_local_gate_payload_set_and_no_production_admission(tmp_path):
    directory=tmp_path/'local-gates';directory.mkdir()
    names=('q8/q8.so','q8/manifest.json','moe/bench','moe/manifest.json')
    for name in names:
        p=directory/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode())
    gate=dict(schema='quactlize.local-decode-experiments.v1',production_selection_changed=False,
              files={n:sha(directory/n) for n in names})
    manifest=directory/'manifest.json'
    def receipt(value):
        manifest.write_text(json.dumps(value))
        return dict(path='local-gates/manifest.json',sha256=sha(manifest))
    assert len(payload_paths(tmp_path,receipt(gate)))==5
    for changed in ('missing','extra','default','hash'):
        bad=copy.deepcopy(gate)
        if changed=='missing':bad['files'].pop('q8/q8.so')
        if changed=='extra':bad['files']['../foreign']='a'*64
        if changed=='default':bad['production_selection_changed']=True
        if changed=='hash':bad['files']['moe/bench']='b'*64
        with pytest.raises(ValueError):payload_paths(tmp_path,receipt(bad))
    link=directory/'q8/q8.so'
    link.unlink();link.symlink_to(directory/'moe/bench')
    with pytest.raises(ValueError):payload_paths(tmp_path,receipt(gate))


def test_prebuilt_entry_preserves_shell_and_does_not_compile():
    script=ROOT/'tools/run_kpack_local_closure_box.sh'
    subprocess.run(['bash','-n',script],check=True)
    text=script.read_text()
    assert '\n(\n' in text and 'Current Docker shell is preserved.' in text
    assert 'lfs pull origin' in text and 'verify_kpack_dispatch.py' in text
    assert 'build_kpack' not in text and 'cmake' not in text and 'nvcc' not in text
    assert 'dev/bf16_fastpath/gate.py' in text and 'dev/bf16_compute/run.py' in text
    assert 'failed=$((failed+1))' in text and 'production_defaults=UNCHANGED' in text


def test_local_closure_phase_filter_rejects_ambiguous_or_empty_requests():
    import sys
    text=(ROOT/'tools/run_kpack_local_closure_box.sh').read_text()
    code=text.split('"$PYTHON" - "$LOCAL_PHASES" <<\'PY\'\n',1)[1].split('\nPY\n',1)[0]
    for value,ok in [('moe-prepare,bf16',True),('q8',True),('q8,bf16,bf16-selected-q4',True),
                     ('',False),('q8,q8',False),('bf16,',False),('all',False),('bf16; q8',False)]:
        result=subprocess.run([sys.executable,'-c',code,value],capture_output=True)
        assert (result.returncode==0)==ok
    assert 'requested-phases.txt' in text
    for name in ('q8','moe-prepare','bf16','bf16-selected-q4'):
        assert f'if selected_phase {name}; then' in text


def test_model_chain_gate_uses_requested_compute_and_matched_lookup():
    script=(ROOT/'tools/run_kpack_q4_model_box.sh').read_text()
    assert '--compute "$MODEL_COMPUTE"' in script
    assert script.index('stage=bf16-selected-q4')<script.index('stage=ci-build')
    helper=(ROOT/'tools/run_kpack_moe_gate.py').read_text()
    assert 'automatic_smallm(d,gc,arr,compute)' in helper
    assert 'prepare_compute(choice,call,tokens,compute,indexed=io)' in helper
