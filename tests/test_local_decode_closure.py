import copy
import gzip
import json
from pathlib import Path
import statistics
import subprocess

import pytest

from quactlize.runtime.compiler import sha
from tools.attach_kpack_local_gates import payload_paths

ROOT=Path(__file__).resolve().parents[1]


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


def test_model_chain_gate_uses_requested_compute_and_matched_lookup():
    script=(ROOT/'tools/run_kpack_q4_model_box.sh').read_text()
    assert '--compute "$MODEL_COMPUTE"' in script
    assert script.index('stage=bf16-selected-q4')<script.index('stage=ci-build')
    helper=(ROOT/'tools/run_kpack_moe_gate.py').read_text()
    assert 'automatic_smallm(d,gc,arr,compute)' in helper
    assert 'prepare_compute(choice,call,tokens,compute,indexed=io)' in helper
