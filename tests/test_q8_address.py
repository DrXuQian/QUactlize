"""Host structure/ownership checks; device numerics and cycles remain Box gates."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.gemv_simt.q8_address_source import ARMS,body,source,geometry
from dev.gemv_simt.q8_address_plan import plan,inventory
from dev.gemv_simt.q8_dense_source import body as old_body
from dev.gemv_simt.general_plan import configs,workload

ROOT=Path(__file__).resolve().parents[1]


def test_three_prior_controls_preserved():
    assert body(1)==old_body(4).replace('dense_ablation_4','address_body_1')
    assert body(2)==old_body(5).replace('dense_ablation_5','address_body_2')
    text=source(8,'f16',configs(8))
    assert text.count('q8_vector::kernel_body<')==2


def test_same_dot_order_and_reducer_outside_load_window():
    old=body(1)
    for arm in (3,4,5):
        new=body(arm)
        assert new[new.index('        #pragma unroll\n        for(int p=0;p<Pairs;'):]==old[old.index('        #pragma unroll\n        for(int p=0;p<Pairs;'):]
        assert new.count('dot[p].x=fmaf(values[residue],q.x,dot[p].x);')==old.count('dot[p].x=fmaf(values[residue],q.x,dot[p].x);')
        assert new.count('scales<P>(')==1
        assert 'for(int g=' in new and 'unsigned g=' not in new
        assert ('weight_cursor+=' in new)==(arm in (3,5))
        assert ('uint32_t words[2][4][Pairs]' in new)==(arm in (4,5))
        if arm in (4,5):
            window=new[new.index('if constexpr(Hoist)'):new.index('} else {\n            // Keep the original narrow')]
            assert window.count('for(int half=0;half<2;++half)')==1
            assert 'words[half]' not in window
            assert window.index('for(int half')<window.index('uint32_t words')<window.index('load_words')
    with pytest.raises(ValueError):body(0)


def test_signed_induction_keeps_all_addresses_and_wide_products():
    for n in (256,1024,5120,8192,2**30):
        for workers in (16,32,64):
            for worker in range(workers):
                col=32
                low=worker*16*n+col
                units=2*(worker*n+col)
                for trip in range(5):
                    g=worker+trip*workers
                    for half in range(2):
                        for segment in range(2):
                            for residue in range(4):
                                assert low+(segment*8+half*4+residue)*n==(g*16+segment*8+half*4+residue)*n+col
                    assert units==2*(g*n+col)
                    low+=workers*16*n
                    units+=workers*2*n
    assert 'uint64_t(segment*8+half*4+residue)*c.n' in body(3)


def test_frozen_inventory_and_geometry():
    p=plan()
    assert p['numeric_pairs']==280 and p['timing_pairs']==160 and p['acu_reports']==36
    assert len(p['points'])==8 and not p['production_changed']
    for spec in inventory():
        for m in spec['numeric_m']:
            w=workload(8,'dense',m,spec['workload']['n'],spec['workload']['k'])
            for c in configs(8):
                for arm in range(len(ARMS)):
                    grid,block=geometry(w,c,arm)
                    assert grid[0]*grid[1]*grid[2]==m*w.n//(c['columns']*c['values'])
                    assert block==[c['warps']*32,1,1]


def test_address_plan_cli_and_resume(tmp_path):
    out=tmp_path/'plan'
    cmd=[sys.executable,str(ROOT/'tools/run_q8_dense.py'),'--sdk',str(tmp_path),'--output',str(out),
         '--experiment','address','--plan-only']
    subprocess.run(cmd,check=True,capture_output=True)
    assert json.loads((out/'plan.json').read_text())==plan()
    subprocess.run(cmd+['--resume'],check=True,capture_output=True)
    bad=subprocess.run(cmd+['--resume','--experiment','ablation'],capture_output=True)
    assert bad.returncode and b'frozen plan changed' in bad.stderr
    assert not (out/'build').exists()
