from dataclasses import replace

from dev.strategy.q4_space import proposals
from dev.strategy.prescreen import compile_key, rejection
from quactlize.dispatch.table import Recipe, Workload


def test_floor_is_bounded_shape_only_and_does_not_change_with_n_or_m():
    w=Workload(12,1024,5120,1)
    rows=proposals(w)
    assert len(rows)<=16
    assert rows==proposals(replace(w,n=8192,output_stride=0,m=8))
    assert {dict(r.parameters)['reader'] for r in rows}=={0,1,2}
    assert all(rejection(r,w) is None for r in rows)
    assert proposals(replace(w,m=128))==()
    assert proposals(replace(w,qtype=13))==()


def test_balanced_warps_follow_actual_k_group_loop():
    for k in (512,2048,4096,5120,8192,25600):
        w=Workload(12,1024,k,1)
        rows=proposals(w)
        for r in rows:
            p=dict(r.parameters)
            assert rejection(r,w) is None
            if p['reader']==1 and p['warps']!=8:
                workers=p['warps']*32//p['columns']
                assert (k//32)%workers==0
                assert (k//32)//workers in (2,4)
            if p['reader']==2 and p['warps']!=8:
                workers=p['warps']*32//p['columns']
                baseline=8*32//p['columns']
                assert (k//32)%workers==0
                assert (k//32)//workers == ((k//32)+baseline-1)//baseline-1


def test_q4_specialization_cost_depends_on_nk_compute_not_m_or_experts():
    w=Workload(12,1024,2048,1)
    r=proposals(w)[0]
    assert compile_key(w,r) is not None
    assert compile_key(w,r)==compile_key(replace(w,m=8),r)
    assert compile_key(w,r)!=compile_key(replace(w,n=2048,output_stride=0),r)
    assert compile_key(w,r)!=compile_key(replace(w,compute='bf16'),r)
    assert compile_key(w,r)==compile_key(replace(w,operation='indexed',experts=256,topk=8),r)


def test_reader_contracts_reject_invalid_groups_storage_and_launches():
    w=Workload(12,1024,2048,1)
    r=Recipe.make('q4','fq',reader=2,variant=7,warps=5,values=4,columns=8)
    assert rejection(r,w)=='Q4_WORKER_GROUP_ALIGNMENT'
    valid=proposals(w)[0]
    assert rejection(valid,w,max_threads=128)=='CTA_THREAD_LIMIT'
    assert rejection(valid,replace(w,storage='f16'))=='Q4_DECODE_STORAGE_CONTRACT'
