import io
import pytest

from dev.strategy.moe_inventory import collect
from tools.gguf_internal_shape_inventory import _synthetic_gguf, read_gguf_header


def loader(extra=(), experts=256):
    tensors=[('blk.0.ffn_gate_exps.weight',(2048,512,256),12),
             ('blk.0.ffn_up_exps.weight',(2048,512,256),12),
             ('blk.0.ffn_down_exps.weight',(512,2048,256),13),
             ('blk.0.attn_q.weight',(2048,4096),8), *extra]
    data=_synthetic_gguf([('general.name','fixture'),('general.architecture','qwen35moe'),
                         ('qwen35moe.expert_count',experts),('qwen35moe.expert_used_count',8)],tensors)
    header=read_gguf_header(io.BytesIO(data),'fixture')
    return lambda path:([dict(path=path,header=header)],header['tensors'])


def test_actual_weights_not_every_format_or_fused_shape():
    p=collect([dict(name='fixture',path='/fixture',tp=1)],loader=loader())
    assert p['status']=='PASS' and len(p['points'])==36
    assert {r['workload']['qtype'] for r in p['points']}=={12,13}
    gate=[r for r in p['points'] if r['workload']['qtype']==12]
    down=[r for r in p['points'] if r['workload']['qtype']==13]
    assert all(r['workload']['channels']==1 and r['workload']['n']==512 and len(r['consumers'])==2 for r in gate)
    assert all(r['workload']['channels']==8 and r['workload']['k']==512 for r in down)
    assert not p['gpu_work'] and not p['weights_read']
    assert len(p['fusion']['points'])==54  # 18 token counts x three actual output contracts
    fused=p['fusion']['points']
    assert all(r['contract']['weight_n']==1024 and r['contract']['logical_n']==512 for r in fused)
    assert {r['contract']['output_n'] for r in fused}=={512,1024}
    assert all(r['sweep_required'] and not r['timing_reuse'] and not r['execution_ready'] for r in fused)
    assert all(r['workload']['fusion']!='none' for r in fused)


def test_tp_is_not_guessed_from_world_size():
    p=collect([dict(name='tp',path='/fixture',tp=2)],loader=loader())
    assert p['status']=='INCOMPLETE' and not p['points']
    assert len(p['blocked'])==3 and {r['reason'] for r in p['blocked']}=={'TP_LOCAL_RECORDS_MISSING'}


def test_unknown_experts_and_unsupported_types_remain_visible():
    p=collect([dict(name='fixture',path='/fixture')],loader=loader(extra=[
        ('blk.1.future_exps.weight',(2048,512,256),12),
        ('blk.1.ffn_up_exps.weight',(2048,512,256),30)]))
    assert p['status']=='INCOMPLETE'
    assert p['blocked'][0]['reason']=='UNKNOWN_EXPERT_OPERATOR'
    assert p['excluded'][0]['reason']=='UNSUPPORTED_QUANT_OR_FLOAT'


def test_expert_metadata_mismatch_fails():
    with pytest.raises(ValueError,match='extent disagrees'):
        collect([dict(name='fixture',path='/fixture')],loader=loader(experts=128))


def test_duplicate_models_and_invented_m_are_rejected():
    model=dict(name='fixture',path='/fixture')
    with pytest.raises(ValueError,match='unique'):
        collect([model,model],loader=loader())
    with pytest.raises(ValueError,match='positive'):
        collect([model],loader=loader(),ms=[0])


def test_gate_up_mismatch_keeps_unfused_points_but_cannot_pair():
    def mismatched(path):
        rows,tensors=loader()(path)
        next(t for t in tensors if 'ffn_up_exps' in t['name'])['qtype']=13
        return rows,tensors
    p=collect([dict(name='fixture',path='/fixture')],loader=mismatched,ms=[1])
    assert len(p['points'])==3 and not p['fusion']['points']
    assert p['fusion']['unavailable'][0]['reason']=='GATE_UP_TYPE_SHAPE_OR_PARTITION_MISMATCH'


def test_fusion_recipe_floor_recalls_current_moe_references_not_old_times():
    from dev.strategy.moe_fusion import historical_recall, paired_recipes
    p=historical_recall()
    assert p['status']=='PASS' and len(p['points'])==17
    assert p['old_latency_rows_imported']==p['kernel_launches']==0
    assert not p['performance_admitted']
    req=p['points'][0]['request'] | dict(n=768,k=1536)
    candidates=paired_recipes(req)
    assert candidates and all(r['fixed_n'] in (0,1536) and r['fixed_k'] in (0,1536) for r in candidates)
    # Layout/body, not just TM/S, is part of the reference identity.
    fake=p['points'][0]['reference'] | dict(reader=999)
    assert fake not in p['points'][0]['candidates']


def test_fused_config_candidates_withhold_original_and_physical_nk():
    from dev.strategy.moe_inventory import add_candidates
    from dev.strategy.moe_plan import RULES
    from quactlize.dispatch.table import Observation, Recipe, StrategyTable, Workload
    p=collect([dict(name='fixture',path='/fixture')],loader=loader(),ms=[1])
    r=Recipe.make('tc','fq',symbol='target_only',qtype=12,tm=16,tn=64,tk=64,
                  wm=16,wn=16,stages=2,ap=0,dn=16,persistent=0,split=1,grid_mode=0,grid_b=0)
    table=StrategyTable([Observation(Workload(12,n,2048,1,'indexed','bf16',experts=256,topk=8),
                                     r,'target-only','PROPOSAL',0,'REFERENCE','fixture') for n in (512,1024)])
    p=add_candidates(p,table,RULES)
    assert len(p['fusion']['points'])==3 and not p['sweep_ready']
    for point in p['fusion']['points']:
        assert point['main_template_candidates']  # Q4 API floor, not target TC answers.
        assert all(p['configurations'][key]['kind']!='tc' for key in point['main_template_candidates'])
        if point['contract']['fusion']=='gate-up-swiglu':
            assert point['paired_api_candidates']
            assert 'CAPABILITY_GATE' in point['template_status']
