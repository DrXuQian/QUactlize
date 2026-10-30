"""Host control-flow/ABI tests, not simulated PPU performance."""
from copy import deepcopy
import ctypes as C
from dataclasses import asdict
from types import SimpleNamespace as NS
import time

import pytest

from dev.moe_confirm import selected,native_ab,plan
from dev.strategy.fixture_identity import expected
from dev.strategy.prefill_fixture import routing
from dev.strategy.resources import probe_spec
from dev.strategy.space import resolved_grid
from quactlize.dispatch.native import Choice
from quactlize.dispatch.table import Workload,Recipe


@pytest.mark.parametrize('route',['fq','sf'])
@pytest.mark.parametrize('fault',[None,'policy','parent','launch','workspace','helper'])
def test_selected_public_ticket_and_dynamic_bindings(monkeypatch,route,fault):
    p=next(p for p in plan.wired_plan()['points'] if p['arms'][0]['recipe']['route']==route)
    w=Workload(**p['workload']);arm=deepcopy(p['arms'][0]);recipe=Recipe.from_dict(arm['recipe'])
    params=dict(recipe.parameters);launch=resolved_grid(w,recipe,72,6)
    arm['binding']=dict(launch=launch)
    query=dict(request=probe_spec(w,recipe,launch['algorithm']),workspace_bytes=256)
    calls=[];planes=[dict(low=100+i,high=200+i,units=300+i,zero=400+i) for i in range(2)]
    b=NS(w=w,dispatch_root='host-double',offsets=NS(ptr=60),r=NS(stream=C.c_void_p(80)),
         ring=NS(planes=planes),rt=NS(sync=lambda:calls.append('sync')),
         prepare=lambda ptr:calls.append(('prepare',ptr)) or 0,
         finish=lambda ptr,bf:calls.append(('finish',ptr,bf)) or 0)
    def initialize(self,b,arm,record,query,sf):
        self.b=b;self.sf=sf;self.receipt={}
        self.a=NS(ptr=10);self.out=NS(ptr=20);self.buffers=[NS(ptr=30,size=256)]
    monkeypatch.setattr(selected.backend.TensorCore,'__init__',initialize)
    monkeypatch.setattr(selected.backend.TensorCore,'close',lambda self:calls.append('base-close'))
    class Function:
        def __init__(self,fn):self.fn=fn
        def __call__(self,*args):return self.fn(*args)
    class Dispatcher:
        def __init__(self,root):
            assert root=='host-double';self.runtime=4;self.handles=[]
            def prepare(runtime,choice,out):
                assert runtime==4;out._obj.value=77;return 0
            def run(handle,current,phase):
                assert handle.value==77 and phase==0
                c=current._obj.device_call.call
                assert current._obj.compute_type==current._obj.metadata_type==1
                calls.append(('run',c.a,c.output,c.workspace,c.offsets_device,c.low,c.metadata,c.zero,c.stream))
                return 0
            self.lib=NS(quactlize_kpack_dispatch_prepare_grouped_dynamic_v1=Function(prepare),
                quactlize_kpack_dispatch_run_grouped_dynamic_v1=Function(run))
        def prefill(self,req,compute,channels,mask):
            assert (req.m,req.max_rows,compute,channels,mask)==(w.m*8,w.m,1,w.channels,3)
            return NS(policy=19 if fault=='policy' else 22,base=NS(route=int(route=='sf'),
                dequant_config=params['dequant_config']+(1 if fault=='helper' else 0)))
        def query_prefill(self,req,compute,channels):
            return Choice(policy=22,parent=b'wrong' if fault=='parent' else params['symbol'].encode(),
                build_key=b'a'*64,algorithm=launch['algorithm'],split=launch['split']+(1 if fault=='launch' else 0),
                grid=launch['grid'],workspace_bytes=128 if fault=='workspace' else 256,device=0,compute_units=72)
        def close(self):calls.append('dispatch-close')
    monkeypatch.setattr(selected,'Dispatch',Dispatcher)
    sf=NS(planes=[dict(low=500+i,high=600+i,units=700+i,zero=800+i) for i in range(2)]) if route=='sf' else None
    if fault:
        with pytest.raises(ValueError):selected.TensorCore(b,arm,dict(key='a'*64),query,sf)
        assert calls[-2:]==['dispatch-close','base-close']
        return
    core=selected.TensorCore(b,arm,dict(key='a'*64),query,sf)
    assert core.receipt['selection']=='PUBLIC_CPP_POLICY22'
    for i in range(2):
        core.a.ptr=10+i;core.out.ptr=20+i;core.buffers[-1].ptr=30+i;b.offsets.ptr=60+i;b.r.stream.value=80+i
        assert core.launch(i)==0
        plane=(sf.planes if sf else planes)[i]
        assert calls[-3:]==[('prepare',10+i),('run',10+i,20+i,30+i,60+i,
                            plane['low'],plane['units'],plane['zero'],80+i),('finish',20+i,True)]
    core.close();assert calls[-3:]==['sync','dispatch-close','base-close']


def comparison_fixture(native=10.,candidate=9.):
    w=Workload(12,512,2048,16,'indexed','bf16',experts=256,topk=8,router='spread')
    p=dict(id=w.key,workload=asdict(w));passes=[]
    for block,arm in enumerate(native_ab.gate.ORDER):
        cost=native if arm=='native' else candidate
        parts=[dict(role=role,node=dict(name=role),samples_us=[[v]*5 for _ in range(6)],
                    timing_method='ASYS_GPU_ACTIVITY') for role,v in [('prepare',1.),('producer',cost-2),('reducer',1.)]]
        passes.append(dict(arm=arm,block=block,components=parts,raw_sha256=['same-gguf'],
            timed_router_sha256=routing(w,0)['sha256'],timing_input_repeat=0,timed_fixture=expected(w,0),
            data=dict(native_activation_contract='NATIVE_DISPATCH_UNMODIFIED')))
    return p,passes


@pytest.mark.parametrize('fault',[None,'raw','ids','input','repeat','arithmetic','order'])
def test_abba_compares_same_inputs_and_component_means(fault):
    p,passes=comparison_fixture()
    if fault=='raw':passes[1]['raw_sha256']=['changed']
    elif fault=='ids':passes[1]['timed_router_sha256']='changed'
    elif fault=='input':passes[1]['timed_fixture']['a_crc32']^=1
    elif fault=='repeat':passes[1]['timing_input_repeat']=1
    elif fault=='arithmetic':passes[3]['data']['native_activation_contract']='changed'
    elif fault=='order':passes[1]['block']=0
    if fault:
        with pytest.raises(ValueError):native_ab.comparison(p,passes)
    else:
        result=native_ab.comparison(p,passes)
        assert result['status']=='PASS' and result['delta_pct']==pytest.approx(-10)
        assert result['quactlize_main_us']==7 and result['quactlize_reducer_us']==1
        assert result['quactlize_helper_us']==1 and not result['model_tpot_admitted']
        # Do not use the minimum or discard a valid loss as unstable.
        passes[1]['components'][1]['samples_us'][0][0]=100
        assert native_ab.comparison(p,passes)['status']=='REGRESSION'


def test_selected_pass_forces_fresh_helpers_and_preserves_loss_on_resume(tmp_path,monkeypatch):
    from tools import run_decode_refresh as base
    p=plan.wired_plan()['points'][0];args=NS(output=tmp_path,worker=0)
    run=dict(sha256='run',cards=['0'])
    result=tmp_path/'gpu-0/points'/(p['id']+'.json')
    base.sealed(result,dict(run='run',point=p['id'],status='REGRESSION',evidence={}))
    monkeypatch.setattr(native_ab.engine,'invocation',lambda *a,**k:pytest.fail('remeasured an existing loss'))
    assert native_ab.point(args,run,p,time.monotonic()+100)
    seen=[]
    def invoke(a,run,job,**kw):
        seen.append(job);raise RuntimeError('stop before GPU')
    monkeypatch.setattr(native_ab.engine,'invocation',invoke)
    with pytest.raises(RuntimeError,match='stop before GPU'):
        native_ab.candidate_pass(args,run,p,{},1)
    assert seen[0]['fresh_components'] and seen[0]['arms']==['original']


def test_ordinary_and_native_worker_routes_are_distinct(tmp_path,monkeypatch):
    from tools import run_moe_confirmation as runner
    p=plan.wired_plan()['points'][0]
    runner.base.sealed(tmp_path/'plan.json',dict(points=[p]))
    runner.base.sealed(tmp_path/'run.json',dict(native_comparison=True,cards=['0'],assignments=[[p['id']]],
                       devices=[dict(device=dict(pci='host-test-native-worker'))]))
    monkeypatch.setattr(runner,'verify',lambda *a:None)
    monkeypatch.setattr(runner,'point',lambda *a:pytest.fail('ordinary confirmation called'))
    seen=[];monkeypatch.setattr(native_ab,'point',lambda *a:seen.append(a[2]['id']) or True)
    assert runner.worker(NS(output=tmp_path,worker=0,preflight=None,budget_hours=1))==0
    assert seen==[p['id']]


def test_native_plan_declares_the_real_abba_denominator():
    p=plan.make_plan('wired-prefill',native_compare=True)
    assert p['native_comparison'] and p['counts']['native_passes']==p['counts']['candidate_passes']==400
    assert p['counts']['arms']==200 and p['counts']['tc_modules']==8
    assert p['protocol']['order']==list(native_ab.gate.ORDER) and not p['protocol']['helper_reuse']
    kinds=lambda p:(p['workload']['qtype'],p['workload']['channels'],p['arms'][0]['recipe']['route'])
    by_id={p['id']:p for p in p['points']}
    assert {kinds(by_id[i]) for i in plan.preflight_points(p)}=={kinds(x) for x in p['points']}
    with pytest.raises(ValueError):plan.make_plan('decode',native_compare=True)
