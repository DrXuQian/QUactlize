"""Host-only campaign contracts. No timings here are PPU evidence."""
import ctypes as C
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from dev.strategy.components import component_key, recommendation, role, statistics_checked
from dev.strategy.isolated import linear_order, stage_groups, KernelParams,EdgeData,default_edges,check_sdk_graph_abi
from dev.strategy.dense import Params, Info, Candidate
from dev.strategy.plan import make_plan
from quactlize.dispatch.table import Workload, Recipe
from tools.audit_kpack_strategy import audit
from tools.run_kpack_strategy import record_load, save


def r(split=4,tm=8):
    return Recipe.make('tc','fq',tm=tm,tn=64,tk=128,wm=8,wn=16,stages=2,split=split)


def node():
    return dict(name='reduce_decode_rows<4>',symbol='exact-symbol',grid=[16,1,1],block=[32,1,1],shared_bytes=0,split=4)


def key(w=None,recipe=None,n=None,**kwargs):
    return component_key(w or Workload(12,1024,512,1),recipe or r(),n or node(),
        authority=kwargs.get('authority','SDK_DEVICE_BINARY_A'),source='implementation-closure',
        cache=kwargs.get('cache','REPEATED_SCRATCH'))[0]


def test_reducer_dedup_does_not_follow_every_producer_config_or_qtype():
    w=Workload(12,1024,512,1)
    assert key(w,r(tm=8))==key(w,r(tm=16))
    assert key(w)==key(replace(w,qtype=14,k=1024,input_stride=0))
    for other in (replace(w,m=2),replace(w,n=512,output_stride=0),replace(w,compute='bf16'),
                  replace(w,storage='f16'),replace(w,output_stride=1032),replace(w,fusion='swiglu')):
        assert key(w)!=key(other)
    assert key(w)!=key(w,r(split=8))
    assert key(w)!=key(w,authority='OTHER_SDK')
    assert key(w)!=key(w,cache='ROTATING_PARTIALS')


def test_expansion_reuse_keeps_actual_expert_domain_precision_and_mapping():
    w=Workload(12,1024,512,128)
    n=dict(node(),name='dequant_full',symbol='exact-dequant',split=1)
    assert key(w,n=n)==key(replace(w,m=512),n=n)
    assert key(w,n=n)!=key(replace(w,compute='bf16'),n=n)
    g=Workload(12,1024,512,128,'indexed',experts=256,topk=8,router='real')
    assert key(g,n=n)!=key(replace(g,m=512),n=n)
    assert key(g,n=n)!=key(replace(g,router='cluster'),n=n)


def test_no_subtraction_no_extra_reducer_no_launch_formula():
    def sample(us):return statistics_checked([[us]*5 for _ in range(3)])
    records={'base':sample(10.),'producer':sample(6.),'reducer':sample(2.),'adapter':sample(1.)}
    result=recommendation({'original':['base'],'candidate':['producer','reducer','adapter']},records)
    assert result['costs_us']=={'original':10.,'candidate':9.}
    assert result['recommended']=='candidate'
    assert result['production_changed'] is False and result['host_launch_cost_added'] is False
    assert result['admission']=='ORIGINAL_WIRING_MODEL_AB_REQUIRED'
    # Reuse profiling, not eliminate a real second invocation.
    result=recommendation({'original':['base'],'candidate':['producer','reducer','adapter','adapter','adapter']},records)
    assert result['recommended']=='original' and result['costs_us']['candidate']==11.
    with pytest.raises(ValueError,match='baseline'):
        recommendation({'candidate':['producer']},records)
    result=recommendation({'original':['base'],'candidate':['producer','missing']},records)
    assert result['recommended']=='original' and result['missing']
    records['reducer']=dict(records['reducer'],scope='COMPLETE_CALL')
    result=recommendation({'original':['base'],'candidate':['producer','reducer']},records)
    assert result['recommended']=='original' and result['missing']


def test_noise_is_not_an_admitted_improvement():
    a=statistics_checked([[10.]*3 for _ in range(3)])
    b=statistics_checked([[8.]*3,[9.]*3,[8.]*3])
    result=recommendation({'original':['a'],'candidate':['b']},{'a':a,'b':b})
    assert result['recommended']=='original' and result['noisy']==['candidate']
    with pytest.raises(ValueError):statistics_checked([[float('nan')]*3 for _ in range(3)])


def test_graph_isolation_uses_dependency_order_not_enumeration_order():
    assert linear_order([4,2,3,1],[(1,2),(2,3),(3,4)])==[1,2,3,4]
    with pytest.raises(ValueError,match='nonlinear'):
        linear_order([1,2,3],[(1,2),(1,3)])
    with pytest.raises(ValueError):linear_order([1,2],[(1,2),(2,1)])
    nodes=[dict(symbol='producer'),dict(symbol='reducer')]*3
    assert stage_groups(nodes,3)==[[0,2,4],[1,3,5]]
    nodes[-1]={'symbol':'different'}
    with pytest.raises(ValueError):stage_groups(nodes,3)


def test_component_isolation_cannot_erase_pdl_or_nondefault_edges():
    edges=(EdgeData*2)()
    default_edges(edges)
    edges[0].type=1
    with pytest.raises(ValueError,match='PDL'):default_edges(edges)
    edges[0].type=0;edges[1].from_port=1
    with pytest.raises(ValueError,match='PDL'):default_edges(edges)


def test_unknown_kernels_cannot_disappear_from_costs():
    assert role('PpuMixedInputSplitKParallelM1FastReductionKernel')=='reducer'
    assert role('moe_chain_prepare_m1')=='prepare'
    assert role('convert_rows<true>')=='adapter'
    assert role('unrecognized_vendor_kernel')=='unclassified'


@pytest.mark.parametrize('kernel', [
    'GroupDirectoryMixedInputKernel', 'GemmUniversal', 'GemmUniversalMixedInputSplitKParallel',
    'PersistentMixedInputKernel', 'StreamKMixedInputKernel',
])
def test_grouped_tc_is_not_its_directory_or_metadata_template(kernel):
    name = ('void cutlass::device_kernel<cutlass::gemm::kernel::' + kernel +
            '<ProblemShape, Metadata, Dequant, MoeBlockDirectoryScheduler, SwiGLU>>(Params)')
    assert role(name) == 'producer'


@pytest.mark.parametrize(('name', 'expected'), [
    ('void quactlize::runtime::indexed_prepare<8, cutlass::gemm::kernel::GemmUniversal>(Params)', 'prepare'),
    ('void quactlize::runtime::grouped_device_metadata<cutlass::gemm::Gemm>(Params)', 'prepare'),
    ('void quactlize::moe_directory::build_kernel<8, 256>(Params)', 'prepare'),
    ('void cutlass::Kernel<cutlass::gemm::device::splitk_parallel::PpuMixedInputSplitKParallelM1FastReductionKernel<2,4>>(Params)', 'reducer'),
    ('void quactlize::runtime::reduce_bf16_grouped<2>(float const*, cutlass::bfloat16_t*, long)', 'reducer'),
    ('void quactlize::runtime::indexed_finish<2>(cutlass::gemm::kernel::GemmUniversal::Params)', 'postprocess'),
    ('void (anonymous namespace)::gather<float>(cutlass::gemm::kernel::GemmUniversal::Params)', 'adapter'),
    ('void sf_prepare<quactlize::gemm>(Params)', 'expansion'),
    ('void quactlize::q4_decode::kernel<1, 2>(Metadata)', 'producer'),
    ('void unknown_kernel(cutlass::gemm::kernel::GemmUniversal::Params)', 'unclassified'),
    ('void cutlass::device_kernel<UnknownKernel<cutlass::gemm::Gemm>>(Params)', 'unclassified'),
])
def test_component_role_uses_callable_not_argument_substrings(name, expected):
    assert role(name) == expected


def test_grouped_captured_inventory_retains_one_producer_and_actual_split_finish():
    from dev.strategy.components import stage_role
    names = ['void quactlize::runtime::indexed_prepare<8>(Params)',
             'void cutlass::device_kernel<cutlass::gemm::kernel::GroupDirectoryMixedInputKernel<Shape, Metadata>>(Params)',
             'void quactlize::runtime::indexed_finish<2>(Params)']
    assert [stage_role(n, 1) for n in names] == ['prepare', 'producer', 'postprocess']
    assert [stage_role(n, 2) for n in names] == ['prepare', 'producer', 'reducer_finish']


def test_historical_audit_has_scope_downgrades_not_deletions():
    report=audit()
    assert report['counts']['observations']==7237
    assert report['counts']['scopes']['PROPOSAL']==305
    assert report['policy']['raw_files_deleted'] is False
    assert report['policy']['default_selection_changed'] is False
    assert report['open']['router_sensitive']==88
    assert report['open']['raw_matched_missing_points']==16
    assert 'FULL_CALL_MINUS_INDEPENDENT_REDUCER_AS_PRODUCER_TIME' in report['excluded_as_latency']


def test_pilot_is_bounded_retains_original_and_never_small_m_full_dequant():
    p=make_plan('pilot')
    assert len(p['points'])==6 and p['original_baseline_mandatory']
    assert all(x['baseline']=='original' for x in p['points'])
    for point in p['points']:
        if point['workload']['m']<=8:
            assert all(r['kind']!='full' for r in point['candidates'])
    assert p['measurement']=='ISOLATED_KERNEL_COMPONENTS_ONLY'


def test_corrupt_component_is_never_reused(tmp_path):
    from quactlize.runtime.tuning import digest
    path=tmp_path/'record.json';identity={'kernel':'same'}
    data=statistics_checked([[1.]*3 for _ in range(3)])|dict(identity=identity)
    data['record_sha256']=digest(data)
    save(path,data)
    assert record_load(path,identity)['median_us']==1.
    with pytest.raises(ValueError):record_load(path,{'kernel':'changed'})
    data['median_us']=.5
    save(path,data)
    before=path.read_bytes()
    with pytest.raises(ValueError):record_load(path,identity)
    assert path.read_bytes()==before


def test_ctypes_abis_match_real_headers(tmp_path):
    root=Path(__file__).resolve().parents[1]
    exe=tmp_path/'abi'
    source='''
#include <cstdio>
#include "quactlize/dense/api.h"
#include <hggc_runtime_api.h>
int main() { printf("%zu %zu %zu %zu\\n",sizeof(quactlize_ppu_dense_params_v1),
  sizeof(quactlize_ppu_dense_plan_info_v1),sizeof(quactlize_ppu_dense_candidate_v1),sizeof(hggcKernelNodeParams)); }
'''
    subprocess.run(['g++','-x','c++','-std=c++17','-I'+str(root),'-I'+str(root/'quactlize/include'),
                    '-I/root/ppu-sdk/2.1.1/include','-','-o',str(exe)],input=source,text=True,check=True)
    got=list(map(int,subprocess.check_output([str(exe)],text=True).split()))
    assert got==[C.sizeof(x) for x in (Params,Info,Candidate,KernelParams)]
    assert check_sdk_graph_abi('/root/ppu-sdk/2.1.1')['edges_arguments']==5
