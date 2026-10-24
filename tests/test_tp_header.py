"""CPU-only TP rules: real caller source, synthetic GGUF headers, no GPU weights."""
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from dev.prefill_asys.cohort import collect
from dev.prefill_asys.tp_header import (HeaderTP, CAPTURE_KEYS, parameters, runtime_tensors,
                                       read_headers, split_function, parse)
from tools.gguf_internal_shape_inventory import _synthetic_gguf, _encoded_string, read_gguf_header

ROOT=Path(__file__).resolve().parents[1]


def fixtures(arch):
    def t(name,q,k,n,*other):return dict(name=name,qtype=q,dims_gguf=[k,n,*other])
    if arch=='qwen35moe':
        meta=dict(block_count=9,nextn_predict_layers=1,embedding_length=2048,
            **{'attention.head_count':16,'attention.head_count_kv':2,'attention.key_length':128,
               'attention.value_length':128,'ssm.state_size':128,'ssm.conv_kernel':4,
               'ssm.inner_size':1536,'ssm.time_step_rank':12,'ssm.group_count':6})
        tensors=[]
        for layer in (0,1,4):
            tensors.extend([t(f'blk.{layer}.attn_qkv.weight',12,2048,3072),
                            t(f'blk.{layer}.ssm_out.weight',14,1536,2048)])
        tensors.extend([t('blk.0.ffn_up_shexp.weight',12,2048,1536),
                        t('blk.0.ffn_down_shexp.weight',14,1536,2048),
                        t('blk.3.attn_q.weight',12,2048,4096),t('blk.3.attn_k.weight',8,2048,256),
                        t('blk.3.attn_output.weight',12,2048,2048),
                        t('blk.8.attn_q.weight',12,2048,4096)])
    else:
        meta=dict(block_count=3,nextn_predict_layers=1,embedding_length=4096,
            expert_feed_forward_length=2048,**{'attention.head_count':48,'attention.head_count_kv':1,
            'attention.key_length':128,'attention.value_length':128,'attention.output_group_count':6,
            'attention.output_lora_rank':512,'hyper_connection.count':4})
        tensors=[]
        for layer in (0,1):
            tensors.extend([t(f'blk.{layer}.attn_q_b.weight',8,512,6144),
                            t(f'blk.{layer}.attn_output_a.weight',8,1024,3072),
                            t(f'blk.{layer}.attn_output_b.weight',12,3072,4096)])
        tensors.extend([t('blk.0.attn_q_a.weight',12,4096,512),
                        t('blk.0.ffn_up_shexp.weight',12,4096,2048),
                        t('blk.0.ffn_down_shexp.weight',14,2048,4096),
                        t('blk.2.nextn.eh_proj.weight',12,8192,4096)])
    return dict({'general.name':'synthetic-'+arch,'general.architecture':arch},
                **{arch+'.'+k:v for k,v in meta.items()}),tensors


@pytest.fixture(scope='module')
def helper(tmp_path_factory):
    source=os.environ.get('LLAMA_TP_SOURCE')
    if not source:
        pytest.skip('set LLAMA_TP_SOURCE for CPU execution of the actual caller TP function')
    return HeaderTP(source,tmp_path_factory.mktemp('tp-header-cpu'))


def run(helper,arch,world):
    meta,tensors=fixtures(arch)
    return helper(dict(name=arch,tp=world),[dict(header=dict(metadata=meta))],tensors)


def test_qwen_repeated_segments_and_layer_kind_rotation(helper):
    records,receipt,excluded=run(helper,'qwen35moe',2)
    def sizes(name,index):return [v[index] for v in sorted(records[name])]
    assert sizes('blk.0.attn_qkv.weight',2)==[1024,2048]
    assert sizes('blk.1.attn_qkv.weight',2)==[2048,1024]
    assert sizes('blk.4.attn_qkv.weight',2)==[2048,1024]  # rotation counts recurrent predecessors, not layer % TP
    assert sizes('blk.0.ssm_out.weight',3)==[512,1024]
    assert sizes('blk.0.ffn_up_shexp.weight',2)==[1536,1536]  # native Qwen rule: replicated shared expert
    assert sizes('blk.3.attn_q.weight',2)==[2048,2048]
    assert sizes('blk.3.attn_k.weight',2)==[128,128]
    assert excluded['blk.8.attn_q.weight']=='MTP_DISABLED_BY_DEFAULT'
    split=next(s for s in receipt['splits'] if s['name']=='blk.0.attn_qkv.weight')
    assert split['segments']==[dict(repeat=4,extents=[256,512])]
    assert not receipt['gpu_work'] and not receipt['weights_read']


def test_deepseek_groups_and_reshape_are_not_a_naive_division(helper):
    records,receipt,excluded=run(helper,'deepseek4',4)
    def sizes(name,index):return [v[index] for v in sorted(records[name])]
    assert sizes('blk.0.attn_q_b.weight',2)==[1024,2048,1024,2048]
    assert sizes('blk.1.attn_q_b.weight',2)==[2048,1024,2048,1024]
    assert sizes('blk.0.attn_output_b.weight',3)==[512,1024,512,1024]
    assert sizes('blk.0.attn_q_a.weight',2)==[512]*4
    assert sizes('blk.0.ffn_up_shexp.weight',2)==[512]*4
    assert sizes('blk.0.ffn_down_shexp.weight',3)==[512]*4
    assert excluded['blk.0.attn_output_a.weight']=='RUNTIME_GROUPED_OUTPUT_PROJECTION_NOT_2D_DENSE'
    assert 'blk.0.attn_output_a.weight' not in records
    assert excluded['blk.2.nextn.eh_proj.weight']=='MTP_DISABLED_BY_DEFAULT'


def test_source_hparams_methods_are_real_and_binary_has_no_gpu_runtime(helper):
    deps=subprocess.check_output(['ldd',str(helper.binary)],text=True)
    assert not any(x in deps.lower() for x in ('hggc','cuda','llama','torch','cublas'))
    receipt=json.loads((helper.build/'receipt.json').read_text())
    assert 'src/llama-hparams.cpp' in receipt['identity']['caller_sources']
    command=json.loads((helper.build/'command.json').read_text())
    assert str(helper.source/'src/llama-hparams.cpp') in command
    assert (helper.build/'tp-selector.inc').read_text()==split_function((helper.source/'src/llama-model.cpp').read_text())


def test_header_keys_and_bounded_arrays_do_not_read_weight_payload():
    prefix='qwen35moe.attention.recurrent_layers'
    data=_synthetic_gguf([('general.name','test'),('general.architecture','qwen35moe')],[])
    data=data[:16]+struct.pack('<Q',3)+data[24:]
    data+=_encoded_string(prefix)+struct.pack('<IIQ',9,7,4)+bytes((1,1,1,0))
    source=io.BytesIO(data+b'WEIGHT_BYTES_MUST_NOT_BE_READ')
    header=read_gguf_header(source,'fixture',capture_keys=CAPTURE_KEYS)
    assert header['metadata'][prefix]==[True,True,True,False]
    assert source.tell()==len(data)
    old=read_gguf_header(io.BytesIO(data),'fixture')
    assert prefix not in old['metadata']
    assert old['metadata_fingerprints']==header['metadata_fingerprints']
    oversized=data[:-4-16]+struct.pack('<IIQ',9,7,4097)
    with pytest.raises(ValueError,match='4096'):
        read_gguf_header(io.BytesIO(oversized),'oversize',capture_keys=CAPTURE_KEYS)


@pytest.mark.parametrize('fault',['arch','group','layers','heads','mask','reshape'])
def test_unknown_or_inconsistent_metadata_is_not_guessed(fault):
    meta,tensors=fixtures('deepseek4' if fault=='reshape' else 'qwen35moe')
    if fault=='arch':meta['general.architecture']='unknown'
    elif fault=='group':meta['qwen35moe.ssm.group_count']=5
    elif fault=='layers':meta['qwen35moe.block_count']=0
    elif fault=='heads':meta['qwen35moe.attention.head_count_kv']=3
    elif fault=='mask':meta['qwen35moe.attention.recurrent_layers']=[True]
    elif fault=='reshape':next(t for t in tensors if 'output_a' in t['name'])['dims_gguf'][1]+=1
    with pytest.raises(ValueError):runtime_tensors(tensors,parameters(meta,tensors))


def test_deepseek_missing_mtp_matches_loader_reset():
    meta,tensors=fixtures('deepseek4')
    tensors=[t for t in tensors if '.nextn.' not in t['name']]
    assert parameters(meta,tensors)['scalars']['n_layer_nextn']==0


def test_deepseek_output_reshape_uses_swa_head_size():
    meta,tensors=fixtures('deepseek4')
    meta['deepseek4.attention.key_length_swa']=256
    for t in tensors:
        if '.attn_output_a.' in t['name']:t['dims_gguf'][0]*=2
    runtime,_=runtime_tensors(tensors,parameters(meta,tensors))
    wo_a=next(t for t in runtime if '.attn_output_a.' in t['name'])
    assert wo_a['runtime_dims']==[2048,512,6,1]


def test_source_change_during_planning_is_rejected(helper,monkeypatch):
    from dev.prefill_asys import tp_header
    real_sha=tp_header.sha
    changed=helper.source/'src/llama-model.cpp'
    monkeypatch.setattr(tp_header,'sha',lambda p:'changed' if Path(p)==changed else real_sha(p))
    with pytest.raises(ValueError,match='sources changed'):run(helper,'qwen35moe',2)


def test_existing_observations_crosscheck_cpu_rules(helper,tmp_path):
    meta,tensors=fixtures('deepseek4')
    log=tmp_path/'observed.log'
    model=dict(name='deepseek4',tp=4,path='/metadata',tp_logs=[str(log)])
    args=dict(loader=lambda _:([dict(header=dict(metadata=meta))],tensors),partitioner=helper)
    line='[quactlize-dense] tensor=blk.0.attn_q_b.weight device=0 q=8 m=128 n={} k=512 route=0\n'
    log.write_text(line.format(1024))
    doc=collect([model],[128],**args)
    assert doc['status']=='READY'
    assert doc['tp_scope'].startswith('SOURCE_BOUND_CPU')
    assert any(t['reason']=='RUNTIME_GROUPED_OUTPUT_PROJECTION_NOT_2D_DENSE' for t in doc['excluded'])
    log.write_text(line.format(1536))
    with pytest.raises(ValueError,match='observed TP log differs'):collect([model],[128],**args)


def test_default_entry_plans_both_models_without_binary_sdk_or_four_gpus(helper,tmp_path):
    files=[]
    for arch in ('qwen35moe','deepseek4'):
        meta,tensors=fixtures(arch)
        path=tmp_path/(arch+'.gguf')
        path.write_bytes(_synthetic_gguf(list(meta.items()),[(t['name'],tuple(t['dims_gguf']),t['qtype']) for t in tensors]))
        files.append(path)
    output=tmp_path/'plan'
    result=subprocess.run([sys.executable,str(ROOT/'tools/run_tp_prefill_supplement.py'),
        '--qwen-model',str(files[0]),'--deepseek-model',str(files[1]),
        '--llama-source',str(helper.source),'--llama-build','/no-llama-binary','--sdk','/no-sdk',
        '--gpus','7','--m','128','--output',str(output)],text=True,capture_output=True)
    assert result.returncode==0, result.stdout+result.stderr
    plan=json.loads((output/'supplement.json').read_text())
    assert {m['name'] for m in plan['models']}=={'qwen35-122b-tp2','deepseek4-flash-tp4'}
    assert all(not m['static_partition']['gpu_work'] for m in plan['models'])
    assert all('output_a' not in c['tensor'] and '.nextn.' not in c['tensor']
               for p in plan['points'] for c in p['consumers'])
    assert 'DENSE_MODEL_AB_START' not in result.stdout and not (output/'timing').exists()
    assert not list(output.glob('*-inventory'))


def test_source_parser_does_not_take_partial_or_ambiguous_functions():
    with pytest.raises(ValueError):split_function('no function')
    body='struct ggml_backend_meta_split_state llama_meta_device_get_split_state(void*) {\nreturn split_state;\n}\n'
    assert split_function('prefix\n'+body+'suffix')==body
    with pytest.raises(ValueError):split_function(body+body)
