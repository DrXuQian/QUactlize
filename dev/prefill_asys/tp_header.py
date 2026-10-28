"""Read headers, execute the caller's TP split function on CPU, never load weights."""
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from dev.prefill_asys.cohort import headers, local_shapes
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import QTYPES
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.gguf_internal_shape_inventory import load_format_registry, qtype_traits

HERE = Path(__file__).resolve().parent
ARCHES = ('qwen35moe', 'deepseek4')
SUFFIXES = ('block_count', 'embedding_length', 'feed_forward_length', 'expert_feed_forward_length',
    'nextn_predict_layers', 'full_attention_interval', 'attention.head_count', 'attention.head_count_kv',
    'attention.key_length', 'attention.value_length', 'attention.key_length_swa', 'attention.value_length_swa',
    'attention.recurrent_layers', 'attention.output_group_count', 'attention.output_lora_rank',
    'hyper_connection.count', 'ssm.conv_kernel', 'ssm.inner_size', 'ssm.state_size',
    'ssm.time_step_rank', 'ssm.group_count')
CAPTURE_KEYS = {arch+'.'+key for arch in ARCHES for key in SUFFIXES}
FIELDS = ('n_layer_all','n_layer_nextn','n_embd','n_ff_exp','ssm_d_conv','ssm_d_inner',
    'ssm_d_state','ssm_dt_rank','ssm_n_group','dsv4_o_group_count','dsv4_o_lora_rank',
    'dsv4_hc_mult','n_embd_head_k_full','n_embd_head_v_full','n_embd_head_k_swa','n_embd_head_v_swa')
SOURCE_FILES = ('src/llama-model.cpp','src/llama-hparams.cpp','src/llama-hparams.h',
    'src/llama-arch.h','src/llama-arch.cpp','include/llama.h','ggml/include/ggml.h',
    'ggml/include/ggml-backend.h','src/models/qwen35moe.cpp','src/models/deepseek4.cpp')


def read_headers(path):
    return headers(path,capture_keys=CAPTURE_KEYS)


def split_function(text):
    signature = 'struct ggml_backend_meta_split_state llama_meta_device_get_split_state('
    if text.count(signature) != 1:
        raise ValueError('caller TP split function missing/ambiguous; no model run fallback')
    start = text.index(signature)
    end = text.index('\n}\n', start)+3
    result = text[start:end]
    if 'return split_state;' not in result:
        raise ValueError('incomplete caller TP function extraction')
    return result


def uint(value, key, minimum=0):
    if type(value) is not int or not minimum <= value <= 0x7fffffff:
        raise ValueError(f'invalid/missing GGUF TP metadata {key}: {value!r}')
    return value


def parameters(metadata, tensors):
    arch = metadata['general.architecture']
    if arch not in ARCHES:
        raise ValueError('header-only TP currently supports qwen35moe/deepseek4, got '+str(arch))
    def get(key, default=None, minimum=0):
        return uint(metadata.get(arch+'.'+key,default),key,minimum)
    count = get('block_count',minimum=1)
    if count > 512:
        raise ValueError('header-only TP exceeds LLAMA_MAX_LAYERS')
    nextn = get('nextn_predict_layers',0)
    if arch=='deepseek4' and 0 < nextn < count and not any(
            t['name']==f'blk.{count-nextn}.nextn.eh_proj.weight' for t in tensors):
        nextn = 0  # Same missing-MTP-tensor rule as load_arch_hparams.
    if nextn >= count:
        raise ValueError('invalid nextn layer count')
    def array(key, default):
        value = metadata.get(arch+'.'+key,default)
        rows = value if isinstance(value,list) else [value]*count
        if len(rows) != count:
            raise ValueError('per-layer GGUF metadata length differs: '+key)
        return [uint(v,key) for v in rows]
    heads = array('attention.head_count',0)
    kv = array('attention.head_count_kv',heads)
    ff = array('feed_forward_length',0)
    if any(not h or not k or h%k for h,k in zip(heads,kv)):
        raise ValueError('unsupported attention head counts for static TP')
    embd = get('embedding_length',minimum=1)
    key = get('attention.key_length',embd//heads[0],1)
    value = get('attention.value_length',embd//heads[0],1)
    h = dict.fromkeys(FIELDS,0)
    h.update(n_layer_all=count,n_layer_nextn=nextn,n_embd=embd,
        n_ff_exp=get('expert_feed_forward_length',0),n_embd_head_k_full=key,n_embd_head_v_full=value,
        n_embd_head_k_swa=get('attention.key_length_swa',key,1),
        n_embd_head_v_swa=get('attention.value_length_swa',value,1))
    recurrent = [False]*count
    if arch=='qwen35moe':
        for field,key in (('ssm_d_conv','conv_kernel'),('ssm_d_inner','inner_size'),
                          ('ssm_d_state','state_size'),('ssm_dt_rank','time_step_rank'),('ssm_n_group','group_count')):
            h[field] = get('ssm.'+key,minimum=1)
        if h['ssm_dt_rank'] % h['ssm_n_group']:
            raise ValueError('unsupported Qwen head broadcast ratio')
        rec = metadata.get(arch+'.attention.recurrent_layers')
        if rec is None:
            interval = get('full_attention_interval',4,1)
            recurrent = [i<count-nextn and (i+1)%interval!=0 for i in range(count)]
        else:
            rec = rec if isinstance(rec,list) else [rec]*count
            if len(rec)!=count or any(type(v) not in (int,bool) or v not in (0,1) for v in rec):
                raise ValueError('invalid per-layer recurrent mask')
            recurrent = list(map(bool,rec))
    else:
        h.update(dsv4_o_group_count=get('attention.output_group_count',minimum=1),
                 dsv4_o_lora_rank=get('attention.output_lora_rank',minimum=1),
                 dsv4_hc_mult=get('hyper_connection.count',minimum=1))
        if any(head%h['dsv4_o_group_count'] for head in heads):
            raise ValueError('DeepSeek heads do not form complete output groups')
    per_layer = [dict(heads=heads[i],kv=kv[i],ff=ff[i],recurrent=int(recurrent[i]),swa=int(arch=='deepseek4'))
                 for i in range(count)]
    return dict(architecture=arch,scalars=h,layers=per_layer)


def runtime_tensors(tensors, params, *, indexed=False):
    """Only metadata reshapes/exclusions; no packed bytes or weight allocation."""
    h=params['scalars']; result=[]; excluded={}
    for tensor in tensors:
        t=dict(tensor);name=t['name'];dims=list(t['dims_gguf'])
        layer=re.fullmatch(r'blk\.(\d+)\..+',name)
        if layer and int(layer[1]) >= h['n_layer_all']-h['n_layer_nextn']:
            excluded[name]='MTP_DISABLED_BY_DEFAULT'
        if params['architecture']=='deepseek4' and re.fullmatch(r'blk\.\d+\.attn_output_a\.weight',name):
            groups=h['dsv4_o_group_count'];rank=h['dsv4_o_lora_rank']
            width=params['layers'][0]['heads']*h['n_embd_head_k_swa']//groups
            if dims not in ([width,rank*groups],[width,rank,groups]):
                raise ValueError('DeepSeek grouped output metadata/reshape differs: '+name)
            dims=[width,rank,groups]
            excluded[name]='RUNTIME_GROUPED_OUTPUT_PROJECTION_NOT_2D_DENSE'
        if not 1<=len(dims)<=4:
            raise ValueError('invalid GGUF rank: '+name)
        t['runtime_dims']=dims+[1]*(4-len(dims))
        if indexed:
            from quactlize.gguf_roles import match_role
            role = match_role(name, len(dims)) if 'exps' in name else None
            matrix = role is not None and role[0].operation == 'MUL_MAT_ID'
        else:
            matrix = len(dims) == 2
        t['target']=bool(name not in excluded and layer and name.endswith('.weight') and
                         matrix and t['qtype'] in QTYPES)
        result.append(t)
    return result,excluded


def encode(params, tensors, world):
    registry=load_format_registry()
    lines=['QZ_TP_HEADER_V1 '+params['architecture']+' '+str(world),
           ' '.join(str(params['scalars'][key]) for key in FIELDS)]
    lines += [' '.join(str(layer[k]) for k in ('heads','kv','ff','recurrent','swa')) for layer in params['layers']]
    lines.append(str(len(tensors)))
    for t in tensors:
        block=qtype_traits(t['qtype'],registry)['storage_block_k']
        lines.append(json.dumps(t['name'])+' '+' '.join(map(str,
            [t['qtype'],block,int(t['target']),*t['runtime_dims']])))
    return '\n'.join(lines)+'\n'


def parse(text, tensors, world):
    expected={t['name']:t for t in tensors if t['target']}
    rows=[json.loads(line) for line in text.splitlines()]
    if len(rows)!=len(expected) or {r['name'] for r in rows}!=set(expected):
        raise ValueError('CPU TP split coverage differs')
    records=defaultdict(set)
    for r in rows:
        t=expected[r['name']];axis=r['axis']
        if r['qtype']!=t['qtype'] or axis not in (0,1,10) or not 1<=len(r['segments'])<=16:
            raise ValueError('CPU TP split type/axis differs')
        extents=[0]*world
        for s in r['segments']:
            if type(s['repeat']) is not int or s['repeat']<=0 or len(s['extents'])!=world:
                raise ValueError('CPU TP split segment coverage differs')
            for i,n in enumerate(s['extents']):
                if type(n) is not int or n<0:
                    raise ValueError('negative/invalid CPU TP extent')
                extents[i]+=n*s['repeat']
        k,n=t['runtime_dims'][:2]
        if axis!=10 and sum(extents)!=(k if axis==0 else n):
            raise ValueError('CPU TP split extent sum differs: '+t['name'])
        for rank,e in enumerate(extents):
            records[t['name']].add((rank,t['qtype'],e if axis==1 else n,e if axis==0 else k))
        # N/K tensor parallelism leaves the expert dimension replicated.
        # axis=2 (expert parallelism) is deliberately not accepted above.
        matrix = dict(t, dims_gguf=t['runtime_dims'][:2])
        if local_shapes(matrix,world,records)[1]:
            raise ValueError('CPU TP split does not describe a full local matrix')
    return records,rows


class HeaderTP:
    def __init__(self, llama_source, output, *, indexed=False):
        self.source=Path(llama_source).resolve(strict=True)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.indexed=indexed
        hashes={name:sha(self.source/name) for name in SOURCE_FILES}
        body=split_function((self.source/'src/llama-model.cpp').read_text())
        self.identity=dict(schema='quactlize.tp-header-cpu.v1',caller_sources=hashes,
            function_sha256=digest(body),carrier_sha256=sha(HERE/'tp_header_host.cpp'),
            metadata_binding_sha256=sha(__file__),storage_traits_sha256=sha(
                HERE.parents[1]/'tools/gguf_internal_shape_inventory.py'),tensor_split='EQUAL_EXPLICIT',
            device_work=False,weights_read=False,scope='SOURCE_BOUND_CPU_METADATA_NOT_RUNTIME_OBSERVATION')
        if indexed:
            self.identity.update(operation='MUL_MAT_ID', expert_partition='REPLICATED',
                                 role_source_sha256=sha(HERE.parents[1]/'quactlize/gguf_roles.py'))
        key=digest(self.identity);self.build=self.output/('host-'+key[:16])
        self.binary=self.build/'tp-header'
        receipt=self.build/'receipt.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['identity']!=self.identity or old['binary_sha256']!=sha(self.binary):
                raise ValueError('CPU TP helper receipt changed')
        else:
            self.build.mkdir(exist_ok=True)
            (self.build/'tp-selector.inc').write_text(body)
            command=['g++','-std=c++17','-O2','-ffunction-sections','-fdata-sections','-Wl,--gc-sections',
                '-I'+str(self.build),'-I'+str(self.source/'src'),'-I'+str(self.source/'include'),
                '-I'+str(self.source/'ggml/include'),str(HERE/'tp_header_host.cpp'),
                str(self.source/'src/llama-hparams.cpp'),'-o',str(self.binary)]
            atomic_json(self.build/'command.json',command)
            with (self.build/'build.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise ValueError('CPU TP helper build failed; see '+str(self.build/'build.log'))
            self.check_sources()
            atomic_json(receipt,dict(identity=self.identity,binary_sha256=sha(self.binary)))

    def check_sources(self):
        if {name:sha(self.source/name) for name in SOURCE_FILES} != self.identity['caller_sources']:
            raise ValueError('caller TP sources changed during planning; preserve this run and retry')

    def __call__(self, model, rows, tensors):
        self.check_sources()
        world=model['tp']
        if world not in (1,2,4):
            raise ValueError('header-only supplement supports TP1/TP2/TP4 equal splits')
        params=parameters(rows[0]['header']['metadata'],tensors)
        runtime,excluded=runtime_tensors(tensors,params,indexed=self.indexed)
        payload=encode(params,runtime,world)
        result=subprocess.run([str(self.binary)],input=payload,text=True,capture_output=True)
        path=self.output/(model['name']+'-cpu.json')
        if result.returncode:
            atomic_json(path,dict(status='FAIL',error=result.stderr,identity=self.identity))
            raise ValueError('CPU TP split failed: '+result.stderr[-4000:])
        records,splits=parse(result.stdout,runtime,world)
        receipt=dict(status='PASS',identity=self.identity,world=world,parameters=params,
            inputs_sha256=digest(payload),splits=splits,excluded=excluded,
            model_headers_sha256=digest(rows),gpu_work=False,weights_read=False)
        atomic_json(path,receipt)
        print('TP_HEADER_DONE '+json.dumps(dict(model=model['name'],world=world,
            weights=len(splits),excluded=len(excluded),gpu_work=False,weights_read=False,receipt=str(path))),flush=True)
        return records,receipt,excluded
