"""Exercise device-gate orchestration with a CPU fake, not GPU admission."""
import ctypes as C

import numpy as np

from dev.tp2_decode import finish_gate as gate
from quactlize.dispatch.native import IndexedIO


class HostMemory:
    def __init__(self):
        self.allocations=[]
        self.storage={}
        self.stream=C.c_void_p(1)

    def allocate(self, size):
        block=C.create_string_buffer(size)
        ptr=C.addressof(block)
        self.storage[ptr]=block
        self.allocations.append(ptr)
        return ptr

    def copy(self, ptr, values):
        values=np.ascontiguousarray(values)
        C.memmove(ptr,values.ctypes.data,values.nbytes)

    def fill(self, ptr, size, byte=0xa5):
        C.memset(ptr,byte,size)

    def download(self, ptr, size):
        return np.frombuffer(C.string_at(ptr,size),'u1').copy()

    def release_after(self, mark):
        for ptr in self.allocations[mark:]:del self.storage[ptr]
        del self.allocations[mark:]


class Replay:
    def __init__(self, rt, calls):
        self.calls=calls
        self.sample()  # Same excluded first replay as the real adapter.

    def sample(self):
        for call in self.calls:assert call()==0

    def close(self):
        pass


def test_snapshot_gate_fixture_runs_all_controls(monkeypatch):
    rt=HostMemory()
    invalid={}
    cast=lambda ptr,typ:C.cast(ptr,C.POINTER(typ)).contents

    def query(c, size):
        cast(size,C.c_uint64).value=4096
        return 0

    def bind(c, device, scratch, size, out):
        c=cast(c,gate.Call);p=cast(out,gate.Projection)
        p.version=1;p.size=C.sizeof(p);p.m=c.rows;p.n=c.n;p.k=c.k
        p.experts=256;p.tile_m=8;p.splits=1;p.device=0
        p.outputs=scratch;p.directory_header=scratch+2048
        p.workspace=scratch;p.workspace_bytes=size
        p.io=IndexedIO(1,C.sizeof(IndexedIO),c.rows//8,8,c.channels,0,
                       c.ids_stride,c.a_row_stride,c.a_token_stride,c.out_row_stride,
                       c.ids,c.a,c.output,scratch+3072)
        return 0

    def prepare(d, f, stream):
        d=cast(d,gate.MixedPlanV2);f=cast(f,gate.MoeFinish)
        p=d.plan;tokens=p.gate.io.tokens
        if p.router.version:
            scores=rt.download(p.router.logits,tokens*256*4).view('f4').reshape(tokens,256)
            with np.errstate(over='ignore', invalid='ignore'):
                ids=np.argsort(-scores,axis=1,kind='stable')[:,:8]
                vals=np.take_along_axis(scores,ids,axis=1).astype('f8')
                ex=np.exp(vals-vals.max(axis=1,keepdims=True))
                weights=(ex/ex.sum(axis=1,keepdims=True)*p.router.scale).astype('f4')
            for t in range(tokens):rt.copy(p.gate.io.ids+t*p.gate.io.ids_stride*4,ids[t].astype('i4'))
            rt.copy(p.router.weights,weights)
        ids=rt.download(p.gate.io.ids,tokens*p.gate.io.ids_stride*4).view('i4').reshape(tokens,-1)[:,:8]
        invalid[p.down.outputs]=bool(np.any((ids<0)|(ids>=256)))
        rows=rt.download(f.weights,tokens*f.weights_stride*4).view('f4').reshape(tokens,-1)[:,:8]
        rt.copy(p.down.outputs,rows.copy())
        return 0

    def finish(d, f, stream):
        d=cast(d,gate.MixedPlanV2);f=cast(f,gate.MoeFinish);p=d.plan.down
        tokens=p.io.tokens
        if invalid[p.outputs]:
            result=np.full((tokens,p.n),np.nan,'f4')
        else:
            values=rt.download(p.io.output,tokens*8*p.n*4).view('f4').reshape(tokens,8,p.n)
            if d.compute_type:values=gate.round_compute(values,'bf16')
            weights=rt.download(f.weights,tokens*f.weights_stride*4).view('f4').reshape(tokens,-1)
            result=np.multiply(values[:,0],weights[:,0,None],dtype='f4')
            for slot in range(1,8):
                result+=np.multiply(values[:,slot],weights[:,slot,None],dtype='f4')
        rt.copy(f.output,result)
        return 0

    entries={'quactlize_kpack_moe_simt_query_v1':query,'quactlize_kpack_moe_simt_bind_v1':bind,
             'quactlize_kpack_moe_prepare_finish_v1':prepare,'quactlize_kpack_moe_weighted_finish_v2':finish}
    monkeypatch.setattr(gate,'function',lambda lib,name,args:entries[name])
    monkeypatch.setattr(gate,'Graph',Replay)
    for compute in (0,1):
        for tokens in range(1,9):
            for alias in ('none','weights','weights-offset','ids'):
                for router in (False,True):
                    assert gate.case(rt,None,tokens,compute,alias,router)['status']=='PASS'
                    assert not rt.allocations


def test_focus_keeps_measured_minima_and_only_twenty_two_neighbors():
    from dev.tp2_decode import focus
    from dev.gemv_model.source import source
    from dev.gemv_model.access import access
    from tools.generate_decode_selection import EVIDENCE, effective
    import json
    evidence=json.loads(EVIDENCE.read_text());effective(evidence)
    winners={w['point']['name']:w for w in evidence['rows']}
    assert len(focus.POINTS)==3
    assert sum(len(focus.candidates(p)) for p in focus.POINTS)==22
    for p in focus.POINTS:
        configs=focus.candidates(p)
        body=winners[p.name]['body']
        assert all(getattr(configs[0],k)==v for k,v in body.items() if k!='name')
        assert len({c.name for c in configs})==len(configs)
        assert not p.tc and all(c.warps in (2,4,8) and c.values in (2,4,8) for c in configs)
        text=source(p,configs)
        assert 'simt::query_v2' in text and 'simt::buffers_v2' in text
        for c in configs:
            assert access(p,c,dict(A=0,low=0,high=0,units=0))['aggregate_source_footprint']


def test_private_model_trace_keeps_two_devices_and_same_tokens(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    import json
    from tools import run_kpack_model_validation as model
    from tools import kpack_asys_scope
    commands=[]
    def private(argv,temporary,inputs):
        commands.append((argv,temporary,inputs))
        return argv
    def run(argv,log,env):
        assert env['CUDA_VISIBLE_DEVICES']=='6,7' and env['TMPDIR']=='/tmp'
        assert 'PERFETTO_PRODUCER_SOCK_NAME' not in env
        out=Path(argv[argv.index('--output')+1]);out.mkdir()
        (out/'proof.json').write_text(json.dumps(dict(input_tokens_sha256='a'*64,missing_ops=[])))
    monkeypatch.setenv('PERFETTO_PRODUCER_SOCK_NAME','do-not-reuse')
    monkeypatch.setattr(kpack_asys_scope,'private_command',private)
    monkeypatch.setattr(model,'run',run)
    args=SimpleNamespace(llama=Path('/source'),build=Path('/build'),cache=tmp_path/'cache',
        bundle=Path('/bundle'),asys=Path('/asys'),inspector=Path('/inspect'),jit_cache=tmp_path/'jit',
        profiler_scope='private')
    records=model.traces(args,dict(name='model',path='/model.gguf',devices='6,7',split='tensor',
                                  tensor_split='1,1'),tmp_path,tmp_path/'inventory.json')
    assert len(records)==len(commands)==2
    assert commands[0][1]!=commands[1][1]
    assert '--proof-tokens' not in commands[0][0] and '--proof-tokens' in commands[1][0]


def test_focus_profile_accepts_only_exact_measured_entry():
    import pytest
    from dev.tp2_decode import focus
    from dev.gemv_model.run import profile_records
    p=focus.POINTS[0];c=focus.candidates(p)[0]
    name='void quactlize::execution::simt::measured_decode_kernel<8,0,256,3072,1,1,1,0,5,4,8,4,1,0,true,false>(qkg_call_v1)'
    csv='"ID","Kernel Name","Block Size","Grid Size"\n0,"'+name+'","(256,1,1)","(16,1,1)"\n'
    assert profile_records(csv,p,'incumbent',focus.candidates(p))
    with pytest.raises(ValueError,match='implementation'):
        profile_records(csv.replace(',true,false>',',false,false>'),p,'incumbent',focus.candidates(p))
    with pytest.raises(ValueError,match='geometry'):
        profile_records(csv.replace('(16,1,1)','(32,1,1)'),p,'incumbent',focus.candidates(p))
