"""Caller-owned serving ops. Import alone does not load a DSO or touch a device.

Prepare Serving/Matmul before compile/capture; capture on an explicit stream.
No binding allocates output or scratch. Exported custom ops mutate those buffers;
Rust can replay the resulting graph without this module. Keep owners and all
captured buffers alive. close() requires drained work and destroyed graphs.
"""
import ctypes as C
from pathlib import Path
from typing import Optional
import torch
from .compat import serving_native as abi
from .compat.native import Problem, Sizes, Buffers, Merge

# Host library/plan owners only, NEVER device scratch. No lazy loading in ops.
_libraries = {}
_plans = {}
_serial = 0


def _key():
    global _serial
    _serial += 1
    return _serial


def _ptr(t):
    return t.data_ptr() if t is not None and t.numel() else None


def _bytes(t):
    if t is None or not t.numel():
        return 0
    return (1 + sum((n-1)*s for n,s in zip(t.shape,t.stride()))) * t.element_size()


def _dtype(t):
    if t.dtype == torch.float16: return 0
    if t.dtype == torch.bfloat16: return 1
    if t.dtype == torch.float32: return 2
    raise ValueError('expected F16/BF16/F32')


def _matrix(t, dtype=None):
    if t.ndim != 2 or t.stride(1) != 1 or t.stride(0) < t.shape[1]:
        raise ValueError('expected a non-overlapping row-strided matrix')
    if dtype is not None and t.dtype != dtype: raise ValueError('wrong tensor dtype')


def _scalar(t):
    if t is not None and (t.numel() != 1 or t.dtype != torch.int32):
        raise ValueError('count/status must be one device int32')


def _context(key, tensors):
    owner = _libraries[key] # missing preparation is an error, never lazy load
    for t in tensors:
        if t is not None and (t.device.type != 'cuda' or t.device.index != owner.device):
            raise ValueError('all serving tensors must belong to the prepared device')
    stream = torch.cuda.current_stream(owner.device).cuda_stream
    if not stream: raise ValueError('use an explicit non-default stream')
    return owner.lib, stream


class Serving:
    def __init__(self, library, device=0, stream=None):
        self.device = int(device)
        if torch.cuda.is_current_stream_capturing(): raise ValueError('prepare before capture')
        self.lib = abi.Library(Path(library).resolve(strict=True))
        with torch.cuda.device(self.device):
            stream = stream or torch.cuda.current_stream(self.device)
            self.lib.check(self.lib.prepare(self.device, stream.cuda_stream))
        self.key = _key()
        _libraries[self.key] = self

    def close(self):
        """Does not synchronize. Caller must drain work/destroy graphs first."""
        _libraries.pop(self.key, None)

    def router(self, logits, selected, ids, weights, error, count=None, *, multiplier=1.,
               score_func='softmax', renormalize=True, correction_bias=None):
        if score_func != 'softmax' or not renormalize or correction_bias is not None:
            raise ValueError('only selected-logit softmax/renormalize=True/no bias is supported')
        router(logits, selected, ids, weights, error, count, float(multiplier), self.key)

    def swiglu(self, gate, output, up=None, *, round_silu=True):
        swiglu(gate, up, output, round_silu, self.key)

    def gather_rows(self, raw, ids, output, error, *, qtype, table_rows, count=None):
        gather_rows(raw, ids, output, count, error, qtype, table_rows, self.key)

    def merge(self, input, weights, output):
        merge(input, weights, output, self.key)


@torch.library.custom_op('quactlize_serving::router', mutates_args=('selected','ids','weights','error'))
def router(logits: torch.Tensor, selected: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
           error: torch.Tensor, count: Optional[torch.Tensor], multiplier: float, key: int) -> None:
    _router_shapes(logits,selected,ids,weights,error,count)
    lib,stream=_context(key,(logits,selected,ids,weights,error,count))
    p=abi.Router(2,C.sizeof(abi.Router),logits.shape[0],logits.shape[1],ids.shape[1],_dtype(logits),0,1,
        multiplier,0,logits.stride(0),selected.stride(0),ids.stride(0),weights.stride(0),
        _bytes(logits),_bytes(selected),_bytes(ids),_bytes(weights),_ptr(logits),_ptr(selected),_ptr(ids),
        _ptr(weights),_ptr(count),_ptr(error),stream)
    lib.check(lib.router(C.byref(p)))


def _router_shapes(logits,selected,ids,weights,error,count):
    _matrix(logits); _matrix(selected,torch.float32); _matrix(ids,torch.int32); _matrix(weights,torch.float32)
    _scalar(error); _scalar(count)
    if logits.dtype not in (torch.float32,torch.bfloat16): raise ValueError('router requires BF16/F32 logits')
    if selected.shape != ids.shape or weights.shape != ids.shape or logits.shape[0] != ids.shape[0]:
        raise ValueError('router output geometry differs')


@router.register_fake
def _router_fake(logits,selected,ids,weights,error,count,multiplier,key):
    _router_shapes(logits,selected,ids,weights,error,count)


@torch.library.custom_op('quactlize_serving::swiglu', mutates_args=('output',))
def swiglu(gate: torch.Tensor, up: Optional[torch.Tensor], output: torch.Tensor,
           round_silu: bool, key: int) -> None:
    _swiglu_shapes(gate,up,output)
    lib,stream=_context(key,(gate,up,output))
    p=abi.SwiGLU(2,C.sizeof(abi.SwiGLU),output.shape[0],output.shape[1],_dtype(output),int(up is None),
        int(round_silu),0,gate.stride(0),up.stride(0) if up is not None else 0,output.stride(0),
        _bytes(gate),_bytes(up),_bytes(output),_ptr(gate),_ptr(up),_ptr(output),stream)
    lib.check(lib.swiglu(C.byref(p)))


def _swiglu_shapes(gate,up,output):
    _matrix(gate); _matrix(output)
    if output.dtype not in (torch.float16,torch.bfloat16) or gate.dtype != output.dtype:
        raise ValueError('SwiGLU requires matching F16/BF16')
    if gate.shape != (output.shape[0],output.shape[1]*(2 if up is None else 1)):
        raise ValueError('SwiGLU input geometry differs')
    if up is not None:
        _matrix(up,output.dtype)
        if up.shape != output.shape: raise ValueError('SwiGLU up geometry differs')


@swiglu.register_fake
def _swiglu_fake(gate,up,output,round_silu,key): _swiglu_shapes(gate,up,output)


@torch.library.custom_op('quactlize_serving::gather_rows', mutates_args=('output','error'))
def gather_rows(raw: torch.Tensor, ids: torch.Tensor, output: torch.Tensor,
                count: Optional[torch.Tensor], error: torch.Tensor, qtype: int, table_rows: int, key: int) -> None:
    _rows_shapes(raw,ids,output,count,error,qtype,table_rows)
    lib,stream=_context(key,(raw,ids,output,count,error))
    p=abi.Rows(1,C.sizeof(abi.Rows),qtype,_dtype(output),output.shape[0],output.shape[1],table_rows,
        _bytes(raw),_bytes(ids),_bytes(output),_ptr(raw),_ptr(ids),_ptr(output),_ptr(count),_ptr(error),stream)
    lib.check(lib.rows(C.byref(p)))


def _rows_shapes(raw,ids,output,count,error,qtype,table_rows):
    _matrix(output); _scalar(count); _scalar(error)
    if raw.dtype != torch.uint8 or not raw.is_contiguous(): raise ValueError('raw GGUF must be contiguous bytes')
    if ids.dtype != torch.int64 or ids.ndim != 1 or not ids.is_contiguous(): raise ValueError('IDs must be I64[capacity]')
    if output.dtype not in (torch.float16,torch.bfloat16) or not output.is_contiguous():
        raise ValueError('row lookup requires contiguous native16 output')
    if output.shape[0] != ids.shape[0] or output.shape[1]%32 or qtype not in (8,20) or table_rows < 0:
        raise ValueError('invalid raw row lookup geometry/format')
    if raw.numel() < table_rows*(output.shape[1]//32)*(34 if qtype==8 else 18):
        raise ValueError('raw table capacity too small')


@gather_rows.register_fake
def _rows_fake(raw,ids,output,count,error,qtype,table_rows,key):
    _rows_shapes(raw,ids,output,count,error,qtype,table_rows)


@torch.library.custom_op('quactlize_serving::merge', mutates_args=('output',))
def merge(input: torch.Tensor, weights: torch.Tensor, output: torch.Tensor, key: int) -> None:
    _merge_shapes(input,weights,output)
    lib,stream=_context(key,(input,weights,output))
    p=Merge(1,C.sizeof(Merge),input.shape[0],input.shape[1],input.shape[2],_dtype(input),
        input.stride(1),weights.stride(0),output.stride(0),_ptr(input),_ptr(weights),_ptr(output),stream)
    lib.check(lib.merge(C.byref(p)))


def _merge_shapes(input,weights,output):
    _matrix(weights,torch.float32); _matrix(output,torch.float32)
    if input.ndim != 3 or input.dtype not in (torch.float16,torch.bfloat16) or input.stride(2)!=1:
        raise ValueError('merge requires native16 [M,topk,N]')
    if input.stride(1)<input.shape[2] or input.stride(0)!=input.shape[1]*input.stride(1):
        raise ValueError('merge requires fixed slot stride')
    if weights.shape != input.shape[:2] or output.shape != (input.shape[0],input.shape[2]):
        raise ValueError('merge geometry differs')


@merge.register_fake
def _merge_fake(input,weights,output,key): _merge_shapes(input,weights,output)


class Matmul:
    """Thin owner of existing PPU matmul ABI. All tensor storage is external.

    query() is host-only; construct with valid initialized buffers outside
    capture. prepare overwrites outputs/scratch and synchronizes its stream.
    The handle contains no tensor pointers and calls can rebind them.
    """
    def __init__(self, serving, problem, *, input, output, workspace, error, ids=None,
                 raw=None, low=None, high=None, units=None, arrangement=None):
        if torch.cuda.is_current_stream_capturing(): raise ValueError('prepare matmul before capture')
        self.owner=serving
        self.problem=Problem.from_buffer_copy(problem)
        self.handle=C.c_void_p()
        self._bind()
        sizes=Sizes(); self._check(self._query(C.byref(problem),C.byref(arrangement) if arrangement is not None else None,C.byref(sizes)))
        self.sizes=sizes
        b=self._buffers(input,output,workspace,error,ids,raw,low,high,units)
        self._check(self._prepare(C.byref(problem),C.byref(arrangement) if arrangement is not None else None,C.byref(b),C.byref(self.handle)))
        self.key=_key(); _plans[self.key]=self

    def _bind(self):
        lib=self.owner.lib.lib
        # Each binary exports its own matmul implementation, not a runtime
        # cross-device fallback. CUDA compatibility is an explicitly built DSO.
        prefix='quactlize_ppu_matmul_' if hasattr(lib,'quactlize_ppu_matmul_run_v1') else 'quactlize_cuda_matmul_'
        for name,args,result in (
            ('query',[C.POINTER(Problem),C.c_void_p,C.POINTER(Sizes)],C.c_int),
            ('prepare',[C.POINTER(Problem),C.c_void_p,C.POINTER(Buffers),C.POINTER(C.c_void_p)],C.c_int),
            ('run',[C.c_void_p,C.POINTER(Buffers)],C.c_int),('destroy',[C.c_void_p],None),('error',[],C.c_char_p)):
            f=getattr(lib,prefix+name+'_v1'); f.argtypes,f.restype=args,result; setattr(self,'_'+name,f)

    @staticmethod
    def query(serving, problem, arrangement=None):
        plan=object.__new__(Matmul);plan.owner=serving;plan._bind();s=Sizes()
        plan._check(plan._query(C.byref(problem),C.byref(arrangement) if arrangement is not None else None,C.byref(s)))
        return s

    def _check(self,rc):
        if rc: raise RuntimeError(f'matmul rc={rc}: {self._error().decode(errors="replace")}')

    def _buffers(self,input,output,workspace,error,ids,raw,low,high,units):
        tensors=(input,output,workspace,error,ids,raw,low,high,units)
        _,stream=_context(self.owner.key,tensors)
        p=self.problem
        if _dtype(input)!=p.dtype or output.dtype!=input.dtype: raise ValueError('matmul native dtype differs')
        if input.ndim!=3 or input.shape!=(p.m,p.channels,p.k) or input.stride(2)!=1:
            raise ValueError('matmul input shape must be [M,channels,K]')
        if input.stride(0)!=p.input_token_stride or input.stride(1)!=p.input_row_stride:
            raise ValueError('matmul input strides differ from prepared problem')
        if output.shape!=(p.m,p.topk,p.n) or output.stride()!=(p.topk*p.output_stride,p.output_stride,1):
            raise ValueError('matmul output strides differ')
        if ids is not None:
            _matrix(ids,torch.int32)
            if ids.shape!=(p.m,p.topk) or ids.stride(0)!=p.ids_stride: raise ValueError('matmul IDs differ')
        for t in (workspace,raw,low,high,units):
            if t is not None and (t.dtype!=torch.uint8 or not t.is_contiguous()):
                raise ValueError('matmul byte planes/workspace must be contiguous uint8')
        _scalar(error)
        return Buffers(1,C.sizeof(Buffers),_ptr(input),_ptr(raw),_ptr(low),_ptr(high),_ptr(units),_ptr(ids),
            _ptr(output),_ptr(workspace),_bytes(input),_bytes(raw),_bytes(low),_bytes(high),_bytes(units),
            _bytes(ids),_bytes(output),_bytes(workspace),_ptr(error),stream)

    def __call__(self,input,output,workspace,error,*,ids=None,raw=None,low=None,high=None,units=None):
        matmul(input,raw,low,high,units,ids,output,workspace,error,self.key)

    def close(self):
        """Destroy only after graph destruction and completion, never during capture."""
        if self.handle.value: self._destroy(self.handle); self.handle=C.c_void_p()
        _plans.pop(self.key,None)


@torch.library.custom_op('quactlize_serving::matmul',mutates_args=('output','workspace','error'))
def matmul(input: torch.Tensor, raw: Optional[torch.Tensor], low: Optional[torch.Tensor],
           high: Optional[torch.Tensor], units: Optional[torch.Tensor], ids: Optional[torch.Tensor],
           output: torch.Tensor, workspace: torch.Tensor, error: torch.Tensor, key: int) -> None:
    p=_plans[key]
    b=p._buffers(input,output,workspace,error,ids,raw,low,high,units)
    p._check(p._run(p.handle,C.byref(b)))


@matmul.register_fake
def _matmul_fake(input,raw,low,high,units,ids,output,workspace,error,key):
    if input.dtype not in (torch.float16,torch.bfloat16) or output.dtype!=input.dtype:
        raise ValueError('matmul requires matching native16 input/output')
    if workspace.dtype!=torch.uint8: raise ValueError('workspace must be bytes')
    _scalar(error)
