"""Bounded Q4/SIMT entrypoints; changed bodies require explicit recipe flags.

The public Q4 entry deliberately accepts only its selected recipe. This
private carrier can instantiate a frozen offline candidate without changing
that API, its policy, the weight format or the arithmetic implementation.
"""
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from dev.strategy.prescreen import rejection
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import FLAGS, LIBRARIES, HEADER_SUFFIXES, sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]


def execution_contract(w):
    if w.operation not in ('dense', 'indexed') or w.storage != 'f32':
        raise ValueError('vector call requires a dense/indexed F32 workload')
    return dict(qtype=w.qtype, n=w.n, k=w.k, experts=w.experts, rows=w.m*w.topk,
        mode=0 if w.operation=='dense' else 2, input_type=1,
        compute_type=int(w.compute=='bf16'), channels=w.channels, topk=w.topk,
        a_row_stride=w.input_stride, a_token_stride=w.channels*w.input_stride,
        ids_stride=w.topk, out_row_stride=w.output_stride,
        ids_present=w.operation=='indexed')


def checked_call(w, request):
    """Validate the actual ABI request, not just the planner's operation label."""
    expected=execution_contract(w)
    actual={k:int(getattr(request.call,k)) for k in expected if k not in ('compute_type','ids_present')}
    actual.update(compute_type=int(request.compute_type),ids_present=bool(request.call.ids))
    if actual!=expected:
        fields=[k for k in expected if actual[k]!=expected[k]]
        raise ValueError('vector execution contract differs from workload: '+','.join(fields))
    return actual


def spec(w, recipe):
    if (recipe.kind not in ('q4', 'simt') or w.storage != 'f32' or w.m > 8 or
            w.operation not in ('dense', 'indexed') or rejection(recipe, w)):
        raise ValueError('unsupported explicit vector workload/recipe')
    fields = ('reader', 'variant', 'columns', 'warps', 'values') if recipe.kind == 'q4' else (
        'variant', 'columns', 'warps', 'values')
    p = dict(recipe.parameters)
    # Generic SIMT retains runtime N/K, including all admitted measured-body
    # dispatches. Q4 bakes N/K into its existing reader specialization.
    result = dict(kind=recipe.kind, q=w.qtype, compute=w.compute,
                n=w.n if recipe.kind == 'q4' else 0, k=w.k if recipe.kind == 'q4' else 0,
                config={f: p[f] for f in fields})
    if 'moe_pair_rows' in p:
        from dev.strategy.moe_token_pair import GEOMETRIES
        if (w.qtype not in range(10,15) or w.operation!='indexed' or w.compute!='bf16' or
                recipe.kind!='simt' or w.fusion!='none' or w.topk>8 or
                type(p['moe_pair_rows']) is not int or p['moe_pair_rows'] not in (1,2) or p['variant']!=3 or
                tuple(p[f] for f in ('columns','warps','values')) not in GEOMETRIES or
                any(f in p for f in ('cta_tokens','cta_window','q8_body'))):
            raise ValueError('MoE pair reuse requires indexed BF16 F32 Q2-Q6 T1..8/topk<=8')
        result['moe_pair_rows']=p['moe_pair_rows']
    if 'cta_tokens' in p:
        geometry=tuple(p[f] for f in ('variant','columns','warps','values'))
        allowed=((5,4,8,4),(5,8,4,4)) if w.qtype==8 else ((3,4,8,8),(3,4,2,8),(3,8,4,4))
        if (w.qtype not in (8,10,11,12,13,14) or w.operation!='dense' or w.compute!='f16' or recipe.kind!='simt' or
                type(p['cta_tokens']) is not int or p['cta_tokens']!=w.m or not 2<=w.m<=8 or
                geometry not in allowed or 'q8_body' in p):
            raise ValueError('CTA-local tokens require dense F16 M2..8 and a declared format/geometry')
        result['cta_tokens']=p['cta_tokens']
        if w.qtype==8:
            if type(p.get('cta_window')) is not int or p['cta_window'] not in (1,2):
                raise ValueError('Q8 token reuse requires an explicit one/two-half load window')
            result['cta_window']=p['cta_window']
    if 'cta_window' in p and not (w.qtype==8 and 'cta_tokens' in p):
        raise ValueError('CTA load window is only defined for Q8 token reuse')
    if 'q8_body' in p:
        if (w.qtype != 8 or w.operation != 'dense' or recipe.kind != 'simt' or
                p.get('split') != 1 or type(p['q8_body']) is not int or p['q8_body'] not in (0, 1, 3, 4, 5) or
                (p['variant'], p['columns'], p['warps'], p['values']) not in
                ((5, 4, 8, 4), (5, 8, 4, 4))):
            raise ValueError('Q8 address/window candidate is dense S1 with a declared geometry only')
        result['q8_body'] = p['q8_body']
    return result


def source(item):
    if 'moe_pair_rows' in item:
        from dev.strategy.moe_token_pair import source as pair_source
        return pair_source(item)
    if 'cta_tokens' in item:
        from dev.strategy.token_cta import source as token_source
        return token_source(item)
    if 'q8_body' in item:
        from dev.strategy.refresh_vector import source as q8_source
        return q8_source(item)
    w = Workload(item['q'], item['n'] or 1024, item['k'] or 8192, 1, compute=item['compute'])
    config = item['config'] | (dict(split=1) if item['kind']=='simt' else {})
    if spec(w, Recipe.make(item['kind'], 'fq', **config)) != item:
        raise ValueError('vector specialization is not canonical')
    p = item['config']
    compute = int(item['compute'] == 'bf16')
    common = f'''if (!d || !a || !out || d->call.qtype!={item['q']} ||
        d->call.input_type!=QKG_F32 || d->compute_type!={compute}) return QKG_INVALID;
    *out={{}};
    if (d->call.mode!=QKG_DENSE && d->call.mode!=QKG_INDEXED) return QKG_SHAPE;
'''
    if item['kind'] == 'q4':
        tile = 8 if p['reader'] == 0 else p['columns'] * p['values']
        header = '#include "q4_decode_kernel.cuh"\n'
        query = common + f'''
    if (!quactlize::execution::q4_s1::typed_valid(*d)) return QKG_INVALID;
    auto const& c=d->call;
    if (split!=1 || c.n!={item['n']} || c.k!={item['k']} || c.n%{tile} ||
        c.a_row_stride%4 || (c.mode==QKG_INDEXED && c.a_token_stride%4)) return QKG_SHAPE;
    qkg_config_v1 f{{1,sizeof(f),16,4,1}};
    int rc=quactlize::execution::query(c,f,a,*out);
    if (rc) return rc;
    int tokens=c.mode==QKG_INDEXED ? c.rows/c.topk : c.rows;
    if (tokens<1 || tokens>8 || uint64_t(c.rows)*(c.n/{tile})>INT32_MAX) return QKG_SHAPE;
    return QKG_OK;
'''
        buffers = 'quactlize::execution::q4_s1::buffers_v2(*d,sizes)'
        arguments = [p[f] for f in ('reader', 'variant', 'warps', 'values', 'columns')]
        arguments += [item['n'], item['k'], compute]
        launch = 'quactlize::execution::q4_decode::launch<' + ','.join(map(str, arguments)) + '>(d->call)'
    else:
        header = '#include "measured_decode.cuh"\n#include "simt_validation.hpp"\n'
        if item['q'] != 8:
            header += '#include "simt_common.cuh"\n'
        config = 'qkg_simt_config_v1 f{1,sizeof(f),' + ','.join(str(p[f]) for f in (
            'variant', 'columns', 'warps', 'values')) + ',split};\n'
        query = common + config + 'return quactlize::execution::simt::query_v2(*d,f,a,*out);\n'
        buffers = 'quactlize::execution::simt::buffers_v2(*d,sizes)'
        reader = 'simt::q8_vector' if p['variant'] >= 4 else 'simt'
        arguments = [item['q'], *[p[f] for f in ('variant', 'columns', 'warps', 'values')]]
        launch = f'quactlize::execution::{reader}::launch_v2<' + ','.join(map(str, arguments)) + '>(*d,split)'
        # This branch is part of the shipping implementation, not a new
        # selector. Omitting it would benchmark a slower generic lookalike.
        prefix = ''
        if item['q'] != 8:
            # Same ordering and compile-time geometry guard as simt_codegen:
            # common -> fixed measured -> generic. Old frozen experiments
            # remain old evidence; the changed source creates new module keys.
            prefix = f'''if constexpr (quactlize::execution::simt::kCommonReader<{','.join(map(str, arguments))}>) {{
        if (quactlize::execution::simt::common_reader_scope(*d,f))
            return quactlize::execution::simt::common_reader_launch<{item['q']},{p['columns']},{p['warps']},{p['values']}>(*d,split);
    }}
    '''
        launch = config + prefix + f'''if (quactlize::execution::simt::measured_decode(*d,f)!=
        quactlize::execution::simt::MeasuredDecode::None)
        return quactlize::execution::simt::measured_decode_launch<{item['q']}>(*d,f);
    return {launch};'''
    result = header + f'extern "C" char const* strategy_vector_identity() {{ return "{digest(item)}"; }}\n'
    signature = 'qkg_simt_call_v2 const* d,quactlize_ppu_placed_arrangement_v2 const* a,int split'
    result += f'extern "C" int strategy_vector_query({signature},qkg_sizes_v1* out) {{\n{query}}}\n'
    result += f'''extern "C" int strategy_vector_run({signature}) {{
    qkg_sizes_v1 sizes{{}};
    int rc=strategy_vector_query(d,a,split,&sizes); if(rc) return rc;
    rc={buffers}; if(rc) return rc;
    ''' + (f'return {launch};' if item['kind'] == 'q4' else launch) + '\n}\n'
    return result


class Compiler:
    def __init__(self, sdk, cache):
        from quactlize.runtime.native import sdk_identity
        self.sdk, self.cache = Path(sdk).resolve(), Path(cache).resolve()
        self.includes = [ROOT/'quactlize/execution', ROOT/'quactlize/include',
                         ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
        self.paths = sorted({p for folder in [*self.includes, ROOT/'quactlize/runtime', ROOT/'quactlize/integrations',
                                            ROOT/'quactlize/decode']
                             for p in folder.rglob('*') if p.is_file() and p.suffix in HEADER_SUFFIXES})
        # Extra source is experiment-only; seal all generators consumed by it.
        self.includes.append(ROOT)
        self.paths += [ROOT/p for p in ('dev/strategy/refresh_vector.py','dev/strategy/token_cta.py',
            'dev/strategy/moe_token_pair.py','dev/strategy/moe_token_pair.cuh','dev/strategy/moe_token_pair.hpp',
            'dev/gemv_simt/q8_address_source.py', 'dev/gemv_simt/q8_dense_source.py',
            'dev/gemv_simt/general_locate.hpp', 'dev/gemv_simt/model_followup.py')]
        self.identity = dict(sdk=sdk_identity(self.sdk), flags=FLAGS,
            generator=sha(__file__), headers={str(p.relative_to(ROOT)):sha(p) for p in self.paths},
            host=subprocess.check_output(['g++', '--version'], text=True))

    def build(self, item):
        text = source(item)
        key = digest(dict(identity=self.identity, spec=item, source=text))
        folder = self.cache / key
        folder.mkdir(parents=True, exist_ok=True)
        if folder.is_symlink() or folder.resolve().parent != self.cache:
            raise ValueError('vector cache escapes root')
        with (folder/'build.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            if (folder/'manifest.json').exists():
                row = json.loads((folder/'manifest.json').read_text())
                self.verify(item, row)
                return row | dict(cache_hit=True)
            start = time.monotonic()
            with tempfile.TemporaryDirectory(prefix='compile-', dir=folder) as temp:
                stage = Path(temp)
                src, obj, so = (stage/n for n in ('kernel.cu', 'kernel.o', 'kernel.so'))
                src.write_text(text)
                commands = [[str(self.sdk/'bin/hgcc'), *FLAGS, *[f'-I{p}' for p in self.includes],
                             '-c', str(src), '-o', str(obj)],
                            ['g++', '-shared', '-Wl,-Bsymbolic', str(obj), '-o', str(so),
                             f'-L{self.sdk}/lib', *[f'-l{x}' for x in LIBRARIES]]]
                env = dict(os.environ, PATH=str(self.sdk/'bin')+':'+os.environ.get('PATH', ''),
                    LD_LIBRARY_PATH=str(self.sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''))
                with (folder/'build.log').open('w') as log:
                    for command in commands:
                        log.write(json.dumps(command)+'\n'); log.flush()
                        subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
                if any(sha(ROOT/p)!=h for p,h in self.identity['headers'].items()) or sha(__file__)!=self.identity['generator']:
                    raise ValueError('vector compile source changed')
                row = dict(key=key, spec=item, identity=self.identity, sha256=sha(so),
                           path=str(folder/'kernel.so'), seconds=time.monotonic()-start)
                os.replace(so, folder/'kernel.so')
                atomic_json(folder/'manifest.json', row)
            return row | dict(cache_hit=False)

    def verify(self, item, row):
        key = digest(dict(identity=self.identity, spec=item, source=source(item)))
        path = self.cache/key/'kernel.so'
        if (row.get('spec')!=item or row.get('identity')!=self.identity or row.get('key')!=key or
                row.get('path')!=str(path) or path.is_symlink() or path.resolve(strict=True).parent!=self.cache/key or
                sha(path)!=row.get('sha256')):
            raise ValueError('vector module/source/SDK identity differs')


class Vector:
    def __init__(self, b, record, item, split=1, *, workload):
        from dev.smallm_closure.bench import Buffer
        from tools.run_kpack_gemv_gate import Resources
        from quactlize.execution.native import Call, SimtCallV2, Sizes, Arrangement
        from dev.strategy.isolated import checked
        self.b, self.split = b, split
        self.buffers, self.calls = [], []
        if sha(record['path']) != record['sha256'] or record['spec'] != item:
            raise ValueError('vector image differs')
        self.lib = C.CDLL(record['path'], mode=C.RTLD_LOCAL)
        identity = self.lib.strategy_vector_identity
        identity.argtypes, identity.restype = [], C.c_char_p
        if identity().decode() != digest(item): raise ValueError('loaded vector specialization differs')
        query, self.run_fn = self.lib.strategy_vector_query, self.lib.strategy_vector_run
        query.argtypes, query.restype = [C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.POINTER(Sizes)], C.c_int
        self.run_fn.argtypes, self.run_fn.restype = query.argtypes[:3], C.c_int
        p = b.p
        call = Call(version=1,size=C.sizeof(Call),qtype=p['q'],n=p['n'],k=p['k'],experts=p['experts'],
            rows=b.rows,mode=p['mode'],input_type=1,channels=p['channels'],topk=p['topk'],
            a_row_stride=p['k'],a_token_stride=p['channels']*p['k'],ids_stride=p['topk'],out_row_stride=p['n'],
            a=b.a.ptr,ids=b.ids.ptr if b.ids else None,output=b.out.ptr,stream=b.r.stream.value)
        request = SimtCallV2(call,int(p['compute']=='bf16'))
        contract = checked_call(workload, request)
        sizes = Sizes()
        self.r = Resources(b.sdk)
        try:
            checked(query(C.byref(request),C.byref(b.arr),split,C.byref(sizes)), 'explicit vector query')
            self.scratch = Buffer(self.r,sizes.workspace_bytes)
            self.buffers.append(self.scratch)
            for plane in b.ring.planes:
                c = SimtCallV2.from_buffer_copy(request)
                for name,ptr in plane.items(): setattr(c.call,name,ptr)
                c.call.workspace,c.call.workspace_bytes=self.scratch.ptr,self.scratch.size
                self.calls.append(c)
            self.receipt = dict(kind=item['kind'],spec=item,split=split,
                execution_contract=contract,
                build_key=record['key'],module_sha256=record['sha256'],workspace_bytes=sizes.workspace_bytes,
                scope=('EXPERIMENTAL_INDEXED_PAIR_F32_BF16_REAL_REDUCER'
                       if 'moe_pair_rows' in item else 'EXPERIMENTAL_CTA_LOCAL_TOKENS_F32_F16_REAL_REDUCER'
                       if 'cta_tokens' in item else 'EXPLICIT_MEASURED_Q8_CANDIDATE_F32_ENDPOINTS_DENSE_S1'
                       if 'q8_body' in item else 'EXPLICIT_SHIPPING_BODY_F32_ENDPOINTS_REAL_REDUCER'))
            if 'cta_tokens' in item:
                self.receipt['alignment_mod128']=dict(
                    A=b.a.ptr%128,output=b.out.ptr%128,workspace=self.scratch.ptr%128,
                    **{name:sorted({int(p[name])%128 for p in b.ring.planes if p.get(name)})
                       for name in ('low','high','units')})
        except BaseException:
            self.close();raise

    def launch(self, index=0):
        return self.run_fn(C.byref(self.calls[index]),C.byref(self.b.arr),self.split)

    def close(self):
        self.b.sdk.synchronize(self.b.r.stream)
        self.r.close()
