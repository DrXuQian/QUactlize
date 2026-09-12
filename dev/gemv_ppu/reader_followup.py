"""Remaining four cold Q4 shapes; bounded reader transfer, five-percent gate."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

from dev.gemv_cuda.build import digest, replace_once
from dev.gemv_ppu import reader_reuse as frozen
from dev.gemv_ppu.access_pattern import footprint
from dev.gemv_ppu.config_space import Config, disposition
from dev.gemv_ppu.h800_port import candidate_source

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'quactlize.q4-reader-followup.v1'
SHAPES = ((512, 2048), (1024, 5120), (4096, 2048), (4096, 4096))
GEOMETRIES = {
    (512, 2048): ((2, 4, 2), (4, 8, 2), (8, 16, 2), (4, 8, 4)),
    (1024, 5120): ((4, 10, 2), (4, 20, 4), (8, 20, 2), (2, 10, 4), (4, 10, 8)),
    (4096, 2048): ((8, 16, 4), (4, 8, 4), (4, 8, 8), (4, 4, 8), (8, 8, 4)),
    (4096, 4096): ((8, 16, 4), (4, 8, 4), (4, 16, 8), (4, 8, 8), (4, 10, 8)),
}
SMALL_GEOMETRIES = tuple((width, warps) for width in (8, 16, 32) for warps in (4, 8, 16))
READER_REVIEW = ROOT / 'docs/measurements/q4_reader_reuse_20260912/review.json'
REFERENCE_LIMIT_PCT = 5.0


@dataclass(frozen=True)
class Candidate:
    family: str
    recipe: tuple

    @property
    def key(self):
        if self.family == 'affine':
            v, c, w, p = self.recipe
            return f'v{v}-c{c}-w{w}-p{p}'
        h, width, warps = self.recipe
        return f'meta-h{h}-n{width}-w{warps}'

    @property
    def arithmetic(self):
        return 'FP32_GROUP_AFFINE' if self.family == 'affine' else 'PER_WEIGHT_FP16'

    def geometry(self, n, k):
        if self.family == 'affine':
            return Config(*self.recipe[1:]).geometry(n, k)
        _, width, warps = self.recipe
        return dict(width=width, warps=warps, tile_n=width, grid=n//width,
                    threads=warps*32, k_passes=(k//128+warps-1)//warps,
                    last_pass_warps=(k//128-1)%warps+1, inter_cta_split=1)


def inventory(n, k):
    rows = [Candidate('affine', (v, *c)) for c in GEOMETRIES[n, k]
            for v in (range(8) if c[0] in (4, 8) else (0, 4))]
    if (n, k) == (512, 2048):
        rows += [Candidate('small', (h, width, warps))
                 for width, warps in SMALL_GEOMETRIES for h in (0, 1)]
    return rows


def lookup(n, k, key):
    found = [c for c in inventory(n, k) if c.key == key]
    if len(found) != 1:
        raise ValueError('unknown followup candidate: ' + key)
    return found[0]


def baseline(n, k):
    rows = json.loads(frozen.REVIEW.read_text())['cases']
    row = next(r for r in rows if r['shape'] == [1, n, k])
    if (n, k) == (512, 2048):
        if row['selected'] != 'baseline':
            raise ValueError('small historical winner changed')
        return Candidate('small', (0, 8, 16))
    g = row['selected_config']
    candidate = Candidate('affine', (0, g['columns'], g['warps'], g['values']))
    if candidate not in inventory(n, k):
        raise ValueError('historical winning geometry omitted')
    return candidate


def plan():
    cases = []
    for n, k in SHAPES:
        for c, w, p in GEOMETRIES[n, k]:
            if disposition(Config(c, w, p), n, k) != 'CANDIDATE':
                raise ValueError('geometry outside immutable config controls')
            workers = w*32//c
            if workers % 8 or (k//32) % (32//c):
                raise ValueError('cooperative operation has a partial warp')
        cases.append(dict(n=n, k=k, baseline=baseline(n, k).key,
            candidates=[dict(key=c.key, family=c.family, recipe=c.recipe,
                             arithmetic=c.arithmetic, geometry=c.geometry(n, k)) for c in inventory(n, k)]))
    return dict(schema=SCHEMA, qtype=12, m=1, cache='rotating', reference_limit_pct=REFERENCE_LIMIT_PCT,
        cases=cases, prior_closed_shapes=[[5120, 8192], [8192, 5120]],
        scope='REMAINING_FOUR_SHAPES_NOT_NEW_MEASUREMENTS_OF_PRIOR_TWO',
        selection_basis='PRIOR_SHORTLIST_PLUS_P8_READER_TRANSFER; SMALL_COOPERATIVE_FAMILY_RETAINED',
        pruning=['No C2 cooperative A/units: its warp crosses metadata superblocks',
                 'No partial-warp shuffle or idle K-worker geometries',
                 'No new shared staging, inter-CTA Split-K or expanded offline format'])


def payload(n, k, family='affine'):
    if (n, k) not in SHAPES or family not in ('affine', 'small') or (family == 'small' and (n, k) != (512, 2048)):
        raise ValueError('unknown followup payload')
    return f'libq4_followup_{family}_n{n}_k{k}.so'


def source(n, k):
    # Reuse the exact admitted template body, not its fixed-shape C wrapper.
    old = frozen.source(5120, 8192)
    body = old[:old.index('extern "C" int q4_reader_run_5120_8192(')]
    body = replace_once(body,
        'static_assert((Columns==4 || Columns==8) && Workers%8==0 && K%256==0);',
        'static_assert(((Columns==4 || Columns==8) || (Variant&3)==0) && Workers%8==0 && K%256==0);')
    body += f'''extern "C" int q4_followup_run_{n}_{k}(int variant,int columns,int warps,int values,
        void const* a,void const* low,void const* units,void* out,void* stream) {{
    if(!a || !low || !units || !out || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(out)&3)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for r in inventory(n, k):
        if r.family != 'affine':
            continue
        v, c, w, p = r.recipe
        body += f'''    if(variant=={v} && columns=={c} && warps=={w} && values=={p}) {{
        q4_reader_reuse<{v},{c},{w},{p},{n},{k}><<<{n//(c*p)},{w*32},0,static_cast<hggcStream_t>(stream)>>>(
            a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return body + '    return QKG_INVALID;\n}\n'


def small_source():
    old = candidate_source('small')
    body = old[:old.index('extern "C" int qkg_pair_launch_12(')]
    start = body.index('template<int Width, int Warps, bool StageA')
    end = body.index('\n}\n', start) + 3
    original_kernel = body[start:end]
    clone = replace_once(original_kernel, '__global__ void q4_cooperative_metadata(',
                         '__global__ void q4_cooperative_header32(')
    clone = replace_once(clone, 'auto sz = aligned_scale_zero(unit, g & 7);',
                         'auto sz = followup_scale_zero32(unit, g & 7);')
    helper_start = body.index('__device__ __forceinline__ ScaleZero aligned_scale_zero')
    helper_end = body.index('\n}\n', helper_start) + 3
    helper = body[helper_start:helper_end]
    helper = replace_once(helper, 'ScaleZero aligned_scale_zero(',
                          'quactlize::dev::q4_native::ScaleZero followup_scale_zero32(')
    begin = helper.index('    uint64_t const run=')
    finish = helper.index('    __half2_raw codes_raw,header_raw;')
    helper = helper[:begin] + '''    uint32_t scales=(group&4) ? (m.z>>16)|(m.w<<16) : m.y;
    uint32_t mins=(group&4) ? m.w>>8 : (m.y>>24)|(m.z<<8);
    unsigned shift=6*(group&3);
    unsigned sc=(scales>>shift)&63, mn=(mins>>shift)&63;
''' + helper[finish:]
    body = body[:end] + '\n' + helper + '\n' + clone + body[end:]
    body += '''extern "C" int q4_followup_small_run(int h32,int width,int warps,
        void const* a,void const* low,void const* units,void* out,void* stream) {
    if(!a || !low || !units || !out || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(out)&3)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for width, warps in SMALL_GEOMETRIES:
        for h in (0, 1):
            kernel = 'q4_cooperative_header32' if h else 'q4_cooperative_metadata'
            body += f'''    if(h32=={h} && width=={width} && warps=={warps}) {{
        {kernel}<{width},{warps},false,true,512,2048><<<{512//width},{warps*32},0,static_cast<hggcStream_t>(stream)>>>(
            a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return body + '    return QKG_INVALID;\n}\n'


def access(candidate, n, k, bases=None):
    if candidate.family == 'affine':
        v, c, w, p = candidate.recipe
        return frozen.access(frozen.Reader(v, Config(c, w, p)), n, k, bases)
    bases = dict(A=0, B=0, metadata=0) if bases is None else dict(bases)
    _, width, warps = candidate.recipe
    g = [lane//8 for lane in range(32)]
    residue = [lane%8 for lane in range(32)]
    streams = {}
    for vector in range(width//8):
        streams[f'B_v{vector}'] = ([bases['B']+2*((gg*8+rr)*n+vector*8)
                                    for gg, rr in zip(g, residue)], 16)
        streams[f'units_v{vector}'] = ([bases['metadata']+16*(vector*8+rr) for rr in residue], 16)
    streams['A_cooperative_vector'] = ([bases['A']+gg*64+rr*8 for gg, rr in zip(g, residue)], 8)
    return dict(key=candidate.key, model='SOURCE_FOOTPRINT_NOT_MEASURED_TRAFFIC', base_mod128=bases,
        geometry=candidate.geometry(n, k), lane_group=g, residue=residue,
        streams={name:dict(lane_byte_addresses=addr, width_bytes=width,
                          granules={str(s):footprint(addr, width, s) for s in (32, 64, 128)})
                 for name, (addr, width) in streams.items()},
        logical_lane_bytes_per_call=dict(B=n*k//2, A=2*n*k//width, metadata=n*k//2),
        unchanged_A='EIGHT_LANE_REGISTER_TRANSPOSE_ALREADY_IN_BASELINE')


def small_isa(text):
    pattern = re.compile(r'Disassembly of section \.text\.kernel\.[^\n]*q4_cooperative_(metadata|header32)ILi(\d+)ELi(\d+)ELb0ELb1ELi512ELi2048E[^\n]*:')
    rows = {}
    for match in pattern.finditer(text):
        end = text.find('Disassembly of section ', match.end())
        body = text[match.end():end if end >= 0 else None]
        counts = Counter(re.findall(r'\t([a-z][\w.]+)\s', body))
        kind, width, warps = match.groups()
        key = f'meta-h{int(kind=="header32")}-n{width}-w{warps}'
        rows[key] = dict(scope='STATIC_NATIVE_ISA_NOT_DYNAMIC_COUNTS',
            code_fastpath_present=bool(counts['v.lop3.b32'] and any('f16x2' in k for k in counts)),
            fp32_fma_present=any(k.startswith('v.fma.f32') for k in counts),
            operations={k:v for k,v in counts.items() if k.startswith(('vmem.ld.', 'tsm.', 's.cbr', 's.blksyn'))
                        or any(s in k for s in ('shuffle', 'shrl.b64', 'f16x2', 'fma.f32', 'lop3'))})
    return rows


def verify(candidate, reuse, config, previous, controls, baseline_bundle, *, sources=True):
    old = frozen.verify(reuse, config, previous, controls, baseline_bundle, sources=sources)
    m = json.loads((candidate/'manifest.json').read_text())
    expected = {payload(n,k) for n,k in SHAPES} | {payload(512,2048,'small')}
    if (m.get('schema') != SCHEMA or m.get('plan') != json.loads(json.dumps(plan())) or
        m.get('reuse_manifest_sha256') != digest(reuse/'manifest.json') or
        m.get('reader_review_sha256') != digest(READER_REVIEW) or
        m.get('compiler_sha256') != old['compiler_sha256'] or set(m.get('payloads',{})) != expected):
        raise ValueError('followup plan/control package differs')
    for name, row in m['payloads'].items():
        path = candidate/name
        if digest(path) != row['sha256'] or path.read_bytes()[:4] != b'\x7fELF':
            raise ValueError('followup image missing, changed or an LFS pointer: ' + name)
    if digest(candidate/'isa-stats.json') != m['isa_sha256']:
        raise ValueError('followup ISA receipt differs')
    for name, sha in m['source_hashes'].items() if sources else []:
        path = (ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != sha:
            raise ValueError('followup source differs: ' + name)
    return m
