"""Reproducible, bounded transplant of the three frozen Q4 S1 readers.

This is an audit tool, not a runtime source dependency. The standalone
execution headers retain the dequant, dot and CTA fold verbatim. The only
edits are a typed activation view, explicit tile index and removal of the
uninstantiated shared-A branch. Existing control sources are not changed.
"""
from pathlib import Path
from dev.gemv_cuda.build import replace_once
from dev.gemv_ppu.smallm import kernel_parts

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = 'quactlize::execution::q4_s1'


def function(text, signature):
    start = text.index(signature)
    end = text.index('\n}', start) + 2
    return text[start:end] + '\n'


def helpers():
    # Resolve the helper AFTER all dense generator rewrites. The base .cuh
    # is signed; affine dense readers deliberately use unsigned codes. Merely
    # copying the outer kernel body does not preserve that numerical contract.
    signature = 'template<int Slot>\n__device__ __forceinline__ __half2 codes'
    signed = function(kernel_parts(512,2048)[0], signature)
    unsigned = function(kernel_parts(1024,5120)[0], signature)
    if unsigned != signed.replace('-8.f', '-0.f') or function(kernel_parts(4096,2048)[0], signature) != unsigned:
        raise ValueError('frozen dense code-helper conventions differ')
    code = replace_once(signed, 'template<int Slot>', 'template<int Slot,int Bias>')
    code = replace_once(code, '-8.f', '-float(Bias)')
    code = replace_once(code, '    constexpr int Pos=', '    static_assert(Bias==0 || Bias==8);\n    constexpr int Pos=')
    aligned = (ROOT/'dev/gemv_cuda/q4_aligned.cuh').read_text()
    affine = (ROOT/'dev/gemv_cuda/q4_group_affine.cuh').read_text()
    reuse = (ROOT/'dev/gemv_ppu/reader_reuse.hpp').read_text()
    latency = (ROOT/'dev/gemv_ppu/small_latency.hpp').read_text().split('\ntemplate<int Header,int Loading')[0]
    reduce = (ROOT/'dev/gemv_cuda/q4_warp_reduce_scatter.cuh').read_text().replace('#pragma once\n','')
    fold = (ROOT/'dev/gemv_ppu/medium_reduce.hpp').read_text()
    latency = replace_once(latency, 'template<int AMode>\n__device__ __forceinline__ uint4 latency_residue_a(__half const* a,',
                           'template<int AMode,int Input>\n__device__ __forceinline__ uint4 latency_residue_a(Activation<Input> a,')
    latency = replace_once(latency, '*reinterpret_cast<uint2 const*>(a+g*32+residue*4)', 'a.load4(g*32+residue*4)')
    reuse = replace_once(reuse, 'template<int Columns>\n__device__ __forceinline__ uint4 q4_cooperative_a_chunk(void const* a,',
                         'template<int Columns,int Input>\n__device__ __forceinline__ uint4 q4_cooperative_a_chunk(Activation<Input> a,')
    reuse = replace_once(reuse, 'auto ptr=static_cast<__half const*>(a)+group*32+(lane%Columns)*(32/Columns);',
                         'int64_t offset=int64_t(group)*32+(lane%Columns)*(32/Columns);')
    reuse = replace_once(reuse, '*reinterpret_cast<uint4 const*>(ptr)', 'a.load8(offset)')
    reuse = replace_once(reuse, '*reinterpret_cast<uint2 const*>(ptr)', 'a.load4(offset)')
    pieces = ['struct ScaleZero { __half scale,zero; };\n',
              code,
              function(aligned, '__device__ __forceinline__ uint4 aligned_unit'),
              function(affine, '__device__ __forceinline__ float2 q4_affine_header'),
              reuse, latency, reduce, fold]
    body = '\n'.join(pieces).replace('quactlize::dev::q4_native', NAMESPACE)
    return '#pragma once\n#include "q4_s1_activation.cuh"\n\nnamespace '+NAMESPACE+' {\n'+body+'\n}\n'


def row(family):
    shape = {'meta':(512,2048), 'medium':(1024,5120), 'reuse':(4096,2048)}[family]
    _, original, _, name = kernel_parts(*shape)
    s = replace_once(original, 'template<', 'template<int Input,')
    s = replace_once(s, '__global__ void '+name+'(void const* a_ptr,',
                     '__device__ __forceinline__ void row_'+family+'(int tile,Activation<Input> a_ptr,')
    s = s.replace('    using namespace quactlize::dev::q4_native;\n','').replace('blockIdx.x','tile')
    if family != 'reuse':
        s = replace_once(s, 'static_cast<__half const*>(a_ptr)', 'a_ptr')
        start = s.index('    extern __shared__')
        end = s.index('\n    float2 ', start)
        s = s[:start]+'    static_assert(AMode==0 || AMode==1);\n'+s[end:]
    if family == 'medium':
        s = replace_once(s, '*reinterpret_cast<uint4 const*>(act+g*32+slot*8)', 'act.load8(g*32+slot*8)')
    for slot in range(4):
        s = replace_once(s, f'codes<{slot}>', f'codes<{slot},{8 if family=="meta" else 0}>')
    return s


def bodies():
    return ('#pragma once\n#include "q4_s1_helpers.cuh"\n\nnamespace '+NAMESPACE+' {\n'
            + '\n'.join(row(family) for family in ('meta','medium','reuse'))+'\n}\n')


def verify_retained_helpers(current=None):
    """Bind unchanged code/metadata/reduction math after the typed-A extension.

    BF16 register transport has independent executable coverage in
    tests/bf16_fastpath_host.cpp; it is no longer a byte-for-byte transplant.
    """
    if current is None:
        current=(ROOT/'quactlize/execution/q4_s1_helpers.cuh').read_text()
    frozen=helpers()
    for signature in ('template<int Slot,int Bias>',
                      '__device__ __forceinline__ uint4 aligned_unit',
                      '__device__ __forceinline__ float2 q4_affine_header(',
                      '__device__ __forceinline__ float2 q4_affine_header32(',
                      'template<int Header>\n__device__ __forceinline__ uint2 latency_fields',
                      'template<int Header>\n__device__ __forceinline__ quactlize::execution::q4_s1::ScaleZero latency_half_header',
                      'template<int Header>\n__device__ __forceinline__ float2 latency_affine_header',
                      'template<int Count, int Stride, int Width>',
                      'template<int Round,int Warps,int TileN>'):
        if function(current,signature)!=function(frozen,signature):
            raise ValueError('retained F16 helper differs: '+signature)


if __name__ == '__main__':
    import sys
    sys.stdout.write(helpers() if sys.argv[1]=='helpers' else bodies())
