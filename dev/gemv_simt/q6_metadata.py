"""Single-axis Q6 metadata experiment; shipping sources/selection stay intact."""
from pathlib import Path

from dev.gemv_simt.model_followup import once

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('baseline', 'register-fields')


def candidate_body(text):
    start = text.index('template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,')
    end = text.index('\ntemplate<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0>\n__global__', start)
    original = text[start:end]
    modified = once(original, 'void register_reuse_body(', 'void q6_register_fields_body(')
    if modified.count('affine_selected<Q,Changes>(') != 2:
        raise ValueError('metadata seam changed')
    modified = modified.replace('affine_selected<Q,Changes>(', 'q6_metadata::affine(')
    # No address, loads, activation, dot/reduction or launch geometry edits.
    if modified.replace('q6_register_fields_body(', 'register_reuse_body(').replace(
            'q6_metadata::affine(', 'affine_selected<Q,Changes>(') != original:
        raise ValueError('candidate changed more than the metadata seam')
    return modified


def source(compute):
    if compute not in ('f16', 'bf16'):
        raise ValueError('invalid activation compute')
    c = int(compute == 'bf16')
    body = candidate_body((ROOT/'quactlize/execution/simt_kernel.cuh').read_text())
    return '''#include "quactlize/execution/simt_kernel.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "dev/gemv_simt/q6_metadata.cuh"
namespace quactlize::execution::simt {
static_assert(Format<14>::words==9 && Format<14>::Unit::kGroups==16 &&
              Format<14>::Unit::kSbBytes==18 && Format<14>::Unit::kSbPerUnit==2);
''' + body + f'''
__global__ void q6_register_fields(qkg_call_v1 call,int split) {{
    q6_register_fields_body<14,1,3,4,8,8,{c},0>(call,split);
}}
}}
extern "C" int q6_metadata_run(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int arm) {{
    using namespace quactlize::execution;
    if(!d || !a || d->call.qtype!=14 || d->call.input_type!=QKG_F32 ||
       d->compute_type!={c} || arm<0 || arm>1) return QKG_INVALID;
    qkg_simt_config_v1 cfg{{1,sizeof(cfg),3,4,8,8,1}};
    qkg_sizes_v1 sizes{{}};
    int rc=simt::query_v2(*d,cfg,a,sizes);if(rc) return rc;
    rc=simt::buffers_v2(*d,sizes);if(rc) return rc;
    auto const& call=d->call;
    auto stream=static_cast<hggcStream_t>(call.stream);
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    int blocks=call.rows*(call.n/32);
    if(arm==0) simt::register_reuse<14,1,3,4,8,8,{c},0><<<blocks,256,0,stream>>>(call,1);
    else simt::q6_register_fields<<<blocks,256,0,stream>>>(call,1);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}}
'''
