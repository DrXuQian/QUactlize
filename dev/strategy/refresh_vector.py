"""Expose the already-tested Q8 candidate through the isolated component ABI."""
from dev.gemv_simt.q8_address_source import source as measured_source
from dev.gemv_simt.model_followup import once
from quactlize.runtime.tuning import digest


def source(item):
    from dev.strategy.vector import spec
    from quactlize.dispatch.table import Workload, Recipe
    cfg = item['config']
    r = Recipe.make('simt', 'fq', **(cfg | {'split': 1, 'q8_body': item['q8_body']}))
    if spec(Workload(8, 1024, 8192, 1, compute=item['compute']), r) != item:
        raise ValueError('noncanonical Q8 experiment specialization')
    config = cfg | dict(split=1, hoist=True)
    text = measured_source(8, item['compute'], [config])
    text = once(text, 'extern "C" int general_reader_run(',
                'static int refresh_q8_run(')
    text += f'''
extern "C" char const* strategy_vector_identity() {{ return "{digest(item)}"; }}
extern "C" int strategy_vector_query(qkg_simt_call_v2 const* d,
        quactlize_ppu_placed_arrangement_v2 const* a,int split,qkg_sizes_v1* out) {{
    if (!d || !a || !out || split!=1 || d->call.qtype!=8 ||
        d->call.mode!=QKG_DENSE || d->call.input_type!=QKG_F32 ||
        d->compute_type!={int(item['compute']=='bf16')}) return QKG_INVALID;
    *out={{}};
    qkg_simt_config_v1 cfg{{1,sizeof(cfg),5,{cfg['columns']},{cfg['warps']},{cfg['values']},1}};
    return quactlize::execution::simt::query_v2(*d,cfg,a,*out);
}}
extern "C" int strategy_vector_run(qkg_simt_call_v2 const* d,
        quactlize_ppu_placed_arrangement_v2 const* a,int split) {{
    qkg_sizes_v1 sizes{{}};
    int rc=strategy_vector_query(d,a,split,&sizes);
    if (rc) return rc;
    return refresh_q8_run(d,a,0,{item['q8_body']});
}}
'''
    return text
