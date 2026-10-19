"""Two independent Q6 axes: warp count and shallow residue load lookahead.

Canonical K-pack bytes, activation rounding and ordered per-thread arithmetic
are unchanged. Across warp counts the FP32 partial sum association differs.
Only matched-warp ordinary/pipeline pairs have a raw-bit equality contract.
"""
from dev.gemv_simt.model_followup import once
from dev.gemv_simt.q6_index import ROOT, index_body, source as parent_source

WARPS = (8, 4, 2, 8, 4, 2)
ARMS = ('w8', 'w4', 'w2', 'w8-pipe', 'w4-pipe', 'w2-pipe')
KERNELS = ('q6_unsigned_fields', 'q6_w4', 'q6_w2', 'q6_w8_pipe', 'q6_w4_pipe', 'q6_w2_pipe')
KERNEL_NAMES = tuple('::'+name+'(' for name in KERNELS)
CONFIGS = [dict(variant=3, columns=4, warps=w, values=8, split=1) for w in WARPS]
EQUAL_GROUPS = [[0, 3], [1, 4], [2, 5]]


def body(pipeline=False):
    text = index_body((ROOT/'quactlize/execution/simt_kernel.cuh').read_text())
    text = once(text, 'void q6_unsigned_fields_body(',
                'void q6_pipeline_body(' if pipeline else 'void q6_warp_body(')
    text = once(text, 'Warps==8 && P==8', '(Warps==2 || Warps==4 || Warps==8) && P==8')
    if not pipeline:
        return text
    # All Q6 low/high words survive into slot 1. Issue residues 0/1 first,
    # then look ahead two residues during slot 0. Never prefetch another group.
    begin = text.index('            #pragma unroll\n            for (int segment=')
    end = text.index('            #pragma unroll\n            for (int slot=', begin)
    text = text[:begin]+'''            static_assert(Segments==1 && F::group==16);
            #pragma unroll
            for (int residue=0;residue<2;++residue) {
                unsigned k=g*F::group+half*4+residue;
                load_words<P>(low+F::Low::word_index(col,k,call.n),lo[0][residue]);
                load_words<P>(high+F::High::word_index(col,k,call.n),hi[residue]);
            }
'''+text[end:]
    seam = '''                for (int residue=0;residue<4;++residue) {
                    unsigned k=g*F::group+slot*8+half*4+residue;'''
    return once(text, seam, '''                for (int residue=0;residue<4;++residue) {
                    if (slot==0 && residue<2) {
                        unsigned next_k=g*F::group+half*4+residue+2;
                        load_words<P>(low+F::Low::word_index(col,next_k,call.n),lo[0][residue+2]);
                        load_words<P>(high+F::High::word_index(col,next_k,call.n),hi[residue+2]);
                    }
                    unsigned k=g*F::group+slot*8+half*4+residue;''')


def source(compute):
    original = parent_source(compute)
    split = original.index('extern "C" int q6_metadata_run(')
    # Preserve the original A3 W8 body and wrapper verbatim.
    prefix = original[:split]+'\nnamespace quactlize::execution::simt {\n'
    prefix += body()+body(pipeline=True)
    c = int(compute == 'bf16')
    for arm in range(1, len(ARMS)):
        fn = 'q6_pipeline_body' if arm >= 3 else 'q6_warp_body'
        prefix += f'''\n__global__ void {KERNELS[arm]}(qkg_call_v1 call,int split) {{
    {fn}<14,1,3,4,{WARPS[arm]},8,{c},0>(call,split);
}}
'''
    export = original[split:]
    export = once(export, 'arm>1', 'arm>5')
    export = once(export, 'qkg_simt_config_v1 cfg{1,sizeof(cfg),3,4,8,8,1};',
                  '''int warps=arm%3==0?8:arm%3==1?4:2;
    qkg_simt_config_v1 cfg{1,sizeof(cfg),3,4,warps,8,1};''')
    begin = export.index('    if(arm==0) simt::')
    end = export.index('    return hggcGetLastError()', begin)
    launch = '    switch(arm) {\n'
    for arm, (kernel, warps) in enumerate(zip(KERNELS, WARPS)):
        launch += f'    case {arm}: simt::{kernel}<<<blocks,{warps*32},0,stream>>>(call,1);break;\n'
    export = export[:begin]+launch+'    }\n'+export[end:]
    return prefix+'}\n'+export


def validate_record(record):
    if (record.get('kernel_names') != list(KERNEL_NAMES) or record.get('configs') != CONFIGS or
            record.get('equal_groups') != EQUAL_GROUPS):
        raise ValueError('Q6 schedule image contract differs from source')


def load_events(pipeline):
    """Executable host model for load/use lifetime proof, per half/group."""
    yield from (('load', r) for r in range(2 if pipeline else 4))
    for slot in range(2):
        for residue in range(4):
            if pipeline and slot == 0 and residue < 2:
                yield 'load', residue+2
            yield 'use', (slot, residue)
