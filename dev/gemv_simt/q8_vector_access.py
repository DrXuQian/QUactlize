"""Q8 vector candidate address model, including legal two-byte scale bases."""
from dev.gemv_simt.access import pattern as original
from dev.gemv_ppu.access_pattern import footprint


def pattern(config,n,k,*,bases=None,warp=0,partition=0,iteration=0):
    p=original(8,config,n,k,1,bases=bases,warp=warp,partition=partition,iteration=iteration)
    old=[s for s in p['streams'] if s['name'].startswith('metadata-')]
    streams=[s for s in p['streams'] if not s['name'].startswith('metadata-')]
    first=old[0];width=2*config.values
    aligned=all(address%width==0 for address in first['addresses'])
    if aligned:
        streams.append(dict(name='metadata-vector',lanes=first['lanes'],addresses=first['addresses'],
            width_bytes=width,footprint={str(s):footprint(first['addresses'],width,s) for s in (32,64,128)}))
    else:
        streams.extend(old)
    p.update(streams=streams,metadata_vectorized=aligned,baseline_metadata_requests=len(old),
             candidate_metadata_requests=1 if aligned else len(old),
             packed_b_live_words_per_thread=dict(baseline=4*config.values,candidate=2*config.values),
             dot_order='half,segment,slot,residue; identical logical K order to baseline',
             accumulator='FP32',weight_format='UNCHANGED_Q8_KPACK2',
             caveat='Source footprint/live ranges, not emitted ISA or measured DRAM transactions')
    return p
