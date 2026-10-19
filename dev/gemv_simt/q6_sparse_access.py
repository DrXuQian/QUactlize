"""Source request footprints; physical transactions still require ACU evidence."""
from copy import deepcopy

from dev.gemv_simt.access import pattern as original_pattern

WORDS=((0,1,2),(0,2,3,4),(4,5,6),(4,7,8))


def pattern(q,config,n,k,storage=1,**kwargs):
    if q!=14 or (config.variant,config.columns,config.warps,config.values,config.split)!=(3,4,8,8,1):
        raise ValueError('sparse packet requires Q6 V3 C4 W8 P8 S1')
    old=original_pattern(q,config,n,k,storage,**kwargs)
    active=[g for g in old['lane_groups'] if g<k//16]
    phases={g//8%4 for g in active}
    if len(phases)>1:raise ValueError('metadata packet phase is not warp-uniform')
    phase=next(iter(phases),0)
    new=deepcopy(old)
    new['streams']=[s for s in new['streams'] if not s['name'].startswith('metadata') or
                    int(s['name'].rsplit('part',1)[1]) in WORDS[phase]]
    new['metadata_words_per_active_lane']=len(WORDS[phase]) if active else 0
    new['metadata_shuffles_per_column']=4 if active else 0
    return {'register-fields':old,'sparse-metadata':new,
            'scope':'SOURCE_REQUEST_FOOTPRINT_NOT_DRAM_BYTES_OR_NATIVE_VECTOR_WIDTH'}
