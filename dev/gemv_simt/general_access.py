"""Source-level 32/64/128-byte footprints; not measured DRAM traffic."""
from copy import deepcopy
from dev.gemv_simt.access import pattern as original


def pattern(q,config,n,k,**kw):
    old=original(q,config,n,k,**kw);new=deepcopy(old)
    if q in (10,11,14):
        live=[g for g in old['lane_groups'] if g<k//16]
        phases={g//8%(2 if q==10 else 4) for g in live}
        if len(phases)>1:raise ValueError('nonuniform metadata cohort')
        phase=next(iter(phases),0)
        words={10:((0,1,2),(0,3,4)),11:((0,1),(0,2,3),(3,4,5),(3,5,6)),
               14:((0,1,2),(0,2,3,4),(4,5,6),(4,7,8))}[q][phase]
        new['streams']=[s for s in new['streams'] if not s['name'].startswith('metadata') or
                        int(s['name'].rsplit('part',1)[1]) in words]
        new['metadata_words_per_lane']=len(words) if live else 0
        if q==14 and config.columns==4 and config.values==8:
            # The two C4 controls already use A3 sparse packets. Do not
            # attribute A1/A2's old traffic savings to this generalization.
            old=deepcopy(new)
    return dict(control=old,candidate=new,scope='SOURCE_REQUESTS_NOT_DRAM_OR_VECTOR_WIDTH_PROOF')
