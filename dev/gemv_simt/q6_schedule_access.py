"""Physical request model for the bounded W2/4/8 Q6 reader experiment."""
from dev.gemv_simt.access import pattern as original_pattern
from dev.gemv_simt.q6_sparse_access import WORDS


def pattern(q, config, n, k, storage=1, **kwargs):
    if (q != 14 or config.warps not in (2,4,8) or
            (config.variant,config.columns,config.values,config.split) != (3,4,8,1)):
        raise ValueError('schedule requires Q6 V3 C4 W2/4/8 P8 S1')
    row = original_pattern(q, config, n, k, storage, **kwargs)
    active = [g for g in row['lane_groups'] if g < k//16]
    phases = {g//8%4 for g in active}
    units = {g//32 for g in active}
    if len(phases)>1 or len(units)>1:
        raise ValueError('sparse metadata phase/unit is not warp-uniform')
    phase = next(iter(phases), 0)
    row['streams'] = [s for s in row['streams'] if not s['name'].startswith('metadata') or
                      int(s['name'].rsplit('part',1)[1]) in WORDS[phase]]
    row.update(metadata_words_per_active_lane=len(WORDS[phase]) if active else 0,
               metadata_shuffles_per_column=4 if active else 0,
               pipeline_address_change=False, scope='SAME_WARP_IDENTICAL_REQUESTS_ISSUE_ORDER_ONLY_NOT_DRAM')
    return row
