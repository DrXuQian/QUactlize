"""Actual source load addresses for both planes, A and packed metadata."""
from dev.gemv_ppu.access_pattern import footprint
from reference.gguf_kpack import SPECS, _placed_word_slot


def plane_word(q, high, col, k, n):
    bits = 8 if q==8 else (SPECS[q].high_bits if high else SPECS[q].low_bits)
    if high and q==13:
        pn = (col&~15)|(col&7)|(((k>>7)&1)<<3)
        kg = (k//256)*16|(((k>>6)&1)<<3)|(k&7)
        return kg*n+pn
    return _placed_word_slot(k,bits)[0]*n+col


def pattern(q, config, n, k, storage=1, *, bases=None, warp=0, partition=0, iteration=0):
    if q==8 and config.variant>=4:
        from dataclasses import replace
        from dev.gemv_simt.q8_vector_access import pattern as vector_pattern
        if storage!=1:
            raise ValueError('Q8 vector reader requires F32 storage')
        return vector_pattern(replace(config,variant=config.variant-4),n,k,bases=bases,
                              warp=warp,partition=partition,iteration=iteration)
    c = config
    group = 32 if q in (8,12,13) else 16
    low_bits = 8 if q==8 else SPECS[q].low_bits
    high_bits = 0 if q==8 else SPECS[q].high_bits
    unit = 2 if q==8 else SPECS[q].unit_bytes
    unit_groups = 1 if q==8 else SPECS[q].groups*SPECS[q].superblocks_per_unit
    bases = dict(A=0,low=0,high=0,units=0) if bases is None else dict(bases)
    if set(bases)!={'A','low','high','units'} or any(v<0 or v>=128 for v in bases.values()):
        raise ValueError('invalid actual base modulo128')
    workers = c.warps*32//c.columns
    groups = [partition*workers+(warp*32+lane)//c.columns+iteration*c.split*workers for lane in range(32)]
    active = [lane for lane,g in enumerate(groups) if g<k//group]
    if len(active) not in (0,32):
        raise ValueError('partial active warp violates exchange contract')
    columns = [(lane%c.columns)*c.values for lane in range(32)]
    streams = []
    def stream(name, addresses, width, lanes):
        streams.append(dict(name=name, lanes=lanes, addresses=addresses, width_bytes=width,
            footprint={str(s):footprint(addresses,width,s) for s in (32,64,128)}))
    segments = (group+8*(16//low_bits)-1)//(8*(16//low_bits))
    for high in (False,True):
        if high and not high_bits: continue
        for segment in range(1 if high else segments):
            for residue in range(8):
                ks = [g*group+segment*8*(16//low_bits)+residue for g in groups]
                name = 'high' if high else 'low'
                stream(f'{name}-segment{segment}-residue{residue}',
                    [bases[name]+2*plane_word(q,high,columns[l],ks[l],n) for l in active],
                    2*c.values,active)
    count = group//c.columns if c.variant&1 else 4
    input_bytes = 4 if storage else 2
    # Count8 F32 is two float4 requests; direct F16 values4 is two half2
    # requests. Model source instructions separately, not a fictitious float4.
    width = min(16,count*input_bytes) if c.variant&1 else (16 if storage else 4)
    offsets = [0] if c.variant&1 else [slot*8+h*4 for h in range(2) for slot in range(group//8)]
    for offset in offsets:
        address = [bases['A']+input_bytes*(groups[l]*group+offset+
            ((l%c.columns)*count if c.variant&1 else 0)) for l in active]
        reads = count*input_bytes//width
        for part in range(reads):
            stream(f'A-offset{offset}-part{part}',[p+part*width for p in address],width,active)
    for p in range(1 if c.variant&2 else c.values):
        lanes = [l for l in active if l<c.tile_n] if c.variant&2 else active
        address = [bases['units']+unit*((groups[l]//unit_groups)*n+
            (l if c.variant&2 else columns[l]+p)) for l in lanes]
        width = 2 if q==8 else 16 if q in (12,13) else 4
        for part in range(unit//width):
            stream(f'metadata-p{p}-part{part}',[v+part*width for v in address],width,lanes)
    return dict(qtype=q,config=c.key,shape=[n,k],input_storage='F32' if storage else 'F16',
        actual_bases_mod128=bases,warp=warp,partition=partition,iteration=iteration,
        lane_groups=groups,lane_columns=columns,streams=streams,
        low_word_group_reuse=8*(16//low_bits)/group,
        high_word_group_reuse=8*(16//high_bits)/group if high_bits else None,
        note='Packed words may be requested by several scale-group workers. Duplicate lane bytes are not DRAM bytes.',
        scope='SOURCE_LOAD_FOOTPRINT_NOT_HARDWARE_COUNTERS')
