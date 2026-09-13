"""Byte-address models for the unchanged readers, including actual F32 A strides."""
from dev.gemv_ppu import small_latency, reader_reuse
from dev.gemv_ppu.config_space import Config


def access(config, base):
    n,k=base.n,base.k
    mod=dict(A=base.a%128,B=base.base.low%128,metadata=base.base.units%128)
    if config.reader==2:
        packed=reader_reuse.access(reader_reuse.Reader(config.variant,Config(config.columns,config.warps,config.values)),n,k,mod)
    else:
        c=small_latency.Candidate('meta' if config.reader==0 else 'affine',
            config.variant if config.reader==0 else 1,1 if config.reader==0 else int(config.values==2),
            1 if config.reader==0 else int(config.values==2),config.warps,
            config.columns,config.values)
        packed=small_latency.access(c,n,k,mod)
    streams=[]
    # All row origins are known from the fixture, never a timed CPU router.
    origins=sorted({int(r)*(k+8)*4 for r in base.data['arows']})
    for origin in sorted({origins[0],origins[-1]}):
        if config.reader==0:
            offsets=[(lane//8)*32+lane%8 for lane in range(32)];width=4;second=None
        elif config.reader==2 and config.variant&1:
            offsets=[(lane//config.columns)*32+(lane%config.columns)*(32//config.columns) for lane in range(32)]
            width=16;second=16 if config.columns==4 else None
        else:
            offsets=[(lane//config.columns)*32 for lane in range(32)];width=16
            second=16 if config.reader==1 and config.values==2 else None
        addresses=[mod['A']+origin+4*x for x in offsets]
        streams.append(dict(row_origin_bytes=origin,lane_byte_addresses=addresses,width_bytes=width,
            additional_load_offset_bytes=second,
            granules={str(g):small_latency.footprint(addresses,width,g) for g in (32,64,128)}))
    return dict(scope='SOURCE_FOOTPRINT_NOT_MEASURED_TRAFFIC',
        immutable_f16_reader_address_model=packed,f32_A_first_instruction=streams,
        f32_a_row_stride_bytes=(k+8)*4,f32_output_row_stride_bytes=(n+8)*4,
        expert_low_stride_bytes=n*k//2,expert_units_stride_bytes=n*k//16,
        actual_plane_bases_mod128=mod,rows=base.workload['rows'],
        geometry=config.geometry(n,base.workload['rows']),
        cross_row_b_reuse='CACHE_ONLY_NO_EXPLICIT_STAGING',
        note='B/metadata addresses are unchanged by F32 A. F16 A entries in the control model are not the active F32 requests.')
