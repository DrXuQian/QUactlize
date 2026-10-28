"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import math
import statistics
from quactlize.dequant.native import traffic, config_ids, selected_configs
from reference import gguf_kpack as ref
DENSE=((1024,5120),(5120,8192),(5120,25600),(8192,5120),(25600,5120))


GROUPED=((512,2048),(512,3072),(2048,512),(3072,512),(1024,2048),(1024,3072))


def work(qtypes):
    rows=[]
    for q in range(10,15):
        rows.extend(dict(q=q,n=256,k=512,experts=3,operation=o,smoke=True) for o in (0,1))
    for q in qtypes:
        shapes=[(n,k,1) for n,k in DENSE]+[(n,k,e) for n,k in GROUPED for e in (8,256)]
        rows.extend(dict(q=q,n=n,k=k,experts=e,operation=o,smoke=False)
                    for n,k,e in shapes for o in (0,1))
    for r in rows:
        r['id']=f'q{r["q"]}-n{r["n"]}-k{r["k"]}-e{r["experts"]}-'+('full' if r['operation'] else 'sf')
        if r['smoke']:r['id']+='-smoke'
    return rows


def selected_work(qtypes,inventory):
    tasks=work(qtypes)
    if inventory=='all':return tasks
    if inventory in ('full-reader','full-packed') and qtypes and all(q in (12,13) for q in qtypes):
        return [w for w in tasks if w['operation']==1 and w['q'] in qtypes]
    raise ValueError('full-reader inventory is Q4/Q5 full dequant only')


def control_config(record):
    w=record['workload']
    generation=record.get('config_generation',1)
    prior=config_ids(w['q'],w['operation'],max(1,generation-1))
    controls=[r for r in record['rows'] if r['config'] in prior]
    if not controls:raise ValueError('previous best controls missing')
    return min(controls,key=lambda r:r['median_us'])['config']


def pattern(w, config):
    """First warp/pass byte footprint, not measured DRAM traffic."""
    q,n,k=w['q'],w['n'],w['k'];s=ref.SPECS[q]
    if w['operation']:
        coords=[(0,lane) if config==0 else ((lane%4)*8,lane//4) if config>=5 else (lane,0) for lane in range(32)]
    else:
        cols=8 if config==0 else 16 if config==1 else 32
        coords=[(lane//4,(lane%4)*s.group_size) if config==0
                else (lane%cols,(lane//cols)*s.group_size) for lane in range(32)]
    fields={}
    def field(name, addresses, width):
        unique={b for a in addresses for b in range(a,a+width)}
        fields[name]=dict(lane_requests=len(addresses),width=width,unique_bytes=len(unique),
            requested_bytes=width*len(addresses),sectors={str(size):len({b//size for b in unique}) for size in (32,64,128)},
            byte_addresses=addresses)
    if w['operation']:
        field('low_word',[2*(ref._placed_word_slot(kk,s.low_bits)[0]*n+col) for col,kk in coords],16 if config>=5 else 2)
        if s.high_bits:
            addresses=[]
            for col,kk in coords:
                if q==13:
                    pn=(col&~15)|(col&7)|(((kk>>7)&1)<<3)
                    kg=(kk//256)*16|(((kk>>6)&1)<<3)|(kk&7)
                else:pn,kg=col,ref._placed_word_slot(kk,s.high_bits)[0]
                addresses.append(2*(kg*n+pn))
            field('high_word',addresses,16 if config>=5 else 2)
    headers=[];codes=[];mins=[]
    for col,kk in coords:
        sb=kk//256;g=(kk//s.group_size)%s.groups
        unit=((sb//s.superblocks_per_unit)*n+col)*s.unit_bytes+(sb%s.superblocks_per_unit)*s.sb_bytes
        headers.append(unit);codes.append(unit+ref._unit_bit(s,g,0)//8)
        if s.has_min:mins.append(unit+ref._unit_bit(s,g,1)//8)
    if w['operation'] and config>=10:
        field('metadata_unit16',headers[:4],16)
    elif w['operation'] and config>=7:
        field('metadata_cta_unit16',[lane*16 for lane in range(32)],16)
    elif (not w['operation'] and config>=4) or (w['operation'] and config>=5 and q in (12,13)):
        field('metadata_unit16',headers[:4] if w['operation'] else headers,16)
    else:
        select=slice(0,4) if w['operation'] and config==5 else slice(None)
        field('header_d_u16',headers[select],2);field('first_scale_byte',codes[select],1)
        if s.has_min:
            field('header_dmin_u16',[x+2 for x in headers[select]],2);field('first_min_byte',mins[select],1)
    if w['operation'] and config>=4:
        per_row=32 if config in (8,11,12) else 16
        field('bf16_output_uint4',[(i//per_row*k+i%per_row*8)*2 for i in range(32)],16)
    elif w['operation'] and config:
        per_row=64 if config==3 else 16
        field('bf16_output_pairs',[(i//per_row*k+i%per_row*2)*2 for i in range(32)],4)
    else:
        field('output_scalar',[2*(col*k+kk if w['operation'] else (kk//s.group_size)*n+col) for col,kk in coords],2)
    details={}
    if w['operation'] and config>=10:
        tile_k=128 if config==10 else 256
        details=dict(tile_n=32,tile_k=tile_k,threads=128,a_bytes=0,
            cta_count=(w.get('experts',1)*n//32)*(k//tile_k),
            shared_weight_bytes=32*tile_k,shared_affine_bytes=8*tile_k,
            shared_representation='FOUR_RAW_CODES_PER_UINT32_BEFORE_FP32_AFFINE',
            metadata_requested_bytes_per_n_superblock=128 if tile_k==128 else 64,
            cta_barriers=1,scale_broadcast_shuffles=0,
            cta_order='K_FAST' if config==12 else 'N_FAST',
            shared_layout='PackedExchangeLayout; exact host coverage/vector/bank tests',
            shared_load='two aligned uint4 code reads plus float2 affine per output vector')
    elif w['operation'] and config>=6:
        stage_k=256 if config==8 else 128;tile_k=256 if config>=8 else 128
        details=dict(tile_n=32,tile_k=tile_k,stage_k=stage_k,threads=128,a_bytes=0,
            cta_count=(w.get('experts',1)*n//32)*(k//tile_k),
            shared_weight_bytes=32*stage_k*4,shared_metadata_bytes=512 if config>=7 else 0,
            metadata_requested_bytes_per_n_superblock=128 if config==6 else 32 if config==7 else 16,
            cta_barriers={6:1,7:2,8:2,9:4}[config],
            shared_layout='CuTe Swizzle<3,2,3>(N*StageK+K) xor (N&24)',
            shared_load='two aligned uint4 reads per output vector; PPU bank counters decide benefit')
    return dict(scope='FIRST_WARP_PASS_SOURCE_ADDRESS_MODEL_BASE_ALIGNED_128',fields=fields,**details,
                full_arithmetic='FP32_RAW_GGUF_NOT_FP16_FAST_DEQUANT',
                metadata_straddles='second byte only for crossing fields; first-byte footprint shown')


def validate_result(r,w,dev,peak):
    if (r.get('status')!='PASS' or r.get('workload')!=w or r.get('device')!=dev or
        r.get('gemm_calls')!=0 or r.get('scope')!='DEQUANT_ONLY_NOT_COMBINED_OR_REUSABLE_CACHE' or
        r.get('peak_gbps')!=peak or r.get('bytes')!=traffic(w['q'],w['n'],w['k'],w['experts'],w['operation'])):
        raise ValueError('dequant result identity/scope differs: '+w['id'])
    configs=selected_configs(w['q'],w['operation'],r.get('config_generation',1),r.get('inventory','all'))
    if [v['config'] for v in r['rows']]!=configs:raise ValueError('dequant result config set differs')
    for row in r['rows']:
        p=row['proof'];values=row['samples_us']
        if (p['bad'] or p['guard']!='PASS' or p['negative_bad']<=0 or
            (not w['operation'] and p['signed_zero_differences'])):raise ValueError('invalid dequant proof')
        if w['smoke']:
            if values or row['median_us'] is not None:raise ValueError('smoke silently timed')
        else:
            if len(values)!=15 or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid dequant samples')
            if row['median_us']!=statistics.median(values):raise ValueError('dequant median differs')
            if row['round_medians_us']!=[statistics.median(values[i:i+5]) for i in range(0,15,5)]:raise ValueError('round medians differ')
            if row['effective_gbps']!=r['bytes']['useful_bytes']/row['median_us']/1000:raise ValueError('effective byte model differs')
            if row['effective_pct']!=row['effective_gbps']/peak*100:raise ValueError('effective percent differs')
            if r['input_ring_bytes']!=r['bytes']['reads']*r['copies'] or r['output_ring_bytes']!=r['bytes']['writes']*r['copies']:
                raise ValueError('ring byte accounting differs')
            if r['calls_per_graph']!=r['copies']*2:raise ValueError('incomplete ring traversal')
            if r['input_ring_bytes']<2.25*dev['l2_bytes'] or r['output_ring_bytes']<2.25*dev['l2_bytes']:
                raise ValueError('dequant ring is not larger than L2')
    return r
