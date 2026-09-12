"""Address-derived warp footprints, not a simulation of caches or DRAM traffic."""
from collections import Counter
import re


def footprint(addresses, width, granule):
    used={a+i for a in addresses for i in range(width)}
    sectors={b//granule for b in used}
    runs=0;last=None
    for b in sorted(used):
        if last is None or b!=last+1:runs+=1
        last=b
    return dict(lane_bytes=len(addresses)*width,unique_bytes=len(used),
                duplicate_factor=len(addresses)*width/len(used) if used else 0,
                sectors=len(sectors),sector_bytes=len(sectors)*granule,
                unique_sector_utilization=len(used)/(len(sectors)*granule) if sectors else 0,
                contiguous_runs=runs)


def warp_pattern(config,n,k,*,block=0,warp=0,pass_index=0,base_mod128=None):
    bases=dict(A=0,B=0,metadata=0) if base_mod128 is None else dict(base_mod128)
    if set(bases)!={"A","B","metadata"} or any(type(x) is not int or not 0<=x<128 for x in bases.values()):
        raise ValueError("invalid plane-base alignment model")
    c,w,p=config.args;workers=w*32//c
    lanes=[]
    for lane in range(32):
        tid=warp*32+lane;g=pass_index*workers+tid//c;col=block*c*p+(tid%c)*p
        if g<k//32:lanes.append((lane,g,col))
    # These represent source-level load groups. A consists of two half2
    # loads; LLVM may merge them across the unrolled half loop. Native load
    # width histograms are reported separately, never inferred from C++.
    streams={
        "B_vector":([bases["B"]+2*(g*8*n+col) for _,g,col in lanes],2*p),
        "metadata_p0":([bases["metadata"]+16*((g//8)*n+col) for _,g,col in lanes],16),
        "A_half2":([bases["A"]+2*(g*32) for _,g,_ in lanes],4),
        "A_4half_combined":([bases["A"]+2*(g*32) for _,g,_ in lanes],8),
        "A_8half_vector_model":([bases["A"]+2*(g*32) for _,g,_ in lanes],16),
    }
    return dict(base_mod128=bases,active_lanes=len(lanes),lane_group=[g for _,g,_ in lanes],lane_n=[col for _,_,col in lanes],
        streams={name:dict(lane_byte_addresses=addresses,width_bytes=width,
                  granules={str(granule):footprint(addresses,width,granule) for granule in (32,64,128)})
                  for name,(addresses,width) in streams.items()})


def analyze(config,n,k):
    g=config.geometry(n,k);last_pass=g["k_passes"]-1
    last_warp=(g["last_pass_workers"]*config.columns-1)//32
    return dict(config=config.key,geometry=g,
        model="SOURCE_WARP_UNIQUE_FOOTPRINT_NOT_DRAM_OR_MEASURED_TRANSACTION_COUNT",
        alignment_assumption="plane base 128-byte aligned; source ABI requires only 16; addresses are offsets",
        B_contiguous_bytes_per_k_group=2*config.columns*config.values,
        logical_lane_bytes_per_call=dict(B=n*k//2,metadata=n*k//2,A=2*n*k//config.values),
        distinct_input_bytes=dict(B=n*k//2,metadata=n*k//16,A=2*k),
        first_warp=warp_pattern(config,n,k),
        next_cta=warp_pattern(config,n,k,block=1),
        last_active_warp=warp_pattern(config,n,k,warp=last_warp,pass_index=last_pass))


def observed_patterns(config,n,k,a,weight_pointers):
    """One first-warp model per actually observed ring alignment; never time it."""
    alignments=sorted({(int(a)%128,int(low)%128,int(units)%128) for low,units in weight_pointers})
    if not alignments or any(v%16 for offsets in alignments for v in offsets):
        raise ValueError("config plane does not meet its native 16-byte alignment contract")
    return [warp_pattern(config,n,k,base_mod128=dict(zip(("A","B","metadata"),offsets))) for offsets in alignments]


def isa_histograms(text):
    pattern=re.compile(r"Disassembly of section \.text\.kernel\.[^\n]*q4_group_affineILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb1E[^\n]*:")
    result={}
    for match in pattern.finditer(text):
        end=text.find("Disassembly of section ",match.end())
        body=text[match.end():end if end>=0 else None]
        counts=Counter(re.findall(r"\t([a-z][\w.]+)\s",body))
        c,w,p,n,k=map(int,match.groups())
        key=f"n{n}-k{k}-c{c}-w{w}-p{p}"
        result[key]=dict(scope="STATIC_ISA_COUNTS_NOT_DYNAMIC_INSTRUCTIONS",
            loads={op:count for op,count in counts.items() if op.startswith("vmem.ld.")},
            code_fastpath_present=bool(counts["v.lop3.b32"] and
                                      (counts["v.add.f16x2"] or counts["v.fma.f16x2"])),
            fp32_fma_present=any(op.startswith("v.fma.f32") for op in counts),
            operations={op:count for op,count in counts.items()
                        if any(x in op for x in ("lop3","f16x2","fma.f32","shrl.b64","conv","cvt"))})
    return result
