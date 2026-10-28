"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
def parents():
    result = []
    for q, tk in ((12,128), (13,256)):
        for quant in ("fq", "sf"):
            for compute in ("f16", "bf16"):
                key = f"matched_q{q}_{quant}_{compute}"
                result.append((key, compute, dict(qtype=q,route=quant+"-grouped",tm=64,tn=128,tk=tk,
                    wm=64,wn=32,stages=2,ap=0,dn=16,persistent=0 if quant=="fq" else -1,symbol=key)))
    return result
