"""Small public-reader floor, independent of ANY measurement/table.

Q4 has a specialized image, unlike generic all-format SIMT. A leave-NK-out
catalog can lose an entire reader family (e.g. medium). Reinstantiate existing
reader bodies; do not inject the hidden policy entry. This is an experimental
inventory, NOT a production fallback or a claim of measured performance.

row_medium / row_reuse: worker=tid/C, col=tile*(C*P)+(tid%C)*P;
B byte address=2*((g*8+r)*N+col), vector width=2*P. C4/P4,
C4/P8, C8/P4 cover contiguous 32/64/64-byte column cohorts. No
layout/decoder changes: reuse V6/7 retain cooperative packed metadata, and
V7 also cooperates on A. Actual coalescing/traffic needs native ISA and ACU.
medium workers=W*32/C process K/32 groups. For C4, W=K/(256*passes)
has no partial group pass. Probe two/four passes when integral and <=32
warps, plus W8. None of this uses N/K exceptions or historical winners.
Reuse additionally tries the exact warp count eliminating one of W8's K
passes, if integral and <=16 warps. This is a local loop-bound neighbor,
not a warp x columns x values Cartesian product or N/K lookup exception.
"""
from quactlize.dispatch.table import Recipe


def proposals(w):
    if w.qtype != 12 or w.m > 8 or w.operation == 'grouped':
        return ()
    out = [Recipe.make('q4','fq',reader=0,variant=1,warps=8,values=1,columns=4)]
    balanced = {8}
    for passes in (2,4):
        denominator=256*passes
        if w.k%denominator==0 and 1<=w.k//denominator<=32:
            balanced.add(w.k//denominator)
    out += [Recipe.make('q4','fq',reader=1,variant=1,warps=b,values=4,columns=4)
            for b in sorted(balanced)]
    for c,p in ((4,4),(4,8),(8,4)):
        warps={8}
        passes=(w.k//32 + 8*32//c-1)//(8*32//c)
        if passes>1:
            denominator=1024*(passes-1)
            if (w.k*c)%denominator==0 and 1<=w.k*c//denominator<=16:
                warps.add(w.k*c//denominator)
        out += [Recipe.make('q4','fq',reader=2,variant=v,warps=b,values=p,columns=c)
                for b in sorted(warps) for v in (6,7)]
    return tuple(out)
