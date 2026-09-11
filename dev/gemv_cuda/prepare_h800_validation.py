#!/usr/bin/env python3
"""Independent random-A checks reusing exactly the original Q4 weight bytes.

No producer or offline layout changes. New FP16-exact activations are dense,
not the original four-category fixture. Gold/conditioning are recomputed by
the official GGUF decoder and FP64 CPU dot, outside device timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

HEADER=struct.Struct("<Q8i8iQ8Q")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[93711,93719])
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in a.fixtures:
        data=path.read_bytes()
        header=data[:HEADER.size]
        h=HEADER.unpack(header)
        if h[0]!=0x3146584D5647514B or h[1:3]!=(1,12) or h[5:9]!=(1,1,1,0):
            raise ValueError("not an independent dense Q4 M1 fixture")
        n,k=h[3:5]
        chunks=[];position=HEADER.size
        for length in h[18:]:
            chunks.append(data[position:position+length]);position+=length
        if position!=len(data) or len(chunks)!=8 or len(chunks[4])!=k*4:
            raise ValueError("fixture lengths differ")
        raw=np.frombuffer(chunks[0],dtype=np.uint8)
        weights=dequantize(raw,GGMLQuantizationType.Q4_K).reshape(n,k).astype(np.float64)
        if not np.isfinite(weights).all(): raise ValueError("nonfinite reference weights")
        for seed in a.seeds:
            rng=np.random.default_rng(np.random.SeedSequence([seed,n,k]))
            act=rng.normal(0,0.3,k).astype("<f2").astype("<f4")
            gold=weights@act.astype(np.float64)
            denom=np.abs(weights)@np.abs(act.astype(np.float64))
            if not np.isfinite(gold).all() or not (denom>0).all(): raise ValueError("bad independent oracle")
            changed=list(chunks)
            changed[4]=act.tobytes()
            changed[6]=gold.astype("<f8").tobytes()
            changed[7]=denom.astype("<f8").tobytes()
            if [len(x) for x in changed]!=list(h[18:]): raise ValueError("new payload lengths differ")
            folder=a.output/f"seed-{seed}"
            folder.mkdir(exist_ok=True)
            target=folder/path.name
            with target.open("xb") as out:
                out.write(header)
                for payload in changed: out.write(payload)
            records.append(dict(path=str(target.relative_to(a.output)),shape=[1,n,k],seed=seed,
                                source_sha256=hashlib.sha256(data).hexdigest(),
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                weight_bytes_unchanged=True,oracle="OFFICIAL_GGUF_FP64_DOT",
                                input="DENSE_RANDOM_F16_EXACT"))
            print(f"Q4_H800_VALIDATION_FIXTURE seed={seed} shape=1x{n}x{k}",flush=True)
    (a.output/"manifest.json").write_text(json.dumps(records,indent=2)+"\n")


if __name__=="__main__": main()
