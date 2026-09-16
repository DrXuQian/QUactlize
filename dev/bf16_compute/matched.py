"""Same-geometry F16/BF16 MoE prefill comparison, separate from expansion."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parents():
    result = []
    for q, tk in ((12,128), (13,256)):
        for quant in ("fq", "sf"):
            for compute in ("f16", "bf16"):
                key = f"matched_q{q}_{quant}_{compute}"
                result.append((key, compute, dict(qtype=q,route=quant+"-grouped",tm=64,tn=128,tk=tk,
                    wm=64,wn=32,stages=2,ap=0,dn=16,persistent=0 if quant=="fq" else -1,symbol=key)))
    return result


def measure(root, package, sdk, q, quant):
    from dev.bf16_compute.fixture import Weights, compare
    from dev.bf16_compute.native import TensorCore, Resources
    from tools.run_kpack_grouped_decode_probe import Replay
    from quactlize.runtime.native import checked
    n, k = (1024,2048) if q==12 else (2048,512)
    tokens, experts, rows_per_expert = 2048,256,64
    rows = tokens*8
    weights = Weights(q,n,k,experts,pool_size=1)
    r = Resources(sdk)
    kernels, graphs = {}, {}
    samples = {compute: [] for compute in ("f16", "bf16")}
    try:
        act = np.ones((rows,k),dtype="f4")
        # All expert slices carry the same independent dyadic GGUF fixture.
        # This closes a large-M oracle without an enormous CPU matrix multiply.
        w = weights.weight(0).astype("f8")
        gold = np.broadcast_to(w.sum(axis=1), (rows,n))
        denom = np.broadcast_to(np.abs(w).sum(axis=1), (rows,n))
        for compute in samples:
            record = package["matched_modules"][f"matched_q{q}_{quant}_{compute}"]
            kernel = TensorCore(root,record,sdk,r,weights,rows,rows_per_expert)
            kernels[compute] = kernel
            kernel.offsets.upload(np.arange(experts+1,dtype="i4")*rows_per_expert)
            kernel.upload_a(act)
            graph = Replay(sdk,r.stream,kernel.run,1)
            graphs[compute] = graph
            checked(graph(), "matched prefill correctness")
            sdk.synchronize(r.stream)
            compare(kernel.read(),gold,denom)
            kernel.workspace.guard()
            for _ in range(3): checked(graph(), "excluded matched warmup")
        sdk.synchronize(r.stream)
        rounds=[]
        for repeat in range(3):
            blocks=[]
            for compute in ("f16","bf16","bf16","f16"):
                values=r.samples(graphs[compute],11)
                samples[compute].extend(values)
                blocks.append(dict(compute=compute,samples_us=values))
            rounds.append(blocks)
        median={compute:statistics.median(values) for compute,values in samples.items()}
        return dict(q=q,quant=quant,status="PASS",tokens=tokens,rows=rows,n=n,k=k,experts=experts,
            rows_per_expert=rows_per_expert,median_us=median,
            bf16_delta_pct=100*(median["bf16"]/median["f16"]-1),rounds=rounds,
            kernels={name:kernel.receipt for name,kernel in kernels.items()},
            scope="SAME_CONFIG_RESIDENT_GEMM_PLUS_GPU_DIRECTORY_NO_SF_PREPASS",
            warmup_excluded=True,heuristic_admitted=False)
    finally:
        sdk.synchronize(r.stream)
        for graph in graphs.values(): graph.close()
        for kernel in kernels.values(): kernel.close()
        r.close()


def main():
    from dev.bf16_compute.run import validate_package
    from quactlize.runtime.native import SDK
    from tools.run_kpack_grouped_device_gate import graph_bind
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,required=True);p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();package=validate_package(a.package)
    if set(package.get("matched_modules",{})) != {key for key,_,_ in parents()}:
        raise ValueError("same-config comparison modules missing")
    sdk=SDK(a.sdk);graph_bind(sdk);a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for q in (12,13):
        for quant in ("fq","sf"):
            print(f"BF16_MATCHED_START q={q} quant={quant}",flush=True)
            record=measure(a.package,package,sdk,q,quant);records.append(record)
            (a.output/f"q{q}-{quant}.json").write_text(json.dumps(record,indent=2)+"\n")
            print("BF16_MATCHED_RESULT "+json.dumps({k:v for k,v in record.items() if k not in ("rounds","kernels")}),flush=True)
    (a.output/"summary.json").write_text(json.dumps(dict(status="PASS",records=records),indent=2)+"\n")


if __name__=="__main__": main()
