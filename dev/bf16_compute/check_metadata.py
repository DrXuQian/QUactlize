"""Bit gate for actual model SF producers and full BF16 weight expansion."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reference import gguf_kpack as ref
from dev.bf16_compute.fixture import bf16_bits, bf16_float
from dev.bf16_compute.native import Buffer, Resources, function
from quactlize.dequant.native import Call, TypedCall
from quactlize.execution.native import Arrangement, arrangement
from quactlize.runtime.native import SDK, checked
from tools.kpack_warmup_fixture import prepare_expert
from tools.kpack_dequant_fixture import compare
from tools.run_kpack_gemv_gate import metadata_oracle
from tools.verify_kpack_dispatch import verify


def fixture(q, n=256, k=512, experts=2):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    s = ref.SPECS[q]
    rng = np.random.default_rng(916001 + q)
    raw = rng.integers(0, 256, (experts, n, k // 256, s.raw_bytes), dtype="u1")
    for offset in (s.d_offset, s.dmin_offset):
        if offset >= 0:
            headers = rng.uniform(.005, .025, raw.shape[:-1]).astype("<f2")
            headers.flat[:4] = np.array([1, 0x203, 0x1357, 0x33ab], dtype="<u2").view("<f2")
            raw[..., offset:offset+2] = headers.view("u1").reshape(*headers.shape, 2)
    scales, minima = np.zeros((2, experts, k // s.group_size, n), dtype="f4")
    for e in range(experts):
        for col in range(n):
            for sb in range(k // 256):
                block = raw[e, col, sb].tobytes()
                d = np.frombuffer(block[s.d_offset:s.d_offset+2], dtype="<f2").astype("f4")[0]
                dm = np.frombuffer(block[s.dmin_offset:s.dmin_offset+2], dtype="<f2").astype("f4")[0] if s.has_min else np.float32(0)
                for g in range(s.groups):
                    sc, mn = ref._metadata_codes(block, 0, s, g)
                    sc = sc-32 if q == 11 else sc-256 if q == 14 and sc >= 128 else sc
                    scales[e, sb*s.groups+g, col] = d*np.float32(sc)
                    minima[e, sb*s.groups+g, col] = -dm*np.float32(mn) if s.has_min else 0
    scale = bf16_float(bf16_bits(scales))
    zero = bf16_float(bf16_bits(minima))
    zmul = {10: 0, 11: -4, 12: 8, 13: 8, 14: -24}[q]
    if zmul:
        zero = bf16_float(bf16_bits(zero + bf16_float(bf16_bits(np.float32(zmul)*scale))))
    golden = np.stack((bf16_bits(scale), bf16_bits(zero)))
    items = [prepare_expert(raw[e].reshape(-1, s.raw_bytes), q, n, k) for e in range(experts)]
    planes = {name: np.stack([p[name] for p in items]) for name in ("low", "high", "units")}
    full = bf16_bits(dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(experts, n, k))
    legacy = np.stack(metadata_oracle(planes["units"], q, n, k, experts)).view("<u2")
    if np.array_equal(bf16_bits(legacy[0].view("<f2").astype("f4")), golden[0]):
        raise ValueError("fixture cannot reject a decoded FP16 intermediate")
    return planes, golden, full, legacy


def run(bundle, sdk, q):
    n, k, experts = 256, 512, 2
    planes, golden, full, legacy = fixture(q, n, k, experts)
    execution = C.CDLL(str(bundle / "libquactlize_ppu_execution.so"), mode=C.RTLD_LOCAL)
    prefill = C.CDLL(str(bundle / "libquactlize_ppu_prefill.so"), mode=C.RTLD_LOCAL)
    sf = function(execution, "quactlize_kpack_sf_prepare_v2",
        [C.c_int]*4 + [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p, C.c_uint64,
                       C.POINTER(Arrangement), C.c_int, C.c_void_p])
    dequant = function(prefill, "quactlize_kpack_dequant_v2", [C.POINTER(TypedCall), C.POINTER(Arrangement)])
    r, proofs = Resources(sdk), []
    try:
        source = {name: r.upload(value) if value.size else None for name, value in planes.items()}
        sdk.synchronize(None)
        scale, zero = Buffer(r, golden[0].nbytes), Buffer(r, golden[1].nbytes)
        weights = Buffer(r, full.nbytes)
        arr = arrangement(q)
        for metadata_type, expected in ((0, legacy), (1, golden)):
            scale.poison(); zero.poison()
            checked(sf(q,n,k,experts,source["units"],planes["units"].nbytes,scale.ptr,zero.ptr,
                scale.size,C.byref(arr),metadata_type,r.stream), "typed SF prepare")
            sdk.synchronize(r.stream)
            proofs.append(dict(operation="sf-execution", metadata_type=metadata_type,
                scale=compare(scale.read("<u2", expected[0].shape), expected[0]),
                zero=compare(zero.read("<u2", expected[1].shape), expected[1])))
        for operation, configs in ((0, (0,4,5) if q in (12,13) else (0,)), (1, (0,4,5))):
            for config in configs:
                target = weights if operation else scale
                target.poison(); zero.poison()
                c = Call(1,C.sizeof(Call),q,n,k,experts,operation,config,
                    source["low"],source["high"],source["units"],planes["low"].nbytes,
                    planes["high"].nbytes,planes["units"].nbytes,target.ptr,None if operation else zero.ptr,
                    target.size,r.stream.value)
                typed = TypedCall(2,C.sizeof(TypedCall),c,1)
                checked(dequant(C.byref(typed),C.byref(arr)), "typed dequant")
                sdk.synchronize(r.stream)
                expected = full if operation else golden[0]
                proof = compare(target.read("<u2",expected.shape),expected)
                if not operation: proof["zero"] = compare(zero.read("<u2",golden[1].shape),golden[1])
                proofs.append(dict(operation="full-bf16" if operation else "sf-prefill",config=config,**proof))
        for invalid in (-1,2):
            scale.poison(); zero.poison()
            rc = sf(q,n,k,experts,source["units"],planes["units"].nbytes,scale.ptr,zero.ptr,
                scale.size,C.byref(arr),invalid,r.stream)
            sdk.synchronize(r.stream)
            if rc != 20 or np.any(scale.read("<u2") != 0xa5a5) or np.any(zero.read("<u2") != 0xa5a5):
                raise ValueError("invalid metadata precision changed output or was accepted")
        return dict(q=q,status="PASS",proofs=proofs,invalid_precision="REJECTED_WITHOUT_STORES")
    finally:
        sdk.synchronize(r.stream)
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True); p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args(); a.bundle = a.bundle.resolve(strict=True)
    verify(a.bundle,sdk=a.sdk)
    sdk = SDK(a.sdk)
    a.output.mkdir(parents=True,exist_ok=False)
    records = []
    for q in range(10,15):
        record = run(a.bundle,sdk,q)
        records.append(record)
        (a.output/f"q{q}.json").write_text(json.dumps(record,indent=2)+"\n")
        print(f"BF16_METADATA_GATE q={q} status=PASS SF=F16+BF16 FULL=BF16",flush=True)
    (a.output/"summary.json").write_text(json.dumps(dict(status="PASS",formats=5,records=records),indent=2)+"\n")


if __name__ == "__main__":
    main()
