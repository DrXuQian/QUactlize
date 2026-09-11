"""Frozen H800 implementations, mechanically translated to native PPU APIs.

This is an experiment contract, not shipping admission or a new offline map.
Each implementation lives in its own DSO to keep signed/unsigned half helpers
from interposing across translation units with different arithmetic.
"""
import json
from pathlib import Path
import re

from dev.gemv_cuda.build import digest
from dev.gemv_cuda.build_h800_candidates import source as cuda_source
from dev.gemv_cuda.summarize_h800_confirmation import POLICY
from dev.gemv_ppu.build import ppu_api
from dev.gemv_ppu.run import verify_bundle

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "quactlize.q4-h800-port-ppu.v1"
IMPLEMENTATIONS = {
    "small": "meta-static-global-rs-fast-bare-av",
    "medium": "affine8-early-fast-bare-a4",
    "large": "affine4-early-fast-bare",
}
REFERENCE_RECIPES = [(c, w, kw) for c in (1, 2, 4, 8) for w in (1, 2, 4, 8)
                     for kw in (1, 2, 4, 8) if w * kw <= 32]
XPLANE_RECIPES = [(c, w, 1) for c in (1, 2, 4, 8) for w in (2, 4, 8)]


def selection(n, k):
    name, recipe = POLICY[(1, n, k)]
    family = next(key for key, value in IMPLEMENTATIONS.items() if value == name)
    return family, list(recipe)


def candidate_source(family):
    name = IMPLEMENTATIONS[family]
    body, _ = cuda_source(name)
    start = body.index('extern "C" int qkg_pair_launch_12(')
    first = body.index("    if(f.columns==", start)
    launches = []
    for (_, n, k), (arm, (c, w, _)) in POLICY.items():
        if arm != name:
            continue
        pattern = (rf"    if\(f.columns=={c} && f.warps=={w} && c.n=={n} && c.k=={k}\)"
                   r" \{\n.*?\n    \}\n")
        found = re.findall(pattern, body[first:], flags=re.S)
        if len(found) != 1:
            raise ValueError("frozen launch seam differs: " + pattern)
        launches.append(found[0])
    body = body[:first] + "".join(launches) + "    return QKG_INVALID;\n}\n"
    # Expand only the helper includes used by these bodies. The generated
    # body otherwise stays identical, including FP32 order and ownership.
    for helper in ("q4_warp_activation.cuh", "q4_warp_reduce_scatter.cuh"):
        path = ROOT / "dev/gemv_cuda" / helper
        body = body.replace(f'#include "{path}"', path.read_text())
    body += '''
extern "C" int q4_h800_port_run(int n,int k,void const* a,void const* low,
        void const* units,void* output,void* stream) {
    if(!a || !low || !units || !output || (uintptr_t(output)&3)) return QKG_INVALID;
    qkg_call_v1 call{};
    call.n=n;call.k=k;call.mode=QKG_DENSE;call.rows=1;call.experts=1;
    call.input_type=QKG_F16;call.a=a;call.low=static_cast<uint8_t const*>(low);
    call.units=static_cast<uint8_t const*>(units);call.output=static_cast<float*>(output);call.stream=stream;
    qkg_config_v1 recipe{};recipe.split=1;
'''
    for (_, n, k), (arm, (c, w, _)) in POLICY.items():
        if arm == name:
            body += (f"    if(n=={n} && k=={k}) {{recipe.columns={c};recipe.warps={w};"
                     "return qkg_pair_launch_12(call,recipe);}\n")
    return ppu_api(body + "    return QKG_SHAPE;\n}\n")


def reference_source():
    body = '''#include "gemv_ref_fp32.cuh"
#include "bload_contract.hpp"
extern "C" int q4_ref_fp32_run_v2(int c,int w,int kw,int n,int k,
        void const* a,void const* raw,void* output,void* stream) {
    if(!q4_bload::shape(n,k) || !a || !raw || !output ||
       (uintptr_t(a)&15) || (uintptr_t(raw)&15) || (uintptr_t(output)&3)) return -1;
'''
    for c, w, kw in REFERENCE_RECIPES:
        body += f'''    if(c=={c} && w=={w} && kw=={kw}) {{
        q4k_gemv_fp32::launch_q4k_gemv<{c},{w},{kw}>(static_cast<half const*>(a),
            static_cast<q4k_gemv_fp32::block_q4_K const*>(raw),static_cast<float*>(output),
            1,n,k,static_cast<hggcStream_t>(stream));
        return int(hggcGetLastError());
    }}
'''
    return body + "    return -1;\n}\n"


def verify(candidate, baseline, *, sources=True):
    control = verify_bundle(baseline, sources=sources)
    data = json.loads((candidate / "manifest.json").read_text())
    if (data["schema"] != SCHEMA or data["baseline_manifest_sha256"] != digest(baseline / "manifest.json")
            or data["compiler_sha256"] != control["compiler_sha256"]
            or data["reference_recipes"] != [list(r) for r in REFERENCE_RECIPES]
            or data["implementations"] != IMPLEMENTATIONS
            or set(data["payloads"]) != {*IMPLEMENTATIONS, "reference"}):
        raise ValueError("PPU port package/control identity differs")
    for arm, row in data["payloads"].items():
        path = candidate / row["file"]
        if row["file"] != f"libq4_ppu_port_{arm}.so" or digest(path) != row["sha256"]:
            raise ValueError("PPU port payload differs or LFS pointer: " + arm)
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("not a native ELF: " + arm)
    for name, expected in data["source_hashes"].items() if sources else []:
        path = (ROOT / name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("PPU port source differs: " + name)
    return data
