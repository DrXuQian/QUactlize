#!/usr/bin/env python3
"""Extend only the FP32 reference launch inventory; keep its kernel unchanged."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_cuda.build import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k-warps", action="store_true",
                   help="also check the supplied reader's intra-CTA K warps; no inter-CTA reduction")
    a = p.parse_args(); a.control = a.control.resolve(strict=True)
    a.output = a.output.resolve(); a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.control / "manifest.json").read_text())
    for name, sha in manifest["payloads"].items():
        if digest(a.control / name) != sha: raise ValueError("control differs")
    recipes = [[c, w, kw] for c in (1, 2, 4, 8)
               for w in ((1, 2, 4, 8) if a.k_warps else (2, 4, 8))
               for kw in ((1, 2, 4, 8) if a.k_warps else (1,)) if w * kw <= 32]
    entry = "q4_ref_fp32_run_v2" if a.k_warps else "q4_ref_fp32_run"
    text = '''#include "gemv_ref_fp32.cuh"
#include "bload_contract.hpp"
extern "C" int ENTRY(int c,int w,KWARG int n,int k,void const* a,void const* raw,void* out,void* stream) {
    if(!q4_bload::shape(n,k) || !a || !raw || !out || (uintptr_t(a)&15) || (uintptr_t(raw)&15) || (uintptr_t(out)&3)) return -1;
'''
    text = text.replace("ENTRY", entry).replace("KWARG", "int kw," if a.k_warps else "")
    for c, w, kw in recipes:
        test_kw = f" && kw=={kw}" if a.k_warps else ""
        text += f'''    if(c=={c} && w=={w}{test_kw}) {{
        q4k_gemv_fp32::launch_q4k_gemv<{c},{w},{kw}>(static_cast<half const*>(a),
            static_cast<q4k_gemv_fp32::block_q4_K const*>(raw),static_cast<float*>(out),1,n,k,static_cast<cudaStream_t>(stream));
        return int(cudaGetLastError());
    }}
'''
    text += "    return -1;\n}\n"
    src = a.output / "reference.cu"; src.write_text(text)
    cmd = next(row["argv"] for row in manifest["commands"] if row["label"] == "reference")
    compile_cmd = [*cmd[:cmd.index("-c")], "-c", str(src), "-o", str(a.output / "reference.o")]
    link_cmd = [cmd[0], "-shared", "--cudart=shared", "-Xlinker=-Bsymbolic", str(a.output / "reference.o"),
                str(a.control / "xplane.o"), "-o", str(a.output / "libreference.so")]
    for i, command in enumerate((compile_cmd, link_cmd)):
        with (a.output / f"reference-{i}.log").open("w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    for name in (("libkpack.so", "dispatch.o") if a.k_warps else ("profile", "libkpack.so", "dispatch.o")):
        shutil.copy2(a.control / name, a.output / name)
    if a.k_warps:
        from dev.gemv_cuda.build import replace_once
        profile = (a.control / "profile.cu").read_text()
        profile = replace_once(profile, "using Rawrun=int(*)(int,int,int,int,", "using Rawrun=int(*)(int,int,int,int,int,")
        profile = replace_once(profile, '"q4_ref_fp32_run"', '"q4_ref_fp32_run_v2"')
        profile = replace_once(profile, "rawrun(cfg.columns,cfg.warps,h.n,h.k,", "rawrun(cfg.columns,cfg.warps,cfg.split,h.n,h.k,")
        (a.output / "profile.cu").write_text(profile)
        cmd = next(row["argv"] for row in manifest["commands"] if row["label"] == "runner")
        profile_cmd = [str(a.output / "profile.cu") if x == str(a.control / "profile.cu")
                       else str(a.output / "profile") if x == str(a.control / "profile") else x for x in cmd]
        with (a.output / "profile-build.log").open("w") as f:
            subprocess.run(profile_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    report = dict(manifest, parent_manifest_sha256=digest(a.control / "manifest.json"),
                  reference_recipes=recipes, reference_wrapper_sha256=digest(src),
                  reference_extra_commands=[compile_cmd, link_cmd])
    report["payloads"] = {name:digest(a.output / name) for name in ("profile", "libkpack.so", "libreference.so")}
    if a.k_warps:
        report.update(reference_config_fields=["columns_per_warp", "n_warps", "intra_cta_k_warps"],
                      profile_source_sha256=digest(a.output / "profile.cu"), profile_command=profile_cmd)
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Q4_H800_REFERENCE COMPILED recipes={len(recipes)} control={a.output}", flush=True)


if __name__ == "__main__": main()
