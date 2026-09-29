#!/usr/bin/env python3
"""Build matched production/once-only preparation in a fresh CUDA directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def warp_router():
    path = ROOT / "quactlize/integrations/llama/router.cuh"
    text = path.read_text()
    start = text.index("template<bool HasBias>")
    body = text[start:text.index("\n} // namespace", start)]
    for before, after in (
        ("void router_256_top8(", "void router_256_top8_warp("),
        ("  if (threadIdx.x>=32) return;\n", ""),
        ("int lane=int(threadIdx.x);", "int lane=int(threadIdx.x)%32;"),
    ):
        if body.count(before) != 1:
            raise ValueError("production router seam changed: " + before)
        body = body.replace(before, after)
    return "namespace quactlize::llama {\n" + body + "\n}\n"


def baseline_headers(ref, output):
    """Keep the pre-patch production bodies; rename symbols, not arithmetic."""
    commit=subprocess.check_output(['git','rev-parse',ref+'^{commit}'],cwd=ROOT,text=True).strip()
    names=('quactlize/execution/moe_router_warp.cuh','quactlize/execution/moe_prepare.cuh')
    bodies={n:subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT).decode() for n in names}
    router=bodies[names[0]].replace('router_256_top8_warp','router_256_top8_warp_incumbent')
    router=router.replace('router_from_key','router_from_key_incumbent')
    router=router.replace('../integrations/llama/router.cuh',str(ROOT/'quactlize/integrations/llama/router.cuh'))
    (output/'router-incumbent.cuh').write_text(router)
    prepare=bodies[names[1]].replace('prepare_detail','prepare_incumbent')
    prepare=prepare.replace('router_256_top8_warp','router_256_top8_warp_incumbent')
    prepare=prepare.replace('"moe_router_warp.cuh"','"router-incumbent.cuh"')
    prepare=prepare.replace('"../runtime/moe_chain.cuh"','"'+str(ROOT/'quactlize/runtime/moe_chain.cuh')+'"')
    (output/'prepare-incumbent.cuh').write_text(prepare)
    return dict(commit=commit,source_hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bodies.items()},
                transformation='SYMBOL_NAMESPACE_AND_INCLUDE_PATHS_ONLY')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    p.add_argument("--platform", choices=("cuda","ppu"), default="cuda")
    p.add_argument("--arch", default="sm_120", choices=("sm_120", "sm_90"))
    p.add_argument("--baseline-ref", help="compare against this immutable admitted prepare source")
    p.add_argument("--production-candidate", action="store_true", help="test the actual current prepare dispatcher")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = (ROOT / "dev/moe_prepare/bench.cu").read_text()
    marker = '// INSERT_WARP_ROUTER\n'
    if source.count(marker) != 1:
        raise ValueError("benchmark router seam changed")
    generated = a.output / "bench.cu"
    baseline=baseline_headers(a.baseline_ref,a.output) if a.baseline_ref else None
    source=source.replace(marker, '#define QK_PREPARE_BASELINE 1\n#include "prepare-incumbent.cuh"\n' if baseline else '')
    if a.production_candidate:
        if not baseline:raise ValueError('production comparison requires a frozen baseline')
        source='#define QK_PREPARE_PRODUCTION 1\n'+source
    if a.platform=="ppu":source=re.sub(r"\bcuda(?=[A-Z_])","hggc",source)
    generated.write_text(source)
    paths = [ROOT / n for n in (
        "dev/moe_prepare/bench.cu", "dev/moe_prepare/fast.cuh", "dev/moe_prepare/build.py",
        "quactlize/execution/moe_prepare.cuh",
        "quactlize/execution/moe_router_warp.cuh",
        "quactlize/runtime/moe_chain.cuh", "quactlize/runtime/moe_protocol.h",
        "quactlize/runtime/indexed.cuh", "quactlize/integrations/llama/router.cuh",
        "quactlize/integrations/llama/indexed.h",
    )]
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    hashes = {str(f.relative_to(ROOT)): sha(f) for f in paths}
    includes = [ROOT, ROOT / "quactlize/include",
                ROOT / "third_party/actlize/include", ROOT / "third_party/actlize/tools/util/include"]
    # Bind transitive ABI/CuTe and the exact CUDA-only compatibility boundary.
    paths += [p for directory in includes[1:] for p in directory.rglob('*')
              if p.is_file() and p.suffix in ('.h','.hpp','.cuh','.inc')]
    if a.platform=='cuda':
        paths += [p for p in (ROOT/'dev/gemv_cuda/compat').rglob('*') if p.is_file()]
    hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(paths))}
    env=dict(os.environ)
    if a.platform=="cuda":
      includes.insert(0,ROOT/"dev/gemv_cuda/compat")
      command = [str(a.cuda / "bin/nvcc"), "-std=c++17", "-O3", "-lineinfo", "-arch=" + a.arch,
               "--expt-relaxed-constexpr", "-DCUTLASS_USE_PACKED_TUPLE=1", "-DCUTE_USE_PACKED_TUPLE=1",
               "-include", str(ROOT / "dev/gemv_cuda/compat/compiler_bridge.h"), "-Xptxas=-v",
               *["-I" + str(f) for f in includes], str(generated), "-o", str(a.output / "bench")]
      commands=[command]
    else:
      from quactlize.runtime.compiler import FLAGS,LIBRARIES
      env['PATH']=str(a.cuda/'bin')+os.pathsep+env.get('PATH','')
      env['LD_LIBRARY_PATH']=str(a.cuda/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
      commands=[[str(a.cuda/'bin/hgcc'),*FLAGS,*["-I"+str(f) for f in includes],'-c',str(generated),'-o',str(a.output/'bench.o')],
                ['g++',str(a.output/'bench.o'),'-Wl,--allow-shlib-undefined',f'-L{a.cuda}/lib',*[f'-l{x}' for x in LIBRARIES],'-o',str(a.output/'bench')]]
    with (a.output / "build.log").open("w") as log:
        for command in commands:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if any(sha(ROOT / n) != value for n, value in hashes.items()):
        raise ValueError("source changed while compiling")
    receipt = dict(source_hashes=hashes, generated_sha256=sha(generated),
                   binary_sha256=sha(a.output / "bench"), commands=commands,platform=a.platform,
                   scope="PREPARE_ONLY_NOT_GEMM", device_validated=False,
                   production_candidate=a.production_candidate)
    if baseline: receipt['prepare_baseline']=baseline
    (a.output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("MOE_PREPARE_BUILD PASS", a.output / "bench", flush=True)


if __name__ == "__main__":
    main()
