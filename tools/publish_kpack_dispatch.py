#!/usr/bin/env python3
"""Copy verified native runtime payloads into a new, focused LFS package."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_kpack_dispatch import verify, prefill_paths, pack_paths, router_alias_paths, paired_paths
from quactlize.runtime.compiler import sha


def attach_pack(dst, pack):
    pack = pack.resolve(strict=True)
    if pack.name != 'libquactlize_ppu_pack.so':
        raise ValueError('only the Quactlize packer may be attached')
    result = dict(schema='quactlize.kpack-producer-package.v1', files={})
    (dst / 'pack').mkdir()
    for path in (pack, pack.parent/'manifest.json'):
        target = dst / 'pack' / path.name
        shutil.copy2(path, target)
        result['files'][str(target.relative_to(dst))] = sha(target)
    pack_paths(dst, result)
    return result


def publish(build, output, pack=None):
    src = build.resolve(strict=True)
    dst = output.resolve()
    m = verify(src)
    if pack is None and ('model' in m or 'pack' in m):
        pack = src / 'pack/libquactlize_ppu_pack.so'
    # Historical packages may contain a caller. Never carry it into a new publication.
    m.pop('model', None)
    m.pop('pack', None)
    if dst.exists():
        raise ValueError("publication output already exists")
    if not dst.is_relative_to(ROOT / "prebuilt/ppu0010"):
        raise ValueError("publication must be under prebuilt/ppu0010")
    paths = [
        "manifest.json",
        "libquactlize_kpack_dispatch.so",
        "libquactlize_ppu_execution.so",
    ]
    paths += [r["path"] for r in m["modules"]]
    if "decode_policy" in m:
        paths.append(m["decode_policy"]["path"])
    if "smallm_policy" in m:
        paths.append(m["smallm_policy"]["path"])
    if "smallm_matched_policy" in m:
        paths.append(m["smallm_matched_policy"]["path"])
    if "q8_vector_policy" in m:
        paths.append(m["q8_vector_policy"]["path"])
    for field in ('decode_winners','final_selection'):
        if field in m:paths.append(m[field]['path'])
    if "bf16_gate" in m:
        from dev.bf16_compute.run import validate_package
        gate = validate_package(src / 'bf16')
        paths += ['bf16/manifest.json']
        paths += ['bf16/' + name for name in sorted({gate['simt']['path'], gate['moe']['path']} |
                  {r['path'] for r in gate['modules'].values()})]
        paths += ['bf16/' + r['path'] for r in gate.get('matched_modules',{}).values()]
    if "local_optimization_gate" in m:
        from tools.attach_kpack_local_gates import payload_paths
        paths += payload_paths(src,m['local_optimization_gate'])
    if 'router_alias_gate' in m:
        paths += router_alias_paths(src,m['router_alias_gate'])
    if 'paired_gate_up' in m:
        paths += paired_paths(src,m['paired_gate_up'])
    if "q4_bf16_gate" in m:
        paths += [m['q4_bf16_gate']['path'],m['q4_bf16_gate']['library']]
    if 'prefill' in m:
        paths += prefill_paths(src, m['prefill'])
    for item in (m.get("decode_io_gate", {}).get("simt_binaries", [])+
                 m.get("moe_mixed_gate", {}).get("simt_binaries", [])):
        source = (src / item["path"]).resolve(strict=True)
        if source.parent != src or sha(source) != item["sha256"]:
            raise ValueError("SIMT proof payload identity differs")
        paths.append(item["path"])
    for name in paths:
        if ('llama' in Path(name).parts or
                Path(name).name.startswith(('llama-', 'libllama', 'libggml', 'libmtmd', 'libncp'))):
            raise ValueError('caller binary is outside the Quactlize package: ' + name)
        source = (src / name).resolve(strict=True)
        if not source.is_relative_to(src) or not source.is_file() or (src/name).is_symlink():
            raise ValueError("publication source is not a regular internal payload")
    dst.mkdir(parents=True)
    for name in paths:
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, target)
        if sha(src / name) != sha(target):
            raise ValueError("published payload copy differs")
    if m.get("jit_required"):
        # A JIT-only package has no compiled closure. Model prewarm uses the
        # caller's actual requests, not the development coverage census.
        (dst / "plan.json").write_text(json.dumps(dict(parents=[], requests=[],
            scope="ON_DEMAND_JIT_USE_MODEL_PREWARM_PLAN"), indent=2)+"\n")
    else:
        shutil.copy2(src / "plan.json", dst / "plan.json")
    if pack is not None:
        m['pack'] = attach_pack(dst, pack)
    (dst/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
    verify(dst)
    print(
        f"KPACK_NATIVE_PUBLISHED modules={len(m['modules'])} output={dst} device_validation=PENDING"
    )
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pack-library', type=Path)
    a = p.parse_args()
    publish(a.build, a.output, a.pack_library)


if __name__ == "__main__":
    main()
