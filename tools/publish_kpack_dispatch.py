#!/usr/bin/env python3
"""Copy verified native runtime payloads into a new, focused LFS package."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_kpack_dispatch import verify, prefill_paths
from quactlize.runtime.compiler import sha


def attach_model(dst, build, source, pack):
    build, source, pack = (p.resolve(strict=True) for p in (build, source, pack))
    subprocess.run(['git', '-C', str(source), 'diff', '--quiet'], check=True)
    subprocess.run(['git', '-C', str(source), 'diff', '--cached', '--quiet'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    cache = (build / 'CMakeCache.txt').read_text()
    for option in ('GGML_USE_PPU', 'GGML_NCP_QUACTLIZE'):
        if f'{option}:BOOL=ON\n' not in cache:
            raise ValueError('caller build option differs: ' + option)
    if 'CMAKE_HOME_DIRECTORY:INTERNAL=' + str(source) + '\n' not in cache:
        raise ValueError('caller source/build directories differ')
    targets = ('llama-server', 'llama-batched-bench', 'llama-perplexity')
    paths = [build / 'bin' / name for name in targets]
    paths += sorted(p for p in (build/'bin').glob('lib*.so*') if not p.name.startswith('libquactlize'))
    result = dict(schema='quactlize.q4-model-deployment.v1', llama_source_commit=commit,
        llama_branch='dev/quactlize-v0.3.0', architecture='ppu0010', device_admission='PENDING', files={}, links={})
    for path in paths:
        if path.resolve(strict=True).parent != build/'bin':
            raise ValueError('caller payload escapes build/bin')
        target = dst / 'llama/bin' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            link = str(path.readlink())
            if '/' in link:
                raise ValueError('caller symlink is not a relative soname')
            target.symlink_to(link)
            result['links'][str(target.relative_to(dst))] = link
        else:
            shutil.copy2(path, target)
            result['files'][str(target.relative_to(dst))] = sha(target)
    (dst / 'pack').mkdir()
    for path in (pack, pack.parent/'manifest.json'):
        target = dst / 'pack' / path.name
        shutil.copy2(path, target)
        result['files'][str(target.relative_to(dst))] = sha(target)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--llama-build", type=Path)
    p.add_argument("--llama-source", type=Path)
    p.add_argument("--pack-library", type=Path)
    a = p.parse_args()
    if any((a.llama_build, a.llama_source, a.pack_library)) and not all((a.llama_build, a.llama_source, a.pack_library)):
        p.error('model deployment requires llama-build, llama-source and pack-library')
    src = a.build.resolve(strict=True)
    dst = a.output.resolve()
    m = verify(src)
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
    if 'prefill' in m:
        paths += prefill_paths(src, m['prefill'])
    for item in (m.get("decode_io_gate", {}).get("simt_binaries", [])+
                 m.get("moe_mixed_gate", {}).get("simt_binaries", [])):
        source = (src / item["path"]).resolve(strict=True)
        if source.parent != src or sha(source) != item["sha256"]:
            raise ValueError("SIMT proof payload identity differs")
        paths.append(item["path"])
    for name in paths:
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
    if a.llama_build:
        m['model'] = attach_model(dst, a.llama_build, a.llama_source, a.pack_library)
        (dst/'manifest.json').write_text(json.dumps(m, indent=2)+'\n')
    verify(dst)
    print(
        f"KPACK_NATIVE_PUBLISHED modules={len(m['modules'])} output={dst} device_validation=PENDING"
    )


if __name__ == "__main__":
    main()
