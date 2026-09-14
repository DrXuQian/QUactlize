#!/usr/bin/env python3
"""Validate a native dispatch package before allowing model execution."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler, sha, source_contract
from quactlize.decode.compiler import DecodeCompiler
from quactlize.runtime.tuning import digest


def prefill_paths(root, receipt, *, sdk=None):
    """Verify the optional composition closure, never load a device runtime."""
    root = Path(root).resolve(strict=True)
    names = {'library': 'libquactlize_ppu_prefill.so',
             'receipt': 'prefill-runtime.json', 'helper': 'kpack_deepgemm_prewarm.py'}
    if not isinstance(receipt, dict):
        raise ValueError('prefill package receipt is missing')
    for field, name in names.items():
        path = root / name
        if (receipt.get(field) != name or path.is_symlink() or
                not path.is_file() or sha(path) != receipt.get(field + '_sha256')):
            raise ValueError('prefill payload differs: ' + name)
    build = json.loads((root / names['receipt']).read_text())
    if (build.get('schema') != 'quactlize.prefill-runtime.v1' or
            build.get('library') != names['library'] or
            build.get('sha256') != receipt['library_sha256']):
        raise ValueError('prefill build/library identity differs')
    if sdk is not None:
        for name, expected in build['runtime'].items():
            if Path(name).name != name or sha(Path(sdk) / 'lib' / name) != expected:
                raise ValueError('prefill SDK runtime differs: ' + name)
    return list(names.values())


def model_paths(root, model):
    """Verify the optional prebuilt llama caller and paired producer."""
    root = Path(root).resolve(strict=True)
    if model.get('schema') != 'quactlize.q4-model-deployment.v1':
        raise ValueError('model deployment schema differs')
    commit = model.get('llama_source_commit', '')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('model caller source commit differs')
    files, links = model.get('files', {}), model.get('links', {})
    required = {f'llama/bin/{name}' for name in ('llama-server', 'llama-batched-bench', 'llama-perplexity')}
    required |= {'pack/libquactlize_ppu_pack.so', 'pack/manifest.json'}
    if not required <= files.keys() or files.keys() & links.keys():
        raise ValueError('model deployment payload set differs')
    for name, expected in files.items():
        path = root / name
        if (not (name.startswith('llama/bin/') or name.startswith('pack/')) or
                path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or sha(path) != expected):
            raise ValueError('model payload differs: ' + name)
    for name, target in links.items():
        path = root / name
        if (not name.startswith('llama/bin/') or '/' in target or not path.is_symlink() or
                str(path.readlink()) != target or not path.resolve(strict=True).is_relative_to(root/'llama/bin')):
            raise ValueError('model library link differs: ' + name)
    return list(files) + list(links)


def verify(root, *, sdk=None):
    root = Path(root).resolve(strict=True)
    m = json.loads((root / "manifest.json").read_text())
    if (m.get("schema") != "quactlize.kpack-native-dispatch.v1" or
        not isinstance(m.get("modules"), list) or
        (not m["modules"] and m.get("jit_required") is not True)):
        raise ValueError("native package schema or module set differs")
    for name, field in (
        ("libquactlize_kpack_dispatch.so", "dispatch_sha256"),
        ("libquactlize_ppu_execution.so", "execution_sha256"),
    ):
        if sha(root / name) != m[field]:
            raise ValueError("native payload differs: " + name)
    if "decode_policy" in m:
        receipt = m["decode_policy"]
        path = (root / receipt["path"]).resolve(strict=True)
        if (path.parent != root or sha(path) != receipt["sha256"] or
            m["execution_receipt"].get("q4_decode_policy_sha256") != receipt["sha256"]):
            raise ValueError("decode policy/execution identity differs")
    if 'prefill' in m:
        prefill_paths(root, m['prefill'], sdk=sdk)
    elif (root / 'libquactlize_ppu_prefill.so').exists():
        raise ValueError('unmanifested prefill runtime would arm the model loader')
    if 'moe_mixed_gate' in m:
        gate=m['moe_mixed_gate']
        if (gate.get('schema')!='quactlize.moe-mixed-gate.v1' or gate.get('stage_cases')!=80 or
                gate.get('chain_cases')!=24 or len(gate.get('simt_binaries',[]))!=1):
            raise ValueError('mixed MoE gate denominator differs')
        item=gate['simt_binaries'][0]
        if item.get('path')!='mixed-stages' or (root/'mixed-stages').is_symlink() or sha(root/'mixed-stages')!=item.get('sha256'):
            raise ValueError('mixed MoE stage binary differs')
    if 'model' in m:
        model_paths(root, m['model'])
    if m.get("jit_required") or "jit_source_contract" in m:
        if source_contract(m.get("jit_source_identity", {})) != m.get("jit_source_contract"):
            raise ValueError("JIT source contract differs")
        if sdk is not None:
            # Constructor hashes inputs only: no cache directory, compilation,
            # runtime loading or GPU work. Reject stale bundles before a sweep.
            identity = Compiler(sdk, root / ".source-check-unused").identity
            if source_contract(identity) != m["jit_source_contract"]:
                changed = [key for key in ("kernel", "flags", "generator")
                           if identity[key] != m["jit_source_identity"][key]]
                raise ValueError("JIT checkout differs from dispatcher (" + ",".join(changed)
                                 + "); rebuild the small dispatcher")
    seen = set()
    for r in m["modules"]:
        if (
            r["key"] in seen
            or len(r["key"]) != 64
            or any(c not in "0123456789abcdef" for c in r["key"])
        ):
            raise ValueError("duplicate/invalid module key")
        seen.add(r["key"])
        expected = "modules/" + r["key"] + "/kernel.so"
        path = (root / expected).resolve(strict=True)
        if (
            r["path"] != expected
            or not path.is_relative_to(root)
            or sha(path) != r["sha256"]
        ):
            raise ValueError("module payload differs: " + r["key"])
        compiler = DecodeCompiler if r["identity"].get("endpoints") == "decode-m1-8-f32-bf16-v1" else Compiler
        source = compiler.source(None, r["parent"], "")
        if (
            digest(dict(identity=r["identity"], parent=r["parent"], source=source))
            != r["key"]
        ):
            raise ValueError("module compiler identity differs")
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--sdk", type=Path, help="also check the live JIT source contract, without compiling")
    a = p.parse_args()
    manifest = verify(a.bundle, sdk=a.sdk)
    print(
        f"KPACK_NATIVE_PACKAGE VERIFIED modules={len(manifest['modules'])} "
        f"execution_sha256={manifest['execution_sha256']} device_admission=PENDING"
    )
