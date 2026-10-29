"""Reuse unchanged device artifacts after a Python-only chain fixture repair."""
import json
from pathlib import Path

from dev.moe_grouped.build import Compiler, ROOT, verify
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


# These files never enter the helper or TC compiler. Changes to a kernel,
# generator, type contract, SDK or compiled payload still require rebuilding.
HARNESS_FILES = (
    'dev/moe_scaffold/bench.py',
    'dev/moe_grouped/bench.py',
    'tools/run_moe_grouped.py',
)


def reuse(sdk, previous, output):
    previous = Path(previous).resolve(strict=True)
    path = previous/'receipt.json'
    original = json.loads(path.read_text())
    # Identity construction hashes the actual compiler/header closure; it does
    # not compile or execute kernels. build.py (including TC generation) stays
    # unchanged, so its old cache identities remain valid too.
    if Compiler(sdk, previous/'cache').identity != original['compiler']:
        raise ValueError('compiled grouped source/SDK differs; use a fresh build')
    receipt = json.loads(json.dumps(original))
    changes = {}
    for name in HARNESS_FILES:
        current = sha(ROOT/name)
        for sources in (receipt['sources'], receipt['helper']['sources']):
            if name in sources and sources[name] != current:
                changes[name] = dict(before=sources[name], after=current)
                sources[name] = current
    receipt['sources']['dev/moe_grouped/reuse.py'] = sha(ROOT/'dev/moe_grouped/reuse.py')
    verify(receipt)  # All other sources, runtime DSOs and device binaries.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'reused-receipt.json', original)
    receipt['reused_build'] = dict(receipt=str(path), sha256=sha(path),
        harness_changes=changes, compiled_artifacts_unchanged=True)
    atomic_json(output/'receipt.json', receipt)
    print('MOE_GROUPED_BUILD_REUSED '+json.dumps(receipt['reused_build']), flush=True)
    return receipt
