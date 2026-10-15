"""One reviewed migration of the stopped 4ad5bb8 large-M campaign.

Keep the original TC, dequant and adapter images AND isolated-kernel timers.
Only dense cuBLAS is remeasured with the ordinary native event frontend proved
at 4e46a0f. This is not a general source-hash override or a timing cache.
"""
from pathlib import Path

from dev.strategy.prefill_retry import ORCHESTRATION, read
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


# Exact changed-file/hash pairs of 4ad5bb8 -> 4e46a0f, excluding orchestration.
# The intervening small-M selector changes do not alter ANY of the 430 large-M
# points, candidates or incumbent recipes (checked again for each import).
# prefill_measure/isolated/cold_decode/fixture/oracles and TC codegen are not
# exceptions: ANY change to them rejects reuse. Full's grouped call only gained
# a Python matmul method; its captured device work and default workspace stay
# unchanged. Retain the OLD helper DSO, including its capture bridge.
REVIEWED_DELTAS = {
    'source': 'e2d15a6e2b12794c537f225ddd04e9b135e2b3a0ff03c4ec4d51b851b2f93d02',
    'harness': 'df9bcecb882acd032448890b533a818342c44db7346715a29285f93a40e5a2e0',
}
MIGRATION_ONLY = ORCHESTRATION | {
    'dev/strategy/prefill_upgrade.py',
    'dev/strategy/cublas_components.py',  # dense cuBLAS evidence is NEVER imported
    'tools/run_kpack_strategy_prefill_multigpu.py',
}
OLD_HELPER_SOURCE = 'e160779c8e12259dd1476676f9375a14b6b2e5796baaef3bcc064bf86b9ef9f7'
OLD_PROTOCOL = '3x3_ONE_KERNEL_PER_TIMER_NO_LAUNCH_PENALTY'
NEW_PROTOCOL = 'CUBLAS_PROVIDER_EVENTS_OTHERS_SEPARATE_KERNEL_EVENTS'


def source_delta(previous, current, kind):
    for value in (previous, current):
        if value.get('sha256') != digest(value['files']):
            raise ValueError('provider upgrade source closure digest differs')
    old, new = previous['files'], current['files']
    changed = {p: [old.get(p), new.get(p)] for p in old.keys() | new.keys()
               if old.get(p) != new.get(p) and p not in MIGRATION_ONLY}
    if digest(changed) != REVIEWED_DELTAS[kind]:
        raise ValueError('provider upgrade is only approved for 4ad5bb8 -> 4e46a0f measurement '
                         'inputs; changed files: '+', '.join(sorted(changed)))
    return dict(sha256=digest(changed), files=sorted(changed))


def verify_helpers(sdk, output):
    """Verify and reuse the original two-TU helper, not a rebuilt lookalike."""
    from dev.strategy.prefill_build import identity
    current = identity(sdk)
    expected = dict(compiler=current['compiler'], adapters=current['adapters'], sources={
        'dev/strategy/prefill_helpers.cu': OLD_HELPER_SOURCE,
        'quactlize/dequant/indexed.cu': current['sources']['quactlize/dequant/indexed.cu'],
    })
    row, _ = read(Path(output).resolve(strict=True), 'manifest.json')
    library = Path(output)/'libstrategy_prefill.so'
    if (library.is_symlink() or row['identity'] != expected or row['sha256'] != sha(library)
            or row.get('record_sha256') != digest({k:v for k,v in row.items() if k!='record_sha256'})):
        raise ValueError('provider upgrade original helper source/SDK/image differs')
    return row


def authority_check(old, new):
    # Only a NEW dense cuBLAS helper is added; every imported device image,
    # provider, physical device, oracle, L2 and resource recipe must match.
    exceptions = {'harness', 'protocol', 'cublas_helpers'}
    if (old.get('protocol') != OLD_PROTOCOL or new.get('protocol') != NEW_PROTOCOL or
            'cublas_helpers' in old or not new.get('cublas_helpers') or
            {k:v for k,v in old.items() if k not in exceptions} !=
            {k:v for k,v in new.items() if k not in exceptions}):
        raise ValueError('provider upgrade changed isolated binary/device/oracle authority')
    delta = source_delta(old['harness'], new['harness'], 'harness')
    return dict(schema='quactlize.prefill-cublas-upgrade.v1', harness_delta=delta,
                old_protocol=OLD_PROTOCOL, new_protocol=NEW_PROTOCOL,
                cublas_helpers=new['cublas_helpers'],
                reused_images='BYTE_IDENTICAL_TC_DEQUANT_AND_ISOLATED_HELPERS',
                remeasure='ALL_DENSE_BF16_PROVIDER_JOBS_EVEN_IF_PREVIOUSLY_PASS')


def replaces(point, job):
    return point['workload']['operation'] == 'dense' and job['roles'] == ['bf16']
