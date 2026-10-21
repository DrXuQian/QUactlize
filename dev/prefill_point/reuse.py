"""Read-only validation of the previous point's unchanged device payloads."""
import json

from dev.strategy.resources import compiler
from dev.strategy.resource_pilot import current_module
from quactlize.runtime.compiler import sha


def tc_record(sdk, payload, spec):
    c = compiler(spec, sdk, payload/'tc')
    matches = []
    for path in (payload/'tc').glob('*/manifest.json'):
        row = json.loads(path.read_text())
        if row['parent'] != spec['parent']: continue
        row['path'] = str(path.parent/'kernel.so')
        current_module(row, spec, c)
        if sha(row['path']) != row['sha256']:
            raise ValueError('previous TC binary differs')
        matches.append(row | {'cache_hit': True})
    if len(matches) != 1:
        raise ValueError('one exact previous TC module required; no implicit JIT')
    return matches[0]
