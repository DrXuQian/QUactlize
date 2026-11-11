"""Bounded PPU1.5 public-package correctness, not a timing sweep."""
import json

import pytest

from .support import QTYPES
from .test_public_library import run_dense
from .test_moe_library import test_public_indexed_projection as indexed_projection
from .test_ffn_library import run_ffn

MS = (*range(1, 10), 16, 128)


@pytest.fixture(scope='module', autouse=True)
def native_package(package):
    assert json.loads((package/'manifest.json').read_text())['arch'] == 'ppu_15'


@pytest.mark.parametrize('q', QTYPES)
@pytest.mark.parametrize('m', MS)
@pytest.mark.parametrize('compute', [0, 1], ids=['f16', 'bf16'])
def test_dense_auto(package, sdk, q, m, compute, record_property):
    def record(key, value):
        record_property(key, value)
        print('ZW890_PUBLIC_PLAN '+json.dumps(dict(q=q, m=m, compute=compute, **{key:value})), flush=True)
    # Same independent GGUF oracle, changing addresses, two streams, replay,
    # padded strides, output poisoning and omitted-producer negative as CI.
    run_dense(package, sdk, q, m, compute, record)


@pytest.mark.parametrize('q', QTYPES)
@pytest.mark.parametrize('channels', [1, 2])
def test_moe_auto(package, sdk, q, channels):
    # Covers changing IDs, M1..8/9/16/128/5120, BF16 range and graph replay.
    indexed_projection(package, sdk, q, channels)


@pytest.mark.parametrize('q', QTYPES)
@pytest.mark.parametrize('kind', [0, 1], ids=['dense', 'moe'])
@pytest.mark.parametrize('merged', [False, True], ids=['separate', 'canonical-2n'])
def test_ffn_auto(package, sdk, q, kind, merged):
    run_ffn(package, sdk, kind, merged, q, ms=(1, 8, 9, 128))


@pytest.mark.parametrize('q', [10, 11, 12, 13, 14])
def test_full_provider(package, sdk, q, record_property):
    # cuBLAS remains ordinary submission, not a forced graph-capture test.
    run_dense(package, sdk, q, 128, 1, record_property, full=True)


def test_q8_full_f16_auto(package, sdk, record_property):
    from quactlize.dispatch.dense_prefill import EVIDENCE
    table = json.loads(EVIDENCE.read_text())
    point = next(p for p in table['points'] if p['shape'] == [8, 128, 256, 3072])
    recipe = table['recipes'][point['winner']]
    assert recipe['route'] == 2
    # Also cover the typed F16 Full provider; the legacy forced path above
    # uses BF16 and deliberately does not accept Q8.
    run_dense(package, sdk, 8, 128, 0, record_property, n=256, k=3072, expected=recipe)
