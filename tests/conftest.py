import os

import pytest


def pytest_addoption(parser):
    parser.addoption('--device', action='store_true', help='require the PPU runtime tests')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--device'):
        for name in ('QUACTLIZE_SDK','QUACTLIZE_BUNDLE','QUACTLIZE_JIT_CACHE'):
            if not os.environ.get(name):
                raise pytest.UsageError('--device requires '+name)
        return
    for item in items:
        if 'device' in item.keywords:
            item.add_marker(pytest.mark.skip(reason='PPU checks require explicit --device'))
