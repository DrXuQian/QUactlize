"""Explicit build target; installed JIT inherits the package, never a GPU guess."""
import json
from pathlib import Path

ARCHES = ('ppu_10', 'ppu_15')


def selected_arch(arch=None):
    if arch is None:
        config = Path(__file__).with_suffix('.json')
        arch = json.loads(config.read_text())['arch'] if config.is_file() else 'ppu_10'
    if arch not in ARCHES:
        raise ValueError('unsupported PPU target: ' + str(arch))
    return arch


def definitions(arch):
    return ['-DQUACTLIZE_PPU_ARCH=150'] if selected_arch(arch) == 'ppu_15' else []


def flags_for_arch(flags, arch=None):
    arch = selected_arch(arch)
    # Preserve the established PPU1.0 command exactly.
    return [('-arch='+arch if flag.startswith('-arch=') else flag) for flag in flags] + definitions(arch)
