"""Read-only PPU-SMI clock observations; never change device clock settings."""
import csv
import io
import math
import re
import subprocess
import time


def pci_key(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',
                         value.strip())
    if not match:
        raise ValueError('invalid PCI identity: '+value)
    return tuple(int(x, 16) for x in match.groups())


def parse(text, pci, expected_mhz, memory_mhz=None):
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != 3 or pci_key(rows[0][0]) != pci_key(pci):
        raise ValueError('PPU-SMI clock query must return exactly the assigned physical card')
    try:
        core, memory = map(float, rows[0][1:])
    except ValueError:
        raise ValueError('PPU-SMI did not report numeric current CU/memory clocks') from None
    if (not all(math.isfinite(v) and v > 0 for v in (core, memory, expected_mhz)) or
            abs(core/expected_mhz-1) > .001 or
            memory_mhz is not None and abs(memory/memory_mhz-1) > .001):
        raise ValueError(f'current clocks differ: CU={core}MHz memory={memory}MHz; '
                         f'expected CU={expected_mhz}MHz memory={memory_mhz}; no timings admitted')
    return dict(pci=pci, cu_mhz=core, memory_mhz=memory,
                scope='SMI_OBSERVATION_OUTSIDE_TIMED_EVENTS_NOT_CONTINUOUS_CLOCK_LOCK_PROOF')


def observe(binary, pci, expected_mhz, memory_mhz=None):
    command = [str(binary), '--query-ppu=pci.bus_id,clocks.current.cu,clocks.current.memory',
               '--format=csv,noheader,nounits', '--id='+pci]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError('read-only PPU-SMI clock query failed: '+result.stdout+result.stderr)
    return parse(result.stdout, pci, expected_mhz, memory_mhz) | dict(
        command=command, observed_ns=time.time_ns(), stdout=result.stdout)
