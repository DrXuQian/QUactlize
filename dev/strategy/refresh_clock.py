"""Explicit PPU clock request and separate, untimed actual-clock diagnostics."""
import csv
import io
import math
import re
import subprocess
import time

ADVISORY = 'LOCK_REQUESTED_ACTUAL_CLOCKS_ADVISORY'


def pci_key(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',
                         value.strip())
    if not match:
        raise ValueError('invalid PCI identity: '+value)
    return tuple(int(x, 16) for x in match.groups())


def parse(text, pci, expected_mhz, memory_mhz=None, *, advisory=False):
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != 3 or pci_key(rows[0][0]) != pci_key(pci):
        raise ValueError('PPU-SMI clock query must return exactly the assigned physical card')
    try:
        core, memory = map(float, rows[0][1:])
    except ValueError:
        raise ValueError('PPU-SMI did not report numeric current CU/memory clocks') from None
    if not all(math.isfinite(v) and v > 0 for v in (core, memory, expected_mhz)):
        raise ValueError('PPU-SMI reported invalid clock values')
    differs = (abs(core/expected_mhz-1) > .001 or
               memory_mhz is not None and abs(memory/memory_mhz-1) > .001)
    if differs and not advisory:
        raise ValueError(f'current clocks differ: CU={core}MHz memory={memory}MHz; '
                         f'expected CU={expected_mhz}MHz memory={memory_mhz}; no timings admitted')
    result = dict(pci=pci, cu_mhz=core, memory_mhz=memory,
                  scope='SMI_OBSERVATION_OUTSIDE_TIMED_EVENTS_NOT_CONTINUOUS_CLOCK_LOCK_PROOF')
    if advisory:
        result.update(clock_policy=ADVISORY, target_differs=differs,
                      affects_verdict=False, frequency_normalization=False)
    return result


def observe(binary, pci, expected_mhz, memory_mhz=None, *, advisory=False):
    command = [str(binary), '--query-ppu=pci.bus_id,clocks.current.cu,clocks.current.memory',
               '--format=csv,noheader,nounits', '--id='+pci]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError('read-only PPU-SMI clock query failed: '+result.stdout+result.stderr)
    return parse(result.stdout, pci, expected_mhz, memory_mhz, advisory=advisory) | dict(
        command=command, observed_ns=time.time_ns(), stdout=result.stdout)


def request_lock(binary, pci, expected_mhz):
    """Only the assigned physical device's CU clock; no memory/power/reset writes.

    SDK2.1.1 ppu-smi v1.28 documents --lock-ppu-clocks, NOT nvidia-smi's
    --lock-gpu-clocks. A successful request does not disable hardware throttling.
    Return failed commands too so the caller can persist their diagnostics.
    """
    pci_key(pci)
    if (not math.isfinite(expected_mhz) or not 0 < expected_mhz < 10000 or
            int(expected_mhz) != expected_mhz):
        raise ValueError('clock request must be a positive integer MHz value')
    mhz = int(expected_mhz)
    command = [str(binary), '--id='+pci, f'--lock-ppu-clocks={mhz},{mhz}',
               '--error-on-warning']
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return dict(pci=pci, requested_mhz=mhz, command=command,
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                observed_ns=time.time_ns())
