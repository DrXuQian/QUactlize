#!/usr/bin/env python3
"""Print foreign compute processes on a PPU (from ppu-smi); exit 0 only when the device is idle."""
import os
import re
import subprocess
import sys
import time

dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
wait = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
deadline = time.time() + wait
while True:
    text = subprocess.run(['ppu-smi'], capture_output=True, text=True).stdout
    procs = [m for m in re.finditer(r'^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+\S+\s+(.+?)\s+(\d+)MiB\s*\|', text, re.M)]
    busy = [(int(m[2]), m[3].strip()) for m in procs if int(m[1]) == dev and int(m[2]) != os.getpid()]
    if not busy:
        print(f'device {dev} idle')
        sys.exit(0)
    if time.time() >= deadline:
        print(f'device {dev} busy: {busy}')
        sys.exit(1)
    time.sleep(20)
