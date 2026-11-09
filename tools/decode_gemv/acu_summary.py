#!/usr/bin/env python3
"""Side-by-side ACU comparison of single-kernel reports (raw page metrics).

    python3 tools/acu_summary.py NAME=REPORT.acurep ... [--md out.md]
Prints duration, DRAM read bytes/throughput, DRAM active cycles, cache hit rates, LLC request
size, occupancy, issue rate and the top warp stall reasons (per issued instruction).
"""
import argparse
import re
import subprocess
from pathlib import Path

ROWS = [
    ('duration_us', 'ppu__time_duration.sum', 1e-3),
    ('dram_read_MB', 'dram__bytes_read.sum', 1 / 2**20),
    ('dram_GBps', 'dram__bytes_read.sum.per_second', 1e-9),
    ('dram_read_pct_peak', 'dram__bytes_read.sum.pct_of_peak_sustained_elapsed', 1),
    ('dram_active_cycles_pct', 'dram__cycles_active.avg.pct_of_peak_sustained_elapsed', 1),
    ('llc_active_cycles_pct', 'llc__cycles_active.avg.pct_of_peak_sustained_elapsed', 1),
    ('kvd_hit_pct', 'kvd__requests_hit_rate.pct', 1),
    ('l2_hit_pct', 'l2__requests_hit_rate.pct', 1),
    ('llc_hit_pct', 'llc__requests_hit_rate.pct', 1),
    ('achieved_occupancy_pct', 'cu__warps_active.avg.pct_of_peak_sustained_active', 1),
    ('ipc_active', 'cu__inst_issued.avg.per_cycle_active', 1),
    ('issue_active_pct', 'pu__issue_active.avg.pct_of_peak_sustained_active', 1),
    ('inst_issued_M', 'pu__inst_issued.sum', 1e-6),
    ('warp_cycles_per_issue', 'pu__average_warps_latency_per_inst_issued.ratio', 1),
    ('regs', 'launch__registers_per_thread', 1),
    ('grid', 'launch__grid_size', 1),
    ('block', 'launch__block_size', 1),
]
STALL = re.compile(r'pu__we_average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio\s+\S+\s+([\d.]+)')


def raw(report):
    text = subprocess.run(['acu', '-i', str(report), '--page', 'raw'], capture_output=True, text=True).stdout
    vals = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals[parts[0]] = float(parts[-1].replace(',', ''))
            except ValueError:
                pass
    stalls = {m[1]: float(m[2]) for m in STALL.finditer(text)}
    # LLC request size from fabric bytes and transactions (all LLC slices).
    tx = vals.get('llc__transactions_load_pipe_fabric.avg')
    slices = vals.get('device__attribute_llc_active_num')
    by = vals.get('llc__bytes_load_pipe_fabric.sum')
    if tx and slices and by:
        vals['llc_bytes_per_request'] = by / (tx * slices)
    return vals, stalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reports', nargs='+', help='NAME=path.acurep')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    cols, data = [], []
    for spec in a.reports:
        name, _, path = spec.partition('=')
        cols.append(name)
        data.append(raw(Path(path)))
    lines = ['| metric | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    for label, key, scale in ROWS + [('llc_bytes_per_request', 'llc_bytes_per_request', 1)]:
        cells = []
        for vals, _ in data:
            v = vals.get(key)
            cells.append('-' if v is None else (f'{v * scale:.0f}' if abs(v * scale) >= 100 else f'{v * scale:.2f}'))
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    top = set()
    for _, stalls in data:
        top.update(k for k, _ in sorted(stalls.items(), key=lambda kv: -kv[1])[:4])
    for reason in sorted(top):
        lines.append(f'| stall {reason} | ' + ' | '.join(f"{st.get(reason, 0):.2f}" for _, st in data) + ' |')
    text = '\n'.join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + '\n')


if __name__ == '__main__':
    main()
