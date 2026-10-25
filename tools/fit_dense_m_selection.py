#!/usr/bin/env python3
"""Reproduce the reviewed dense M1-8 exact table from its compact raw upload."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.dense_m_selection import TABLE, capture
from dev.strategy.production import serialize
from tools.generate_decode_selection import EVIDENCE, effective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, default=TABLE)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    baseline = effective(json.loads(EVIDENCE.read_text()), include_dense=False)
    data = capture(a.archive, baseline)
    text = serialize(data)
    if a.check:
        if a.output.read_text() != text:
            raise ValueError('dense M table differs from raw evidence')
    else:
        a.output.write_text(text)
    print('DENSE_M_TABLE '+json.dumps(dict(**data['summary'],
        evidence_sha256=data['evidence_sha256'], output=str(a.output), kernel_launches=0,
        bucket_changed=False, model_tpot_admitted=False)))


if __name__ == '__main__':
    main()
