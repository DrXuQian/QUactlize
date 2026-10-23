#!/usr/bin/env python3
"""Export audited dense prefill winners; input order is oldest to newest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.dispatch.dense_prefill import EVIDENCE, HEADER, import_reviews, render, summary, audit_buckets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, action='append', default=[])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--audit-buckets', action='store_true')
    args = parser.parse_args()
    if args.check:
        data = json.loads(EVIDENCE.read_text())
        if render(data) != HEADER.read_text():
            raise ValueError('dense prefill generated header differs')
    else:
        if not args.review:
            parser.error('--review required in chronological replacement order')
        data = import_reviews(args.review)
        # Generated data, not a source rewrite or a GPU measurement.
        EVIDENCE.write_text(json.dumps(data, sort_keys=True, separators=(',', ':'))+'\n')
        HEADER.write_text(render(data))
    print('DENSE_PREFILL_POLICY '+json.dumps(summary(data), sort_keys=True))
    if args.audit_buckets:
        print('DENSE_PREFILL_BUCKET_AUDIT '+json.dumps(audit_buckets(data), sort_keys=True))


if __name__ == '__main__':
    main()
