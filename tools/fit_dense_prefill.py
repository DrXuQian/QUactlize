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
    parser.add_argument('--base', type=Path, help='existing sealed catalog for an exact-only helper update')
    parser.add_argument('--helper-plan', type=Path, help='measured fixed-mainloop helper factorial plan')
    parser.add_argument('--mean-plan', type=Path, help='frozen M-span plan, complete original + retry reviews')
    parser.add_argument('--mean-admission', type=Path, help='explicit source scope and accepted regression keys')
    args = parser.parse_args()
    if args.check:
        data = json.loads(EVIDENCE.read_text())
        if render(data) != HEADER.read_text():
            raise ValueError('dense prefill generated header differs')
    else:
        if not args.review:
            parser.error('--review required in chronological replacement order')
        if args.mean_plan or args.mean_admission:
            if not (args.base and args.mean_plan and args.mean_admission) or args.helper_plan:
                parser.error('mean update needs --base, --mean-plan and --mean-admission; no --helper-plan')
            from quactlize.dispatch.dense_prefill_mean import update_mean
            data = update_mean(json.loads(args.base.read_text()), args.review,
                               json.loads(args.mean_plan.read_text()), json.loads(args.mean_admission.read_text()))
        elif args.base or args.helper_plan:
            if not (args.base and args.helper_plan and len(args.review) == 1):
                parser.error('helper update needs --base, --helper-plan and one --review')
            from quactlize.dispatch.dense_prefill_update import update_helpers
            data = update_helpers(json.loads(args.base.read_text()), args.review[0], args.helper_plan)
        else:
            data = import_reviews(args.review)
        # Generated data, not a source rewrite or a GPU measurement.
        EVIDENCE.write_text(json.dumps(data, sort_keys=True, separators=(',', ':'))+'\n')
        HEADER.write_text(render(data))
    print('DENSE_PREFILL_POLICY '+json.dumps(summary(data), sort_keys=True))
    if args.audit_buckets:
        print('DENSE_PREFILL_BUCKET_AUDIT '+json.dumps(audit_buckets(data), sort_keys=True))


if __name__ == '__main__':
    main()
