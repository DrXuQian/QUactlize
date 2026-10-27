#!/usr/bin/env python3
"""Audit MoE rule recall locally; launch no GPU work and reuse no old timings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.moe_plan import RULES, load_catalog, make_plan, source_stamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='new host audit JSON, not a sweep result')
    p.add_argument('--seed-width', type=int, choices=(1, 2), default=RULES['per_stratum'],
                   help='blind seeds per rule stratum; missing champions fail, never get appended')
    p.add_argument('--models', type=Path, help='JSON list of {name,path,tp}; header-only actual model inventory')
    p.add_argument('--llama-source', type=Path, help='caller sources for CPU-only TP split derivation')
    args = p.parse_args()
    if args.output.exists():
        p.error('--output must be a new file')
    before = source_stamp()
    models = None
    if args.models:
        from dev.strategy.moe_inventory import collect
        from dev.prefill_asys.tp_header import HeaderTP
        splitter = HeaderTP(args.llama_source, args.output.parent / (args.output.stem+'-tp'), indexed=True) if args.llama_source else None
        models = collect(json.loads(args.models.read_text()), partitioner=splitter)
    table, references, sources = load_catalog()
    options = RULES | {'per_stratum': args.seed_width}
    if models is not None:
        from dev.strategy.moe_inventory import add_candidates
        models = add_candidates(models, table, options)
    result = make_plan(table, references, sources, rules=options)
    from dev.strategy.moe_fusion import historical_recall
    fusion = historical_recall()
    after = source_stamp()
    if before['source_sha256'] != after['source_sha256'] or before['quactlize_commit'] != after['quactlize_commit']:
        raise ValueError('source changed during host audit')
    with args.output.open('x') as out:
        json.dump(dict(plan=result, source=before, models=models, fusion=fusion), out, sort_keys=True, separators=(',', ':'))
        out.write('\n')
    print('MOE_RULE_AUDIT ' + json.dumps(dict(status=result['status'], counts=result['counts'],
          source=before, output=str(args.output), historical_timing_reuse=False,
          fusion_reference_points=len(fusion['points']), fusion_status=fusion['status'],
          sweep_ready=False, kernel_launches=0, production_changed=False)), flush=True)
    return 0 if result['status'] == fusion['status'] == 'PASS' and (models is None or models['status'] == 'PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
