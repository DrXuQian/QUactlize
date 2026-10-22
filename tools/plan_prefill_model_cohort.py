#!/usr/bin/env python3
"""Read GGUF headers and actual TP logs; freeze dense prefill shapes, no GPU work."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.prefill_asys.cohort import collect, MODEL_PATHS
from dev.strategy.journal import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, help='JSON list of {name,path,tp,tp_logs}, overrides presets')
    parser.add_argument('--model-root', type=Path, default=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo'))
    parser.add_argument('--names', nargs='+', choices=tuple(MODEL_PATHS), default=list(MODEL_PATHS))
    parser.add_argument('--deepseek-model', type=Path, help='override exact first-shard path')
    parser.add_argument('--tp4-log', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new; preserve the previous inventory')
    models = json.loads(args.models.read_text()) if args.models else [
        dict(name=name, path=str(args.model_root/MODEL_PATHS[name][0]), tp=MODEL_PATHS[name][1],
             tp_logs=[str(p) for p in args.tp4_log] if name=='deepseek4-flash-tp4' else [])
        for name in args.names]
    if args.deepseek_model:
        for m in models:
            if m['name']=='deepseek4-flash-tp4': m['path']=str(args.deepseek_model)
    result = collect(models, args.m)
    atomic_json(args.output, result)
    print('PREFILL_COHORT '+json.dumps(dict(status=result['status'], points=len(result['points']),
        blocked=len(result['blocked']), excluded=len(result['excluded']), output=str(args.output),
        gpu_work=False, production_changed=False)))
    for item in result['blocked'][:8]: print('PREFILL_COHORT_BLOCKED '+json.dumps(item))
    return 0 if result['status'] == 'READY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
