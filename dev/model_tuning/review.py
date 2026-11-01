"""Small handoff: selection evidence/definitions and diagnostics, not binaries/traces."""
from collections import deque
import io
import json
from pathlib import Path
import sys
import tarfile
import time


def package(run):
    run=Path(run).resolve(strict=True)
    if not run.is_dir():raise ValueError('review needs the retained run directory')
    files=set()
    for pattern in ('source/.quactlize-source.json','results/inventory.json','results/coverage.json',
                    'results/policy.json','results/plans/campaign.json','results/plans/*/plan.json',
                    'results/runs/status.json','results/runs/coordinator.json',
                    'results/runs/*/run.json','results/runs/*/gpu-*/results/authority.json'):
        files.update(run.glob(pattern))
    def local(path):
        return path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(run)
    logs={}
    for pattern in ('console-*.log','results/runs/*.log','results/runs/*/gpu-*/*.log'):
        for path in sorted(run.glob(pattern)):
            if local(path):
                with path.open(errors='replace') as stream:
                    logs[str(path.relative_to(run))]=''.join(deque(stream,maxlen=120))[-32768:]
    output=run/('review-'+str(time.time_ns())+'.tgz')
    with tarfile.open(output,'w:gz') as archive:
        for path in sorted(files):
            if local(path):archive.add(path,arcname=str(path.relative_to(run)),recursive=False)
        data=json.dumps(dict(scope='POLICY_CONFIG_COMPONENT_EVIDENCE_AND_LOG_TAILS',
            raw_asys_retained_on_box=True,raw_root=str(run/'results/runs'),logs=logs),ensure_ascii=False).encode()
        info=tarfile.TarInfo('diagnostics.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
    return output


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('usage: python3 -m dev.model_tuning.review RUN')
    print('results='+str(package(sys.argv[1])),flush=True)
