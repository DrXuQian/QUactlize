"""Use the SAME GGUF bytes/F32 inputs as the historical component campaign."""
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np

from dev.bf16_compute.fixture import raw_weight, round_compute
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def activation(row):
    w=row['workload']
    categories=np.random.default_rng(81811+w['k']).integers(0,4,w['k'])
    compute='bf16'if row['cohort']=='prefill'else w['compute']
    coeff=[]
    for repeat in (0,1):
        values=np.random.default_rng(931+repeat).uniform(-.5,.5,(w['m']*w['channels'],4)).astype('f4')
        coeff.append(round_compute(values,compute).ravel().tolist())
    return dict(categories=categories.tolist(),coefficients=coeff,
                scope='SAME_HISTORICAL_F32_BYTES_NOT_NEW_ACTIVATION_QUANTIZATION',numpy=np.__version__)


def weights(w, folder):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'weights.raw';receipt=folder/'weights.json'
    contract={k:w[k]for k in ('qtype','n','k','experts')}
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['contract']!=contract or saved['sha256']!=sha(path):
            raise ValueError('historical raw fixture changed')
        return path,saved
    patterns=min(w['experts'],17);hashes=[]
    # Interrupted writes never occupy the final fixture name. Keep abandoned
    # staging files for diagnosis; resume does not delete another run's data.
    stage=folder/('weights-'+uuid.uuid4().hex+'.partial')
    last=time.monotonic()
    with stage.open('xb') as out:
        for e in range(patterns):
            h=hashlib.sha256()
            for first in range(0,w['n'],256):
                raw,_=raw_weight(w['qtype'],min(256,w['n']-first),w['k'],710+e*31+w['qtype']*1009+first)
                data=raw.tobytes();h.update(data);out.write(data)
                if time.monotonic()-last>=30:
                    print(f'NATIVE_COMPONENT_FIXTURE q={w["qtype"]} n={w["n"]} k={w["k"]} '
                          f'patterns={e}/{patterns} rows={first}/{w["n"]}',flush=True)
                    last=time.monotonic()
            hashes.append(h.hexdigest())
    if path.exists():
        if path.is_symlink() or sha(path)!=sha(stage):
            raise ValueError('unreceipted raw fixture differs from deterministic rebuild')
    else:
        stage.rename(path)
    saved=dict(contract=contract,sha256=sha(path),raw_sha256=hashes,
               patterns=patterns,repeat_expert_modulo=17,numpy=np.__version__,
               scope='ORIGINAL_HISTORICAL_GGUF_BYTES_NOT_REQUANTIZED_BY_LLAMA')
    atomic_json(receipt,saved)
    return path,saved
