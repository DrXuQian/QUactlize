#!/usr/bin/env python3
"""Check the exact TP2 legacy mask defect and admitted H32 machine-code identity.

No compilation, device execution, policy change or performance verdict.
The code hashes refer to the 62KWWD numerical A/B and its matching local build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

LEGACY_SHA = '8ef56b91df13ece8816ffdcd29c057b4d0393dddcda2bd157d815c6b96591cea'
CODE = {
    0: ('4a2ca349fccd0e714e74c25294439b0fc33bf745d2dd23cde615b42a71f71685',
        '45b1b4636b047ca2af05f9cc5f749de4522338a17727d729c6773792b6e279cb'),
    1: ('8f6dd5293116bfa5043fc519fcb292eb06cef17a1cd33695d94489205dda0d12',
        '8ae9a1787af4f33d9f7532233b35aa6bf610fbc2774ac23f56f340a786b15db4'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def instructions(text):
    rows = re.findall(r'^\s*([0-9a-f]+):((?:\s+[0-9a-f]{2}){8})\s+(.+)$', text, re.M)
    if not rows or [int(a,16) for a,_,_ in rows] != list(range(0,8*len(rows),8)):
        raise ValueError('missing, duplicate, or non-contiguous instruction inventory')
    raw = b''.join(bytes.fromhex(b) for _,b,_ in rows)
    return {int(a,16): op.strip() for a,_,op in rows}, hashlib.sha256(raw).hexdigest()


def mask_model():
    result = []
    for warp in (0,1):
        groups = [8*warp+lane//4 for lane in range(32)]
        bits = [32+48*((g%8)//4)+6*(g%4) for g in groups]
        missing = [lane for lane,b in enumerate(bits) if b%32+6>32]
        result.append(dict(warp=warp, groups=sorted({groups[l] for l in missing}),
                           missing_lanes=missing, mask=sum(1<<l for l in missing),
                           n_residues=[(l%4)*4 for l in missing]))
    for scale in range(64):
        word2, word3 = (scale&15)<<28, scale>>4
        before_restore = ((0>>28)|(word3<<4))&63
        after_restore = ((word2>>28)|(word3<<4))&63
        if before_restore != scale&0x30 or after_restore != scale:
            raise ValueError('field-loss mask model differs')
    return result


def inspect(text, compute, candidate):
    ops, digest = instructions(text)
    if digest != CODE[compute][int(candidate)]:
        raise ValueError('machine code differs from the frozen case; inspect before re-admitting')
    if candidate:
        if any('ivreg' in op for op in ops.values()):
            raise ValueError('H32 candidate still uses indirect register reads')
    else:
        offset = 0 if compute == 0 else 0x88
        anchors = {
            0x30c0: 'v.cmp.lt.i32\tvcc, vreg27, 0x1b',
            0x3120: 's.lop.emsk\tsreg29, vcc # 0x8',
            0x3830: 'v.mov.b32\tvreg39, ivreg',
            0x3850: 's.lop.emsk\tsreg29, sreg29 # 0xe',
            0x3858: 'v.shrl.b32\tvreg39, vreg39, vreg27',
        }
        if any(ops.get(pc-offset) != op for pc,op in anchors.items()):
            raise ValueError('legacy mask/read/reconvergence dataflow differs')
    return dict(code_sha256=digest, instructions=len(ops),
                indirect_register_reads=sum('ivreg' in op for op in ops.values()))


def main(args):
    if sha(args.legacy) != LEGACY_SHA:
        raise ValueError('not the shipped image bound to the 62KWWD results')
    result = dict(scope='STATIC_ISA_IDENTITY_AND_MASK_MODEL_NOT_DEVICE_OR_PERFORMANCE_GATE',
                  legacy_sha256=LEGACY_SHA, candidate_sha256=sha(args.candidate),
                  mask_model=mask_model(), cases=[])
    for compute in (0,1):
        symbol = (f'_ZN9quactlize9execution4simt14register_reuseILi12ELi1ELi0ELi4ELi4ELi4ELi{compute}'
                  'ELi0EEEv11qkg_call_v1i')
        row = dict(compute='BF16' if compute else 'F16', symbol=symbol)
        for name,path in (('legacy',args.legacy),('candidate',args.candidate)):
            dump = subprocess.check_output([args.sdk/'bin/hgobjdump','--dump-isa',
                                            '--dump-function='+symbol,path],text=True)
            row[name] = inspect(dump,compute,name=='candidate')
        result['cases'].append(row)
    result['status'] = 'SHIPPED_MASK_DEFECT_AND_FIXED_CODE_IDENTITY_CONFIRMED'
    text = json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text,end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','legacy','candidate'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output',type=Path)
    main(parser.parse_args())
