"""Product correctness-test support; performance runners are on experiments/ppu-tuning."""
import hashlib
import re
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
