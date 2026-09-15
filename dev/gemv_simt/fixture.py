"""Canonical bytes and factorized dots from official GGUF, not the new decoder."""
import numpy as np

from tools.kpack_execution_fixture import IndexedWeights
from tools.kpack_warmup_fixture import activation_values, pack_plane


def weights(q, n, k, experts):
    if q != 8:
        return IndexedWeights(q, n, k, experts)
    # Match the factorized K-quant oracle's interface, without inventing
    # a packed-metadata unit for Q8. Its d is resident FP16 [E,K/32,N].
    from types import SimpleNamespace
    rng = np.random.default_rng(93908 + n + k)
    category = rng.integers(0, 4, k, dtype=np.uint8)
    result = SimpleNamespace(q=q, n=n, k=k, experts=experts,
        categories=np.tile(category, (experts, 1)), planes={},
        sums=np.empty((experts, 4, n)), abs_sums=np.empty((experts, 4, n)))
    result.planes = dict(low=np.empty((experts, k//2, n), dtype='<u2'),
                        high=np.empty(0, dtype='<u2'), units=np.empty((experts, k//32, n), dtype='<f2'))
    for e in range(experts):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        raw = rng.integers(0, 256, (n, k//32, 34), dtype=np.uint8)
        d = rng.uniform(-0.035, 0.035, (n, k//32)).astype('<f2')
        raw[..., :2] = d.view('u1').reshape(n, k//32, 2)
        result.planes['low'][e] = pack_plane(raw[..., 2:].reshape(n, k)^128, 8)
        result.planes['units'][e] = d.T
        official = dequantize(raw.reshape(-1), GGMLQuantizationType.Q8_0).reshape(n, k)
        for i in range(4):
            subset = official[:, category == i]
            result.sums[e, i] = subset.sum(axis=1, dtype='f8')
            result.abs_sums[e, i] = np.abs(subset).sum(axis=1, dtype='f8')
    return result


def inputs(w, tokens, mode, channels=1, repeat=0):
    if mode not in (0, 1, 2) or not 1 <= tokens <= 8:
        raise ValueError("undeclared small-M mode")
    topk = 8 if mode == 2 else 1
    rows = tokens*topk
    ids = np.empty(0, dtype='<i4')
    offsets = np.empty(0, dtype='<i4')
    if mode == 2:
        if w.experts < 8 or channels not in (1, 8):
            raise ValueError("indexed fixture requires top8 and shared/per-slot A")
        ids = np.full((tokens, 11), -71, dtype='<i4')
        ids[:, :8] = (np.arange(8)[None, :]*3 + np.arange(tokens)[:, None]*5 + repeat*7) % w.experts
        owner = ids[:, :8].reshape(-1)
        arows = np.arange(rows)//8*channels+np.arange(rows)%8%channels
        acount = tokens*channels
    elif mode == 1:
        owner = (np.arange(rows) * w.experts // rows).astype('<i4')
        counts = np.bincount(owner, minlength=w.experts)
        offsets = np.r_[0, counts.cumsum()].astype('<i4')
        arows, acount = np.arange(rows), rows
    else:
        owner, arows, acount = np.zeros(rows, dtype='<i4'), np.arange(rows), rows
    values = activation_values(np.arange(acount) + repeat*137)
    act = np.full((acount, w.k+8), 29, dtype='<f4')
    act[:, :w.k] = values[:, w.categories[0]].astype('<f2').astype('<f4')
    gold = np.stack([values[arows[r]] @ w.sums[e] for r, e in enumerate(owner)])
    denom = np.stack([np.abs(values[arows[r]]) @ w.abs_sums[e] for r, e in enumerate(owner)])
    if not np.isfinite(gold).all() or not np.any(gold) or not np.all(denom > 0):
        raise ValueError("degenerate official oracle")
    return dict(a=act, ids=ids, offsets=offsets, gold=gold, denom=denom, rows=rows,
                mode=mode, topk=topk, channels=channels, tokens=tokens)
