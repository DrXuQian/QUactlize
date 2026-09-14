"""Host-side evidence for the installed BF16 provider's overwrite contract."""
import hashlib

import numpy as np


def zero_observation(bits, indices):
    """Count NaNs, finite residuals and their coordinates without a GPU reduction.

    NaN bits alone do not prove that a store was omitted: arithmetic can also
    produce them. Accept either sign of zero, as the original tensor check did.
    """
    bits = np.asarray(bits)
    indices = np.asarray(indices)
    if bits.dtype != np.dtype('<u2') or bits.ndim != 2 or indices.shape != (bits.shape[0],):
        raise ValueError('BF16 zero observation shape/dtype differs')
    magnitude = bits & 0x7fff
    special = (magnitude & 0x7f80) == 0x7f80
    nan = special & ((magnitude & 0x7f) != 0)
    bad_per_row = np.count_nonzero(magnitude, axis=1)
    bad = int(bad_per_row.sum())
    first = []
    for row in np.flatnonzero(bad_per_row):
        for col in np.flatnonzero(magnitude[row])[:8-len(first)]:
            first.append(dict(row=int(row), expert=int(indices[row]), n=int(col),
                              bits=f'0x{int(bits[row, col]):04x}'))
        if len(first) == 8:
            break
    return dict(cells=int(bits.size), bad=bad, zero=int(bits.size)-bad,
                negative_zero=int(np.count_nonzero(bits == 0x8000)),
                nan=int(np.count_nonzero(nan)), inf=int(np.count_nonzero(special & ~nan)),
                finite_nonzero=int(np.count_nonzero((magnitude != 0) & ~special)),
                bad_rows=int(np.count_nonzero(bad_per_row)), first=first,
                sha256=hashlib.sha256(np.ascontiguousarray(bits)).hexdigest())
