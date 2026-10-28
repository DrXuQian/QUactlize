"""Product correctness-test support."""
import numpy as np
from ci.support.routing import real_ids
def workloads():
    return [dict(q=q, n=1024, k=5120, experts=1, m=4096) for q in range(10, 15)] + [
        dict(q=11, n=n, k=512, experts=256, m=32768) for n in (2048, 3072)]


def routing(m, experts, channels, repeat):
    if experts == 1:
        rows = np.arange(m, dtype='i4')
        return rows, rows, np.array([0, m], dtype='i4'), np.zeros(m, dtype='i4'), rows
    ids = real_ids(m//8).reshape(-1)
    # Include a sparse set with many empty experts, then change it. These
    # are valid directories, not a host-selected production route.
    if repeat == 1:
        ids = (np.arange(m)*7 % 16 + 31).astype('i4')
    else:
        ids = ((ids + 37*repeat) % experts).astype('i4')
    order = np.argsort(ids, kind='stable').astype('i4')
    a_rows = (np.arange(m)//8*channels + np.arange(m)%8%channels).astype('i4')
    offsets = np.r_[0, np.bincount(ids, minlength=experts).cumsum()].astype('i4')
    return a_rows[order], order, offsets, ids, a_rows


def require_guards(raw, byte, size):
    if raw[:size] != bytes([byte])*size or raw[-size:] != bytes([byte])*size:
        raise ValueError('composition output/workspace guard changed')
