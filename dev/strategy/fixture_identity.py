"""Untimed endpoint checksums, not artifact authentication (separate SHA256 seals)."""
import zlib


def fingerprint(a, ids=None):
    import numpy as np
    a = np.ascontiguousarray(a, dtype='<f4')
    ids = np.ascontiguousarray(ids, dtype='<i4') if ids is not None else np.empty(0, dtype='<i4')
    return dict(a_bytes=a.nbytes, a_crc32=zlib.crc32(a.tobytes()),
                ids_bytes=ids.nbytes, ids_crc32=zlib.crc32(ids.tobytes()))


def expected(w, repeat=1):
    import numpy as np
    from dev.native_components.fixture import activation
    from dev.strategy.prefill_fixture import routing
    from dataclasses import asdict
    value = activation(dict(workload=asdict(w), cohort='decode' if w.m <= 8 else 'prefill'))
    coeff = np.asarray(value['coefficients'][repeat], dtype='<f4').reshape(w.m*w.channels,4)
    return fingerprint(coeff[:,value['categories']],
                       routing(w,repeat)['owners'] if w.operation == 'indexed' else None)
