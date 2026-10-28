"""Product correctness-test support."""
import ctypes as C
import numpy as np
def graph_bind(sdk):
    for name, args in {
        "hggcStreamBeginCapture": [C.c_void_p, C.c_int],
        "hggcStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
        "hggcGraphInstantiateWithFlags": [
            C.POINTER(C.c_void_p),
            C.c_void_p,
            C.c_uint64,
        ],
        "hggcGraphLaunch": [C.c_void_p, C.c_void_p],
        "hggcGraphExecDestroy": [C.c_void_p],
        "hggcGraphDestroy": [C.c_void_p],
    }.items():
        fn = getattr(sdk.lib, name)
        fn.argtypes, fn.restype = args, C.c_int


def profiles(experts):
    if experts == 4:
        return [(129, 0, 1, 1), (1, 129, 0, 1), (0, 1, 1, 129)]
    # 128 tokens, top8, all 256 experts, then replay with permuted rows.
    rows = np.bincount(
        np.array([(np.arange(8) * 17 + t * 13) % experts for t in range(128)]).reshape(
            -1
        ),
        minlength=experts,
    )
    return [tuple(int(x) for x in np.roll(rows, shift)) for shift in (0, 3, 17)]
