#!/usr/bin/env python3
"""ASK THE LIBRARY WHAT IT CONTAINS, instead of maintaining a second list of what we think it contains.

tune.py --shipped needs the set of configs the .so can actually select. A hand-maintained file would be a second
copy of a fact the library already knows, and the failure of a second copy is always the same: it goes stale
without saying so, and the tune then either ranks configs that are not there or misses ones that are.

quactlize_ppu_list_configs() exists precisely so the question can be asked. Its header states that no CUDA or PPU
context is required, which is what makes this usable OFFLINE -- a tuner on a machine with the library but no
device can still enumerate.

    python3 tools/list_shipped.py libquactlize_ppu.so                 # names, one per line
    python3 tools/list_shipped.py libquactlize_ppu.so | tee shipped.txt
    python3 tools/tune.py --model ... --shipped shipped.txt

THE RECORD'S FIRST FIELD IS A FAMILY DISCRIMINATOR, and this honours it. When enable_cuda_kernel is set the tile
fields carry no meaning, so printing them would invent a tile geometry for a CUDA-core GEMV that has none. Those
rows print their name alone -- which is also what tune.py's canonical() will refuse to parse as a tile, so a
GEMV entry cannot be silently matched against a tile config.
"""
import argparse
import ctypes
import pathlib
import sys


class Config(ctypes.Structure):
    # Field order and types must match quactlize/include/quactlize_ppu_config.h exactly. A mismatch here does not
    # fail loudly -- it reads adjacent memory and prints plausible numbers, which is worse than a crash.
    _fields_ = [
        ("enable_cuda_kernel", ctypes.c_bool),
        ("name", ctypes.c_char_p),
        ("tile_m", ctypes.c_int32),
        ("tile_n", ctypes.c_int32),
        ("warp_m", ctypes.c_int32),
        ("warp_n", ctypes.c_int32),
        ("stages", ctypes.c_int32),
    ]


def shipped(so_path: str):
    """-> [ (name, dict or None) ]. The dict is None for a CUDA-family entry, whose tile fields are meaningless."""
    lib = ctypes.CDLL(so_path)
    fn = lib.quactlize_ppu_list_configs
    fn.restype = ctypes.c_int32
    fn.argtypes = [ctypes.POINTER(ctypes.POINTER(Config))]
    arr = ctypes.POINTER(Config)()
    n = fn(ctypes.byref(arr))
    if n <= 0:
        raise RuntimeError(f"{so_path} reports {n} configs. A library that ships none cannot be tuned against, "
                           f"and its own static_assert says it must ship a set rather than one frozen tactic.")
    out = []
    for i in range(n):
        c = arr[i]
        name = c.name.decode() if c.name else f"<unnamed #{i}>"
        out.append((name, None if c.enable_cuda_kernel else
                    dict(tm=c.tile_m, tn=c.tile_n, wm=c.warp_m, wn=c.warp_n, st=c.stages)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("so", help="path to libquactlize_ppu.so")
    ap.add_argument("--verbose", action="store_true", help="show the tile fields too")
    a = ap.parse_args()
    if not pathlib.Path(a.so).is_file():
        print(f"no such library: {a.so}", file=sys.stderr)
        return 2
    try:
        rows = shipped(a.so)
    except OSError as e:
        print(f"could not load {a.so}: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    for name, tile in rows:
        if a.verbose:
            print(f"{name:<24} " + ("CUDA-core (tile fields carry no meaning)" if tile is None else
                                    f"tile {tile['tm']}x{tile['tn']}  warp {tile['wm']}x{tile['wn']}  "
                                    f"stages {tile['st']}"))
        else:
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
