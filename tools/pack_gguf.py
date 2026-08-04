#!/usr/bin/env python3
"""GGUF -> BC artifacts, one per tensor, with the arrangement recorded beside them.

WHY THIS DOES NOT WAIT FOR THE ON-DISK FORMAT. Producing (low, high, units) has been possible and
device-validated for a while; what is still undecided is the CONTAINER llama.cpp will read. Those are separate,
and coupling them would have held a working tool behind a decision. So the writer is one small function at the
bottom and everything above it is container-agnostic: when the GGUF-native form lands, that function changes and
nothing else does.

IT IS ALSO USEFUL BEFORE llama.cpp EXISTS AS A CONSUMER. Every artifact this repo has measured so far came from a
SYNTHESISED fixture -- random code bytes with sane fp16 headers, chosen because the official gguf package has no
k-quant quantiser to ask for the bytes of a given weight. That fixture is deliberate and it covers the code space
better than a real checkpoint would, but it cannot answer questions about a real model's shapes, its mix of
formats, or how long packing one takes.

WHAT IT REFUSES TO DO, and each refusal is a mistake this project has made or nearly made:
  * it does not GUESS an arrangement. The (fold, tile_k) a tensor is packed for is recorded per tensor, because
    dense and MoE want different folds and a header that omits them leaves the reader to assume.
  * it does not silently skip a tensor. Anything unsupported is listed with its type, so "packed the model" and
    "packed the tensors we happened to handle" are distinguishable.
  * it does not write a partial artifact on failure. A directory that exists is a directory that finished.

    python3 tools/pack_gguf.py MODEL.gguf OUT_DIR [--limit N] [--dry-run]

Needs the device library (the placement lives there), so it runs on the box:
    QUACTLIZE_PPU_LIB=<...>/libquactlize_ppu.so python3 tools/pack_gguf.py ...
"""
import argparse
import json
import pathlib
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a .gguf file")
    ap.add_argument("out", help="output directory; created, and only finished artifacts land in it")
    ap.add_argument("--limit", type=int, default=0, help="stop after N packable tensors (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be packed, touching no device and writing nothing")
    a = ap.parse_args()

    try:
        from gguf import GGUFReader
    except ImportError:
        print("needs the official gguf package: pip install gguf", file=sys.stderr)
        return 2

    from quactlize import formats as F

    reader = GGUFReader(a.model)
    # THE SUPPORTED SET COMES FROM THE FORMAT TABLE, not from a list here. A second list is a second place to
    # forget a format, and this file would forget it silently -- the tensor would land in "skipped" looking like
    # an unsupported type rather than like an omission.
    supported = {int(q): q for q in (F.QuantType.Q2_K, F.QuantType.Q3_K, F.QuantType.Q4_K,
                                     F.QuantType.Q5_K, F.QuantType.Q6_K)}

    seen, packable, skipped = Counter(), [], []
    for t in reader.tensors:
        tt = int(t.tensor_type)
        seen[t.tensor_type.name] += 1
        if tt in supported and len(t.shape) == 2:
            packable.append(t)
        else:
            why = "not a k-quant this build packs" if tt not in supported else f"{len(t.shape)}-D, expected 2-D"
            skipped.append((t.name, t.tensor_type.name, why))

    print(f"model      {a.model}")
    print(f"tensors    {sum(seen.values())}  ->  {len(packable)} packable, {len(skipped)} skipped")
    print(f"type mix   {dict(seen)}")
    # THE MIX IS THE POINT OF PRINTING IT. A _K_M checkpoint is named for its dominant format and carries others,
    # and one device library serves ONE PPU_PACKED_FORMAT -- so this line says how many libraries a deployment of
    # this model needs, which is a fact about our build rather than about the file.
    kq = {n: c for n, c in seen.items() if n in {q.name for q in supported.values()}}
    if len(kq) > 1:
        print(f"           MIXED: {len(kq)} k-quant formats in one file {kq} -- one library per format")

    if skipped:
        print(f"\nskipped ({len(skipped)}), first 10:")
        for n, ty, why in skipped[:10]:
            print(f"  {n:<48} {ty:<8} {why}")

    if a.dry_run:
        print("\n--dry-run: nothing written, no device touched")
        return 0

    import torch
    from quactlize import routes
    import quactlize

    backend = quactlize.gguf_backend()
    if not backend.startswith("ppu"):
        print(f"\nrefusing to pack on backend '{backend}'.\n"
              f"  The placement lives in the device library, so a host fallback would either fail or -- worse --\n"
              f"  produce something. Set QUACTLIZE_PPU_LIB to a built libquactlize_ppu.so.", file=sys.stderr)
        return 3

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest, t0 = [], time.time()
    todo = packable[:a.limit] if a.limit else packable
    for i, t in enumerate(todo):
        n, k = int(t.shape[1]), int(t.shape[0])         # gguf reports (k, n); the routes want (n, k)
        qtype = int(t.tensor_type)
        blocks = torch.from_numpy(t.data.reshape(n * (k // 256), -1).copy())
        low, high, units = routes.prepare_fully_quantized_dense(blocks, n, k, qtype)

        stem = t.name.replace("/", "_").replace(".", "_")
        tmp = out / (stem + ".partial")
        tmp.mkdir(exist_ok=True)
        _write(tmp, low, high, units)
        tmp.rename(out / stem)                          # a directory that exists is a directory that finished

        arr = F.PlacedArrangement(bits=_low_bits(qtype), fold=1, tile_k=_tile_k(qtype),
                                  high_fold=1)
        manifest.append({"name": t.name, "dir": stem, "ggml_type": qtype, "type_name": t.tensor_type.name,
                         "n": n, "k": k, "arrangement": arr._asdict(),
                         "shapes": {"low": list(low.shape), "high": list(high.shape), "units": list(units.shape)}})
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  packed {i+1}/{len(todo)}  ({time.time()-t0:.1f}s)")

    (out / "manifest.json").write_text(json.dumps({"model": a.model, "tensors": manifest}, indent=2) + "\n")
    print(f"\nwrote {len(manifest)} artifact(s) + manifest.json to {out}")
    print("The manifest carries the ARRANGEMENT per tensor, not per format: dense and MoE want different folds,\n"
          "and a reader that has to infer it is a reader that will one day infer wrongly.")
    return 0


def _low_bits(qtype: int) -> int:
    from quactlize import schemes
    from quactlize.formats import QuantType
    return {"i2": 2, "i2+i1": 2, "i4": 4, "i4+i1": 4, "i4+i2": 4}[schemes.CODE_PLANE[QuantType(qtype)]]


def _tile_k(qtype: int) -> int:
    """Q6_K keeps TK=128; the rest use 256. Not a preference -- TK=256's high-plane map for Q6 is incomplete,
    which is what produced conditioned error 8.76e-1 before it was caught."""
    from quactlize.formats import QuantType
    return 128 if QuantType(qtype) is QuantType.Q6_K else 256


def _write(d: pathlib.Path, low, high, units) -> None:
    """THE CONTAINER, and the only part that changes when the on-disk format is decided.

    Raw .npy today: it needs no schema, any reader has it, and nothing above this function knows about it. When
    the GGUF-native representation lands this becomes a writer into that, and the rest of the tool is unchanged.
    """
    import numpy as np
    for name, t in (("low", low), ("high", high), ("units", units)):
        np.save(d / f"{name}.npy", t.numpy())


if __name__ == "__main__":
    raise SystemExit(main())
