"""Compile the policy's selected S1 closure, never the tuning inventory."""
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "policies/kpack_q4_decode_v1.json"


def recipes():
    groups = defaultdict(set)
    policy = json.loads(POLICY.read_text())
    if policy["schema"] != "quactlize.q4-decode-policy.v1":
        raise ValueError("Q4 decode policy schema differs")
    for r in policy["ranges"]:
        if r["role"] == "auto" and r["recipe"].startswith("simt:"):
            groups[r["n"], r["k"]].add(tuple(int(x[1:]) for x in r["recipe"][5:].split("-")))
    return {shape: sorted(values) for shape, values in sorted(groups.items())}


def sources():
    router = ['#include "q4_decode.h"']
    output = {}
    for (n, k), values in recipes().items():
        name = f"qkg_q4_decode_{n}_{k}"
        declaration = f'extern "C" int {name}(qkg_call_v1 const& c, qkg_q4_decode_config_v1 const& f)'
        router.append(declaration + ";")
        lines = ['#include "q4_decode_kernel.cuh"', declaration + " {"]
        for reader, variant, warps, p, columns in values:
            lines += [f"    if (f.reader=={reader} && f.variant=={variant} && f.warps=={warps} && f.values=={p} && f.columns=={columns})",
                      f"        return quactlize::execution::q4_decode::launch<{reader},{variant},{warps},{p},{columns},{n},{k}>(c);"]
        lines += ["    return QKG_SHAPE;", "}", ""]
        output[f"q4decode_{n}_{k}.cu"] = "\n".join(lines)
    router += ['extern "C" int qkg_q4_decode_launch(qkg_call_v1 const& c, qkg_q4_decode_config_v1 const& f) {']
    for n, k in recipes():
        router.append(f"    if (c.n=={n} && c.k=={k}) return qkg_q4_decode_{n}_{k}(c,f);")
    router += ["    return QKG_SHAPE;", "}", ""]
    output["q4decode_router.cpp"] = "\n".join(router)
    return output
