"""Production ABI/catalog support."""
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))


from quactlize.runtime.compiler import validate_parent
from quactlize.runtime.tuning import ROUTES, digest
def requests():
    result = []
    dense = [
        (1024, 5120),
        (5120, 8192),
        (5120, 25600),
        (8192, 5120),
        (25600, 5120),
        (512, 2048),
        (2048, 512),
        (3072, 512),
    ]
    grouped = [(512, 2048), (2048, 512), (512, 3072), (3072, 512)]
    for q in (8, 10, 11, 12, 13, 14):
        for route in ((1,) if q == 8 else (0, 1)):
            for n, k in dense:
                for m in (1, 4, 8, 64, 128, 256, 512, 1024, 2048, 4096):
                    result.append((q, route, m, n, k, 1, m))
        for route in ((3,) if q == 8 else (2, 3)):
            for n, k in grouped:
                for tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
                    result.append((q, route, tokens * 8, n, k, 256, tokens))
    return result


def plan(output, inputs=None, decode=False,compute_type='f16'):
    if compute_type not in ('f16','bf16'):raise ValueError('unknown compute type')
    executable = output / "policy-query"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            f"-I{ROOT}",
            str(ROOT / "tools/kpack_native_policy.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    inputs = requests() if inputs is None else inputs
    lines = subprocess.check_output(
        [str(executable)] + (["--decode"] if decode else []) + (["--bf16"] if compute_type=='bf16' else []),
        text=True,
        input="".join(" ".join(map(str, r)) + "\n" for r in inputs),
    ).splitlines()
    if len(lines) != len(inputs):
        raise ValueError("native policy omitted requests")
    parents = {}
    selected = []
    for req, line in zip(inputs, lines):
        if line == "MISS":
            selected.append(dict(request=req, status="FALLBACK_REQUIRED"))
            continue
        p = line.split()
        values = list(map(int, p[1:]))
        (
            q,
            route,
            tm,
            tn,
            tk,
            wm,
            wn,
            stages,
            ap,
            dn,
            persistent,
            split,
            grid_mode,
            grid_b,
            kind,
        ) = values
        parent = dict(
            symbol=p[0],
            qtype=q,
            route=ROUTES[route],
            tm=tm,
            tn=tn,
            tk=tk,
            wm=wm,
            wn=wn,
            stages=stages,
            ap=ap,
            dn=dn,
            persistent=persistent,
        )
        validate_parent(parent)
        if p[0] in parents and parents[p[0]] != parent:
            raise ValueError("symbol aliases different parent tuples")
        parents[p[0]] = parent
        selected.append(
            dict(
                request=req,
                status="SELECTED",
                parent=p[0],
                split=split,
                grid_mode=grid_mode,
                grid_b=grid_b,
                policy=kind,
            )
        )
    return sorted(parents.values(), key=lambda p: p["symbol"]), selected


def catalog(records, jit_source=""):
    def typed(r):
        return r["identity"].get("endpoints") == "decode-m1-8-f32-bf16-v1"
    if len({(r["parent"]["symbol"],typed(r),r['identity'].get('compute_type','f16')) for r in records}) != len(records):
        raise ValueError("catalog has multiple builds of one parent")
    rows = []
    for r in records:
        p = r["parent"]
        strings = [p["symbol"], r["key"], digest(r["identity"])]
        values = [p["qtype"], ROUTES.index(p["route"])] + [
            p[f] for f in ("tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn")
        ]
        rows.append(
            "  {"
            + ",".join([json.dumps(s) for s in strings] + list(map(str, values)))
            + (",{},"+str(typed(r)).lower()+",1" if r['identity'].get('compute_type')=='bf16'
               else ",{},true" if typed(r) else "") + "},"
        )
    return ("static std::vector<Image> const kImages = {\n" + "\n".join(rows) + "\n};\n"
            + "static char const kJitSource[] = " + json.dumps(jit_source) + ";\n")
