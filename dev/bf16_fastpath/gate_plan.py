"""Bound the typed Q4 gate by the production selector's compiled inventory."""
import ctypes as C
import json
from pathlib import Path

from quactlize.execution import q4_decode_codegen as codegen
from quactlize.execution.native import Call, SimtCallV2, Arrangement, Sizes, arrangement

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "quactlize.bf16-q4-selected-gate.v1"
MOE_SHAPES = {(512, 2048), (512, 3072), (2048, 512), (3072, 512)}
FIELDS = ("reader", "variant", "warps", "values", "columns")


class Config(C.Structure):
    _fields_ = [("version", C.c_uint32), ("size", C.c_uint32)] + [
        (name, C.c_int32) for name in FIELDS]


def plan():
    policy = json.loads(codegen.POLICY.read_text())
    compiled = codegen.recipes()
    rows = []
    for r in policy["ranges"]:
        if r["role"] != "auto":
            continue
        indexed = r["operator"] == "grouped"
        if indexed and (r["n"], r["k"]) not in MOE_SHAPES:
            continue
        selected = r["recipe"].startswith("simt:")
        recipe = [int(x[1:]) for x in r["recipe"][5:].split("-")] if selected else None
        if selected and tuple(recipe) not in compiled.get((r["n"], r["k"]), []):
            raise ValueError("auto recipe is absent from compiled Q4 inventory")
        for tokens in range(r["first"], r["last"] + 1):
            for channels in ((1, 8) if indexed else (1,)):
                prefix = "indexed" if indexed else "dense"
                rows.append(dict(id=f"{prefix}-n{r['n']}-k{r['k']}-t{tokens}-ch{channels}",
                    n=r["n"], k=r["k"], tokens=tokens, channels=channels,
                    mode=2 if indexed else 0, experts=256 if indexed else 1,
                    rows=tokens * (8 if indexed else 1), recipe=recipe,
                    expected="SELECTED" if selected else "QKG_SHAPE"))
    rows.sort(key=lambda r: (r["mode"], r["n"], r["k"], r["tokens"], r["channels"]))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate selector requests")
    selected = sum(r["expected"] == "SELECTED" for r in rows)
    return dict(schema=SCHEMA, cases=rows, compute="BF16", storage=["F32", "BF16"],
        controls=["F16_V1_NOMINAL", "F16_V2_EQUALS_V1", "F16_OVERFLOW_EXPECTED_RED"],
        compiled={f"{n}x{k}": [list(x) for x in recipes] for (n, k), recipes in compiled.items()},
        denominator=dict(requests=len(rows), selected_requests=selected,
            declined_requests=len(rows)-selected, bf16_cells=selected*2,
            f16_controls=selected, overflow_negatives=selected),
        scope="SELECTED_Q4_S1_ONLY_NOT_GENERIC_SIMT_NOT_TC_NOT_PERFORMANCE")


def make_call(point, storage=1):
    return Call(version=1, size=C.sizeof(Call), qtype=12, n=point["n"], k=point["k"],
        experts=point["experts"], rows=point["rows"], mode=point["mode"], input_type=storage,
        channels=point["channels"], topk=8 if point["mode"] == 2 else 1,
        a_row_stride=point["k"]+8, a_token_stride=(point["k"]+8)*point["channels"]+8,
        ids_stride=11 if point["mode"] == 2 else 0, out_row_stride=point["n"]+8,
        a=0x2000000000, low=0x1000000000, units=0x1800000000,
        ids=0x4000000000 if point["mode"] == 2 else None, output=0x3000000000)


class Library:
    def __init__(self, path):
        self.lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
        for version, calltype in ((1, Call), (2, SimtCallV2)):
            query = getattr(self.lib, f"quactlize_kpack_q4_decode_select_v{version}")
            run = getattr(self.lib, f"quactlize_kpack_q4_decode_run_v{version}")
            query.argtypes = [C.POINTER(calltype), C.POINTER(Arrangement), C.POINTER(Config), C.POINTER(Sizes)]
            run.argtypes = [C.POINTER(calltype), C.POINTER(Config), C.POINTER(Arrangement)]
            query.restype = run.restype = C.c_int
            setattr(self, f"query{version}", query)
            setattr(self, f"run{version}", run)
        self.arr = arrangement(12)

    def select(self, point, storage=1, compute=1):
        call = make_call(point, storage)
        typed = SimtCallV2(call, compute)
        config, sizes = Config(), Sizes()
        rc = self.query2(C.byref(typed), C.byref(self.arr), C.byref(config), C.byref(sizes))
        if point["expected"] == "QKG_SHAPE":
            if rc != 24:
                raise ValueError(f"non-SIMT auto request returned {rc}, expected QKG_SHAPE")
            return None
        if rc or [getattr(config, name) for name in FIELDS] != point["recipe"]:
            raise ValueError(f"selected Q4 recipe differs: rc={rc} id={point['id']}")
        expected = (point["n"]*point["k"]*point["experts"]//2, 0,
                    point["n"]*point["k"]*point["experts"]//16, 0)
        if (sizes.low_bytes, sizes.high_bytes, sizes.units_bytes, sizes.workspace_bytes) != expected:
            raise ValueError("selected Q4 plane or workspace size differs")
        return typed, config, sizes
