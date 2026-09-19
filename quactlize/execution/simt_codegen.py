"""Shared compiled SIMT inventory; config selection remains caller-owned."""
from dataclasses import asdict, dataclass
from itertools import product

QTYPES = (8, 10, 11, 12, 13, 14)
SPLITS = (1, 2, 4, 8)


@dataclass(frozen=True, order=True)
class Config:
    variant: int
    columns: int
    warps: int
    values: int
    split: int = 1

    @property
    def key(self):
        return f"reuse-v{self.variant}-c{self.columns}-w{self.warps}-p{self.values}-s{self.split}"

    @property
    def tile_n(self):
        return self.columns * self.values

    def record(self):
        reader = "Q8_VECTOR" if self.variant >= 4 else "REGISTER_REUSE"
        return asdict(self) | {"key": self.key, "reader": reader}


def inventory(q, profile="full", *, legacy=False):
    if q not in QTYPES or profile not in ("full", "smoke"):
        raise ValueError("undeclared SIMT format/profile")
    # Metadata units shared by a warp must be the same unit. Q8 has one d
    # per K32 group, unlike the paired-superblock K-quant units.
    variants = (0, 1, 4, 5) if q == 8 and not legacy else range(2 if q == 8 else 4)
    geometry = [(c, w, p) for c, w, p in product((4, 8), (2, 4, 8), (2, 4, 8))
                if c * p <= 32]
    if profile == "smoke":
        geometry = [(4, 4, 4), (8, 4, 4)]
    return tuple(Config(v, c, w, p) for v in variants for c, w, p in geometry)


def runtime_inventory(q, profile="full"):
    return tuple(Config(c.variant, c.columns, c.warps, c.values, s)
                 for c in inventory(q, profile) for s in SPLITS)


def source(q, profile="full"):
    candidates = inventory(q, profile)
    conditions = [f"f->variant=={c.variant} && f->columns=={c.columns} && "
                  f"f->warps=={c.warps} && f->values=={c.values}" for c in candidates]
    body = '#include "measured_decode.cuh"\n'
    body += f'extern "C" bool qkg_simt_supported_{q}(qkg_simt_config_v1 const* f) {{\n'
    body += "    return " + " ||\n        ".join(f"({s})" for s in conditions) + ";\n}\n"
    body += f'extern "C" int qkg_simt_launch_{q}(qkg_call_v1 const* c,qkg_simt_config_v1 const* f) {{\n'
    for c, condition in zip(candidates, conditions):
        reader = "simt::q8_vector" if c.variant >= 4 else "simt"
        body += f"    if ({condition}) return quactlize::execution::{reader}::launch<"
        body += f"{q},{c.variant},{c.columns},{c.warps},{c.values}>(*c,f->split);\n"
    body += "    return QKG_INVALID;\n}\n"
    body += f'extern "C" int qkg_simt_launch_v2_{q}(qkg_simt_call_v2 const* c,qkg_simt_config_v1 const* f) {{\n'
    body += f'    if (quactlize::execution::simt::measured_decode(*c,*f)!=quactlize::execution::simt::MeasuredDecode::None) return quactlize::execution::simt::measured_decode_launch<{q}>(*c,*f);\n'
    for c, condition in zip(candidates, conditions):
        reader = "simt::q8_vector" if c.variant >= 4 else "simt"
        body += f"    if ({condition}) return quactlize::execution::{reader}::launch_v2<"
        body += f"{q},{c.variant},{c.columns},{c.warps},{c.values}>(*c,f->split);\n"
    return body + "    return QKG_INVALID;\n}\n"
