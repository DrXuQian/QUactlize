"""Explicit Q4 affine C/W/P space; no change to decoder, format, or shipping policy."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from dev.gemv_cuda.build import digest
from dev.gemv_ppu.cold_shapes import source as fixed_source, verify as verify_previous
from dev.gemv_ppu.run import SHAPES

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "quactlize.q4-cold-config-space.v1"
COLUMNS = (1, 2, 4, 8, 16)
WARPS = (1, 2, 4, 5, 8, 10, 16, 20, 32)
VALUES = (2, 4, 8)
MAX_PASSES = 16


def payload(n,k):
    if (n,k) not in SHAPES:
        raise ValueError("unknown payload shape")
    return f"libq4_ppu_config_n{n}_k{k}.so"


@dataclass(frozen=True, order=True)
class Config:
    columns: int
    warps: int
    values: int

    @property
    def key(self):
        return f"c{self.columns}-w{self.warps}-p{self.values}"

    @property
    def args(self):
        return (self.columns, self.warps, self.values)

    def geometry(self, n, k):
        tile_n = self.columns*self.values
        workers = self.warps*32//self.columns
        groups = k//32
        return dict(**asdict(self), tile_n=tile_n, grid=n//tile_n, threads=self.warps*32,
                    k_workers=workers, k_passes=(groups+workers-1)//workers,
                    last_pass_workers=(groups-1)%workers+1, inter_cta_split=1)


def disposition(config, n, k):
    if (n,k) not in SHAPES or config.columns not in COLUMNS or config.warps not in WARPS or config.values not in VALUES:
        return "OUTSIDE_DECLARED_SPACE"
    if config.columns*config.values > 32:
        return "INVALID_REDUCE_SCATTER_TILE"
    g = config.geometry(n,k)
    if g["k_workers"] > k//32:
        return "PRUNED_IDLE_K_WORKERS"
    if g["k_passes"] > MAX_PASSES:
        return "PRUNED_SERIAL_PASS_BUDGET"
    return "CANDIDATE"


def inventory(n,k):
    return [Config(c,w,p) for p in VALUES for c in COLUMNS for w in WARPS
            if disposition(Config(c,w,p),n,k)=="CANDIDATE"]


def lookup(n,k,key):
    found = [c for c in inventory(n,k) if c.key==key]
    if len(found)!=1:
        raise ValueError("config not in the compiled shape inventory: "+key)
    return found[0]


def plan():
    return dict(schema=SCHEMA, qtype=12, m=1, mode="rotating", columns=COLUMNS, warps=WARPS,
                values=VALUES, max_passes=MAX_PASSES, reference_regression_limit_pct=0,
                scope="BEST_MEASURED_CONFIG_IN_DECLARED_SPACE_NOT_GLOBAL_OPTIMUM",
                cases=[dict(n=n,k=k,candidates=[dict(key=c.key,**c.geometry(n,k)) for c in inventory(n,k)],
                    excluded=[dict(key=c.key,reason=disposition(c,n,k))
                        for p in VALUES for col in COLUMNS for w in WARPS
                        for c in [Config(col,w,p)] if disposition(c,n,k)!="CANDIDATE"])
                    for n,k in SHAPES])


def source(n,k):
    old = fixed_source()
    body = old[:old.index('extern "C" int q4_cold_shapes_run(')]
    # The unused diagnostic C stub is not part of the new config ABI.
    stub = 'extern "C" int qkg_launch_12(qkg_call_v1 const&,qkg_config_v1 const&) {return QKG_INVALID;}'
    if body.count(stub)!=1:
        raise ValueError("old diagnostic stub seam differs")
    body = body.replace(stub, "")
    body += f'''extern "C" int q4_config_run_{n}_{k}(int columns,int warps,int values,
        void const* a,void const* low,void const* units,void* out,void* stream) {{
    if(!a || !low || !units || !out || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(out)&3)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for c in inventory(n,k):
        body += f'''    if(columns=={c.columns} && warps=={c.warps} && values=={c.values}) {{
        q4_group_affine<{c.columns},{c.warps},{c.values},{n},{k},true><<<{n//(c.columns*c.values)},{c.warps*32},0,static_cast<hggcStream_t>(stream)>>>(
            a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return body+"    return QKG_INVALID;\n}\n"


def verify(candidate,previous,controls,baseline,*,sources=True):
    old = verify_previous(previous,controls,baseline,sources=sources)
    m = json.loads((candidate/"manifest.json").read_text())
    expected = json.loads(json.dumps(plan()))
    if (m.get("schema")!=SCHEMA or m.get("plan")!=expected or
            m.get("previous_manifest_sha256")!=digest(previous/"manifest.json") or
            m.get("compiler_sha256")!=old["compiler_sha256"] or
            set(m.get("payloads",{}))!={f"{n}x{k}" for n,k in SHAPES}):
        raise ValueError("config-space/image authority differs")
    for n,k in SHAPES:
        row=m["payloads"][f"{n}x{k}"];path=candidate/payload(n,k)
        if row.get("file")!=path.name or digest(path)!=row["sha256"]:
            raise ValueError("config payload missing, changed, or still an LFS pointer")
        with path.open("rb") as stream:
            if stream.read(4)!=b"\x7fELF":raise ValueError("config payload is not ELF")
    stats=m.get("isa_statistics",{})
    if stats.get("file")!="isa-stats.json" or digest(candidate/"isa-stats.json")!=stats.get("sha256"):
        raise ValueError("config ISA receipt differs")
    for name,sha in m["source_hashes"].items() if sources else []:
        path=(ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path)!=sha:
            raise ValueError("config source differs: "+name)
    return m
