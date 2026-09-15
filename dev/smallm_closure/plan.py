"""Freeze model headers, historical incumbents and a bounded candidate pool."""
from collections import defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from quactlize.execution.simt_codegen import runtime_inventory
from quactlize.gguf_roles import match_role
from quactlize.runtime.compiler import ROOT, sha, validate_parent
from quactlize.runtime.tuning import digest, ROUTES
from tools.fit_kpack_smallm import tc_config
from tools.gguf_internal_shape_inventory import read_gguf_header
from tools.resolve_kpack_batched_models import resolve_plan

SCHEMA = "quactlize.smallm-closure.v1"
QTYPES = (8, 10, 11, 12, 13, 14)
SOURCES = ("policies/kpack_smallm_v1.json", "policies/kpack_q4_decode_v1.json",
           "docs/measurements/smallm_20260915.json.gz")


def load(relative):
    path = ROOT / relative
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def parent(config):
    p = {f: config[f] for f in ("symbol", "qtype", "tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn")}
    p.update(route=ROUTES[config["route"]], persistent=config["parent_persistent"])
    validate_parent(p)
    return p


def tc(config, reason):
    p = parent(config)
    return dict(kind="tc", parent=p, split=config["split"],
                algorithm=int(config["grid_mode"] != 0), grid_b=config["grid_b"], grid_mode=config["grid_mode"],
                reason=reason)


def candidate_id(c):
    return digest({k: v for k, v in c.items() if k != "reason"})[:24]


def q4_config(text):
    if text.startswith("simt:"):
        values = [int(v[1:]) for v in text[5:].split("-")]
        return dict(kind="q4", recipe=dict(zip(("reader", "variant", "warps", "values", "columns"), values)),
                    split=1, reason="Q4_PRODUCTION_INCUMBENT")
    _, symbol, split, b, g = text.split(":")
    c = tc_config(symbol, int(split[1:]), "PERSISTENT" if int(g[1:]) else "NONPERSISTENT")
    c.update(grid_b=int(b[1:]), grid_mode=int(g[1:]))
    return tc(c, "Q4_HISTORICAL_TC")


def historical():
    """Never use historical times to decide this campaign's winner."""
    data = load(SOURCES[2])
    controls = defaultdict(list)
    for row in data["tc"]:
        controls[tuple(row["key"])].append(tc(row["config"], "HISTORICAL_TC"))
    for row in load(SOURCES[0])["exact"]:
        c = row["config"]
        value = tc(c, "PRODUCTION_INCUMBENT") if c["kind"] == "tc" else dict(
            kind="simt", recipe={k: c[k] for k in ("variant", "columns", "warps", "values", "split")},
            split=c["split"], reason="PRODUCTION_INCUMBENT")
        controls[tuple(row["key"])].append(value)
    for row in data["simt"]:
        # Preserve the measured SIMT winner, including its Split-K value.
        import re
        fields = re.fullmatch(r"reuse-v(\d+)-c(\d+)-w(\d+)-p(\d+)-s(\d+)", row["recipe"])
        if not fields:
            raise ValueError("historical SIMT recipe is not reproducible")
        recipe = dict(zip(("variant", "columns", "warps", "values", "split"), map(int, fields.groups())))
        controls[tuple(row["key"])].append(dict(kind="simt", recipe=recipe, split=recipe["split"],
                                              reason="HISTORICAL_SIMT"))
    for row in load(SOURCES[1])["ranges"]:
        mode = 0 if row["operator"] == "dense" else 2
        for m in range(row["first"], row["last"] + 1):
            for ch in ((1,) if mode == 0 else (1, 8)):
                c = q4_config(row["recipe"])
                # Public Q4 entry selects only the auto incumbent; do not
                # silently benchmark a different recipe via that entry.
                if c["kind"] == "q4" and row["role"] != "auto":
                    continue
                controls[12, mode, row["n"], row["k"], m, ch].append(c)
    return controls


def model_inventory(plan_path, model_root):
    resolved = resolve_plan(json.loads(Path(plan_path).read_text()),
        ["qwen35-35b-q4km", "qwen3-32b-q4km"], model_root)
    matrices, exclusions, receipts = [], [], []
    for model in resolved["models"]:
        tensors, metadata, shards = {}, {}, []
        for filename in model["files"]:
            path = Path(filename)
            with path.open("rb") as stream:
                header = read_gguf_header(stream, str(path))
                header_size = stream.tell()
                stream.seek(0)
                header_hash = hashlib.sha256(stream.read(header_size)).hexdigest()
            for item in header["tensors"]:
                if item["name"] in tensors:
                    raise ValueError("duplicate tensor across shards: " + item["name"])
                tensors[item["name"]] = item
            metadata.update(header["metadata"])
            shards.append(dict(path=str(path), size=path.stat().st_size, header_bytes=header_size,
                               header_sha256=header_hash))
        arch = metadata.get("general.architecture")
        topk = metadata.get(str(arch) + ".expert_used_count")
        if "output.weight" not in tensors and "token_embd.weight" in tensors:
            tensors["output.weight"] = tensors["token_embd.weight"] | dict(name="output.weight", tied=True)
        for name, item in tensors.items():
            dims, q = item["dims_gguf"], item["qtype"]
            if len(dims) not in (2, 3):
                continue
            matched = match_role(name, len(dims))
            if q not in QTYPES or (matched and matched[0].route_class not in ("dense", "grouped")):
                exclusions.append(dict(model=model["name"], tensor=name, q=q, dims=dims,
                                       reason="UNSUPPORTED_FORMAT_OR_NON_MATMUL"))
                continue
            if not matched:
                raise ValueError("unclassified quantized matrix; cannot certify coverage: " + name)
            grouped = matched[0].route_class == "grouped"
            if grouped and (not isinstance(topk, int) or topk <= 0):
                raise ValueError("model header lacks expert_used_count: " + model["name"])
            k, n = dims[:2]
            row = dict(q=q, mode=2 if grouped else 0, n=n, k=k,
                       experts=dims[2] if grouped else 1, topk=topk if grouped else 1,
                       channels=topk if grouped and "down_exps" in name else 1,
                       model=model["name"], tensor=name, fused=False)
            matrices.append(row)
            if name.endswith("ffn_gate_exps.weight"):
                up = tensors.get(name.replace("ffn_gate_exps", "ffn_up_exps"))
                if up and up["qtype"] == q and up["dims_gguf"] == dims:
                    matrices.append(row | dict(n=n * 2, fused=True,
                        tensor=name.replace("ffn_gate_exps", "ffn_gate_up_exps")))
            for gate, up_name in (("ffn_gate.weight", "ffn_up.weight"), ("ffn_gate_shexp.weight", "ffn_up_shexp.weight")):
                if name.endswith(gate):
                    up = tensors.get(name.replace(gate, up_name))
                    if up and up["qtype"] == q and up["dims_gguf"] == dims:
                        matrices.append(row | dict(n=n * 2, fused=True,
                            tensor=name.replace(gate, gate.replace("gate", "gate_up"))))
        receipts.append(dict(model=model["name"], shards=shards, topk=topk))
    if not matrices:
        raise ValueError("empty target model matrix inventory")
    return dict(matrices=matrices, exclusions=exclusions, receipts=receipts,
                identity_scope="GGUF_HEADERS_NOT_MODEL_PAYLOAD_HASHES_TP1")


def evidence_audit():
    data = load(SOURCES[2])
    a, b = {tuple(r["key"]) for r in data["simt"]}, {tuple(r["key"]) for r in data["tc"]}
    return dict(simt_contexts=len(a), tc_contexts=len(b), overlap=len(a & b),
                simt_without_tc=[list(k) for k in sorted(a - b)],
                tc_without_simt=[list(k) for k in sorted(b - a)],
                issues=[
        dict(id="CROSS_COHORT", severity="REMEASURE", detail="Old SIMT and TC cohorts differ; table membership is not a paired win."),
        dict(id="FP32_ENDPOINT", severity="REMEASURE", detail="Historical FP16 TC core costs omit external F32 adapters; new typed endpoints are timed in full."),
        dict(id="Q8_PROFILED", severity="REMEASURE", detail="Old Q8 TC costs include profiler/model state, not this rotating-weight cohort."),
        dict(id="BF16", severity="REMEASURE", detail="F16 winners are only BF16 proposals; BF16 capability PASS is not performance evidence. AP1 has no BF16 implementation."),
        dict(id="BUCKET_DISTANCE", severity="REMEASURE", detail="Nearest log bucket has no distance bound; include target heads and projections explicitly."),
        dict(id="SPLITK", severity="REMEASURE", detail="Only producer+real reducer+indexed prepare/finish may win. No estimated reducer cost."),
        dict(id="COMPONENT_SUM", severity="NOT_E2E", detail="Isolated SF/full-dequant/provider sums remain estimates, not measured model or chained latency; prefill is outside this decode sweep."),
        dict(id="MODEL_FUSION", severity="MODEL_GATE_REQUIRED", detail="Standalone indexed projection includes prepare/finish but cannot certify shared gate/up/down routing, topk fusion or whole-model speed."),
        dict(id="LEGACY_BODY", severity="MODEL_GATE_REQUIRED", detail="Recompiled typed TC parent preserves geometry, not an immutable old legacy DSO. Prior legacy output-head timing is a regression alarm, not a paired sample."),
        dict(id="TRAFFIC_MODEL", severity="MODELED_ONLY", detail="MBU uses unique active packed bytes / event time / 2700 GB/s, not hardware counters. A/metadata/repeated loads differ across readers."),
        dict(id="CACHE", severity="CONTROLLED_APPROXIMATION", detail="Rotating unique active weight addresses exceed 2.25x verified L2; activation/output remain resident; not a guarantee that every read misses L2."),
        dict(id="NUMERIC", severity="MODEL_GATE_REQUIRED", detail="Synthetic common-exact weights isolate reader correctness; real activation range, non-dyadic scales, full-model accuracy require the BF16/model gates."),
        dict(id="SEARCH", severity="BOUNDED_SEARCH", detail="Measured-pool best, not global optimum. Keep incumbents and audit omitted/runtime-rejected candidates. No universal 5% guarantee for unseen shapes."),
    ])


def seed_parents(q, mode):
    route = ("sf" if q == 8 else "fq") + ("-grouped" if mode == 2 else "-dense")
    tk = {8: 64, 10: 128, 11: 256, 12: 128, 13: 256, 14: 128}[q]
    # Six geometries, not TM/TN/TK/warp/stages Cartesian multiplication.
    for tm, tn, wn, dn in ((8, 32, 16, 16), (8, 64, 16, 32), (8, 128, 32, 32),
                          (16, 32, 16, 32), (16, 64, 16, 64), (16, 128, 32, 32)):
        p = dict(qtype=q, route=route, tm=tm, tn=tn, tk=tk, wm=tm, wn=wn,
                 stages=2, ap=0, dn=dn, persistent=0 if route == "fq-grouped" else -1,
                 symbol=f"closure_q{q}_{mode}_tm{tm}_tn{tn}_tk{tk}_wn{wn}_dn{dn}")
        validate_parent(p)
        yield p


def make_plan(inventory, computes=("f16", "bf16")):
    if not computes or set(computes) - {"f16", "bf16"}:
        raise ValueError("explicit compute scope required")
    history = historical()
    evidence = load(SOURCES[2])
    simt_history = {tuple(row["key"]): row for row in evidence["simt"]}
    table = load(SOURCES[0])
    families, controls = {}, defaultdict(list)
    for (q, mode, n, k, m, ch), values in history.items():
        key = (q, mode, n, k, 256 if mode else 1, 8 if mode else 1, ch)
        families.setdefault(key, set()).add("HISTORICAL")
        controls[q, mode, n, k].extend(c for c in values if c["kind"] == "tc")
    for row in inventory["matrices"]:
        key = tuple(row[f] for f in ("q", "mode", "n", "k", "experts", "topk", "channels"))
        families.setdefault(key, set()).add(row["model"] + ":" + row["tensor"])
    points, candidates, modules = [], {}, {}
    for family, origins in sorted(families.items()):
        q, mode, n, k, experts, topk, ch = family
        if n % 16 or k % (32 if q == 8 else 256) or (mode and experts < topk):
            raise ValueError("model matrix is outside canonical geometry: " + str(family))
        peers = [c for fk, values in controls.items() if fk[:2] == (q, mode) for c in values]
        own = controls[q, mode, n, k]
        # Missing-family fallback candidates are proposed, then measured here.
        for compute in computes:
            for m in range(1, 9):
                profiles = ("dense",) if not mode else (("spread", "real") if m < 8 else ("spread", "real", "cluster", "repeat2", "repeat4"))
                for router in profiles:
                    h = history.get((q, mode, n, k, m, ch), [])
                    point = dict(q=q, mode=mode, n=n, k=k, experts=experts, topk=topk, channels=ch,
                                 tokens=m, compute=compute, router=router, origins=sorted(origins))
                    pool, mandatory, pruned = {}, set(), []
                    def add(c, required=False):
                        if compute == "bf16" and (c["kind"] == "q4" or (c["kind"] == "tc" and c["parent"]["ap"])):
                            pruned.append(dict(candidate=c, reason="F16_ONLY_BODY_NOT_BF16_CANDIDATE"))
                            return
                        cid = candidate_id(c)
                        pool[cid] = c
                        if required:
                            mandatory.add(cid)
                    for c in h:
                        add(c, True)
                    # Reproduce the current nearest-log-bucket choice too:
                    # otherwise the output-head regression can disappear
                    # from the pool merely because there is no exact row.
                    if not h and q != 12:
                        scored = []
                        mb = 0 if m == 1 else (m - 1).bit_length()
                        for bucket in table["buckets"]:
                            bq, bm, bch, bt, bn, bk = bucket["key"]
                            if (bq, bm, bch) != (q, mode, ch):
                                continue
                            row = table["exact"][bucket["row"]]
                            c = row["config"]
                            if c["kind"] == "tc" and (k % (c["tk"] * c["split"]) or
                                    k // (c["tk"] * c["split"]) < c["stages"] - 1 or (c["ap"] and (mode or m != 1))):
                                continue
                            scored.append((4 * abs(mb - bt) + abs(n.bit_length() - 1 - bn) + abs(k.bit_length() - 1 - bk), bucket["row"]))
                        if scored:
                            # Stable order is the C++ bucket-table tie break.
                            best = min(scored, key=lambda item: item[0])
                            c = table["exact"][best[1]]["config"]
                            value = tc(c, "PRODUCTION_BUCKET_INCUMBENT") if c["kind"] == "tc" else dict(
                                kind="simt", recipe={f: c[f] for f in ("variant", "columns", "warps", "values", "split")},
                                split=c["split"], reason="PRODUCTION_BUCKET_INCUMBENT")
                            add(value, True)
                            point["production_bucket_distance"] = best[0]
                    # Extract candidates from measured winners. Timings only
                    # rank proposals here, never admit cross-cohort winners.
                    nearby = sorted((abs(math.log2(n / key[2])) + abs(math.log2(k / key[3])) +
                                     .25 * abs(math.log2(m / key[4])), key, row)
                                    for key, row in simt_history.items() if key[:2] == (q, mode) and key[5] == ch)
                    geos = []
                    for _, key, _ in nearby:
                        for c in history[key]:
                            if c["kind"] == "simt":
                                geo = {f: c["recipe"][f] for f in ("variant", "columns", "warps", "values")}
                                if geo not in geos:
                                    geos.append(geo)
                        if len(geos) >= 2:
                            break
                    if not geos:
                        geos = [dict(variant=1 if q == 8 else 3, columns=4, warps=4, values=4)]
                    # Two measured geometries + two single-axis variants.
                    first = geos[0]
                    for w in (2, 8):
                        neighbor = first | dict(warps=w)
                        if neighbor not in geos:
                            geos.append(neighbor)
                    allowed = {tuple(asdict(c).values()) for c in runtime_inventory(q)}
                    for geo in geos:
                        for split in (1, 2, 4, 8):
                            recipe = geo | dict(split=split)
                            if tuple(recipe.values()) in allowed and n % (recipe["columns"] * recipe["values"]) == 0:
                                add(dict(kind="simt", recipe=recipe, split=split, reason="MEASURED_GEOMETRY_AND_BOUNDED_NEIGHBORS"))
                    # Retain exact incumbents without a cap. Additional TC
                    # seeds are just two family geometries + two safe seeds.
                    anchors = own or peers
                    parent_pool = {}
                    for c in sorted(anchors, key=lambda c: (abs(c["parent"]["tm"] - min(16, max(8, m))), c["parent"]["symbol"])):
                        parent_pool.setdefault(digest(c["parent"]), c["parent"])
                        if len(parent_pool) >= 2:
                            break
                    seeds = list(seed_parents(q, mode))
                    for p in (seeds[1], seeds[4]):
                        parent_pool[digest(p)] = p
                    for p in parent_pool.values():
                        for split in (1, 2, 4, 8):
                            c = dict(kind="tc", parent=p, split=split, algorithm=int(p["persistent"] == 1),
                                     grid_b=1 if p["persistent"] == 1 else 0, grid_mode=2 if p["persistent"] == 1 else 0,
                                     reason="BOUNDED_TC_POOL")
                            if (p["ap"] and (mode or m != 1)) or k % (p["tk"] * split) or k // (p["tk"] * split) < p["stages"] - 1:
                                pruned.append(dict(candidate=c, reason="PROVEN_M_OR_K_PIPELINE_CONSTRAINT"))
                            else:
                                add(c)
                    point["id"] = digest(point)[:24]
                    point.update(candidates=sorted(pool), mandatory=sorted(mandatory), pruned=pruned)
                    points.append(point)
                    candidates.update(pool)
                    for c in pool.values():
                        if c["kind"] == "tc":
                            modules[digest((compute, c["parent"]))] = dict(compute=compute, parent=c["parent"])
    sources = {s: sha(ROOT / s) for s in SOURCES}
    result = dict(schema=SCHEMA, scope="TP1_DENSE_AND_INDEXED_TOKENS_1_TO_8_F32_IO",
                  inventory=inventory, points=points, candidates=candidates, modules=modules,
                  sources=sources, computes=list(computes), evidence_audit=evidence_audit(),
                  protocol=dict(screen=3, rounds=4, samples=11, confirm_per_family=2,
                                threshold_pct=5, ring_l2_multiplier=2.25,
                                timer="GRAPH_COMPLETE_CALL_EXCLUDES_SETUP_JIT_AND_FIRST_UPLOAD"),
                  search="HISTORICAL_INCUMBENTS_PLUS_TWO_NEARBY_GEOMETRIES_AND_BOUNDED_NEIGHBORS_NOT_ALL_CONFIGS")
    result["plan_sha256"] = digest(result)
    return result


def validate(plan):
    if plan.get("schema") != SCHEMA or plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("frozen plan identity differs")
    if len({p["id"] for p in plan["points"]}) != len(plan["points"]):
        raise ValueError("duplicate workload")
    for point in plan["points"]:
        if not set(point["mandatory"]) <= set(point["candidates"]):
            raise ValueError("historical incumbent missing")
        pool = [plan["candidates"][c] for c in point["candidates"]]
        if not {"simt", "tc"} <= {c["kind"] for c in pool}:
            raise ValueError("missing SIMT/TC comparison")
    return plan
