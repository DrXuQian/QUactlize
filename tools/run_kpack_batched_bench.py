#!/usr/bin/env python3
"""Paired llama-batched-bench model timings, with same-process shape warmup.

No server, HTTP, profiler, correctness sweep or model-byte hashing. The first
complete PP/TG pass for each PP is excluded; repeated -npp values intentionally
exercise the existing benchmark without changing its binary or GEMM modules.
"""

import argparse
from collections import Counter
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.gguf_roles import match_role
from quactlize.runtime.compiler import sha
from tools.gguf_internal_shape_inventory import read_gguf_header
from tools.resolve_kpack_batched_models import resolve_plan
from tools.verify_kpack_dispatch import verify


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_plan(plan):
    for key in ("prompts", "generations"):
        values = plan[key]
        if not values or len(values) != len(set(values)) or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("invalid workload axis: " + key)
    if plan["parallel"] != 1 or min(plan["batch"], plan["ubatch"]) <= 0:
        raise ValueError("this protocol requires NPL=1 and positive token batches")
    names = set()
    for m in plan["models"]:
        if not re.fullmatch(r"[a-z0-9-]+", m["name"]) or m["name"] in names:
            raise ValueError("invalid/duplicate model name")
        names.add(m["name"])
        if not re.fullmatch(r"\d+(,\d+)*", m["devices"]) or m["split"] not in ("none", "tensor"):
            raise ValueError("invalid model device topology")
        devices = m["devices"].split(",")
        if len(devices) != len(set(devices)) or (m["split"] == "none" and len(devices) != 1):
            raise ValueError("invalid model device set")
    return plan


def inventory(path):
    with path.open("rb") as stream:
        header = read_gguf_header(stream, str(path))
    eligible, omitted, q8 = [], [], []
    for t in header["tensors"]:
        dims, q = t["dims_gguf"], t["qtype"]
        role = match_role(t["name"], len(dims)) if q in (8,10,11,12,13,14) else None
        if role and role[0].route_class in ("dense", "grouped") and dims[0] % (512 if q in (11,14) else 256) == 0 and dims[1] % 256 == 0:
            eligible.append(t["name"])
            if q == 8: q8.append(t["name"])
        else:
            omitted.append(dict(name=t["name"], qtype=q))
    return dict(eligible=eligible, q8=q8, omitted=omitted,
                qtypes=dict(Counter(str(t["qtype"]) for t in header["tensors"])),
                shards=header["metadata"].get("split.count", 1))


def sequence(plan, repeats):
    return [(pp, tg, run) for pp in plan["prompts"]
            for run in range(repeats + 1) for tg in plan["generations"]]


def command(binary, model, plan, repeats, names, arm, cache):
    prompts = [pp for pp in plan["prompts"] for _ in range(repeats + 1)]
    argv = [str(binary), "-m", model["path"], "-npp", ",".join(map(str, prompts)),
            "-ntg", ",".join(map(str, plan["generations"])), "-npl", "1", "-ngl", "all",
            "-b", str(plan["batch"]), "-ub", str(plan["ubatch"]),
            "-c", str(max(plan["prompts"]) + max(plan["generations"])),
            "-fa", "1", "-sm", model["split"], "--fit", "off", "--mmap",
            "--output-format", "jsonl", "--log-colors", "off"]
    if model["split"] == "tensor":
        if arm != "reference":
            raise ValueError("K-pack tensor-parallel intake is not admitted")
        argv += ["-ts", model["tensor_split"]]
    elif names:
        # Exact names AND supported qtypes: never catch a Q8/F16 tensor merely
        # because its name ends in output.weight or matches another suffix.
        pattern = "^(" + "|".join(re.escape(n) for n in names) + ")$"
        argv += ["-ot", pattern + "=CUDA0" + ("_KPACK" if arm == "kpack" else "")]
    if arm == "kpack":
        argv += ["--kpack-cache", str(cache)]
    return argv


def parse_row(line, expected, plan):
    begin = line.find('{"n_kv_max"')
    if begin < 0:
        return None
    row = json.loads(line[begin:])
    pp, tg, run = expected
    for name, want in dict(pp=pp, tg=tg, pl=1, n_batch=plan["batch"],
                           n_ubatch=plan["ubatch"], flash_attn=1, is_pp_shared=0,
                           n_kv=pp+tg).items():
        if row.get(name) != want:
            raise ValueError(f"benchmark {name}={row.get(name)!r}, expected {want}")
    if row["n_kv_max"] < max(plan["prompts"]) + max(plan["generations"]):
        raise ValueError("context cannot cover the requested matrix")
    for name in ("t_pp", "t_tg", "speed_pp", "speed_tg"):
        if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]) or row[name] <= 0:
            raise ValueError("invalid benchmark timer: " + name)
    # The upstream JSON rounds seconds to six decimals and t/s to float32.
    if not math.isclose(row["t_pp"] * row["speed_pp"], pp, rel_tol=2e-4, abs_tol=1e-3):
        raise ValueError("prefill timer/token denominator differs")
    if not math.isclose(row["t_tg"] * row["speed_tg"], tg, rel_tol=2e-4, abs_tol=1e-3):
        raise ValueError("decode timer/token denominator differs")
    return row | dict(phase="warmup" if run == 0 else "measured", repeat=run,
                      prefill_us_per_token=1e6*row["t_pp"]/pp,
                      decode_us_per_token=1e6*row["t_tg"]/tg)


def run_arm(args, model, plan, inv, arm, directory, index):
    label = f"{index}-{arm}"
    log = directory / (label + ".log")
    cache = args.cache_root / model["name"]
    if arm == "kpack":
        cache.parent.mkdir(parents=True, exist_ok=True)
    argv = command(args.binary, model, plan, args.repeats, inv["eligible"], arm, cache)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    for k in ("GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION"):
        env.pop(k, None)
    env["CUDA_VISIBLE_DEVICES"] = model["devices"]
    if arm == "reference":
        for k in list(env):
            if k.startswith("QUACTLIZE_KPACK_"):
                env.pop(k)
    else:
        env["QUACTLIZE_KPACK_ROUTE"] = "auto"
        env["QUACTLIZE_KPACK_EXECUTION"] = str(args.bundle)
        env["QUACTLIZE_KPACK_JIT_HELPER"] = str(ROOT / "tools/kpack_jit.py")
        env["QUACTLIZE_KPACK_JIT_PYTHON"] = sys.executable
        env["QUACTLIZE_KPACK_JIT_CACHE"] = str(args.jit_cache)
        env["QUACTLIZE_KPACK_PAIR_WEIGHTS"] = "1"
    save(directory / (label + ".command.json"), dict(argv=argv, devices=model["devices"]))
    expected, records = sequence(plan, args.repeats), []
    started = update = time.monotonic()
    print(f"BATCHED_MODEL_START model={model['name']} arm={label} log={log}", flush=True)
    # A growing regular file avoids a profiler, HTTP overhead, or pipe
    # backpressure on the application. Polling is outside its timed regions.
    with log.open("x") as output, log.open() as reader:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        try:
            pending = ""
            while True:
                pending += reader.read()
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    if '{"n_kv_max"' not in line:
                        continue
                    if len(records) >= len(expected):
                        raise ValueError("extra/duplicate benchmark row")
                    row = parse_row(line, expected[len(records)], plan)
                    records.append(row)
                    print(f"BATCHED_MODEL_PROGRESS model={model['name']} arm={label} "
                          f"phase={row['phase']} pp={row['pp']} tg={row['tg']} "
                          f"completed={len(records)}/{len(expected)} "
                          f"prefill_us={row['prefill_us_per_token']:.3f} decode_us={row['decode_us_per_token']:.3f}", flush=True)
                rc = proc.poll()
                if rc is not None:
                    # Drain once after observing exit, including a partial
                    # last line, before checking the complete denominator.
                    rest = pending + reader.read()
                    for line in rest.splitlines():
                        if '{"n_kv_max"' in line:
                            if len(records) >= len(expected):
                                raise ValueError("extra benchmark row after process exit")
                            records.append(parse_row(line, expected[len(records)], plan))
                    if rc != 0 or len(records) != len(expected):
                        raise ValueError(f"process rc={rc}, rows={len(records)}/{len(expected)}; see {log}")
                    break
                if time.monotonic() - update >= 30:
                    print(f"BATCHED_MODEL_WAIT model={model['name']} arm={label} rows={len(records)}/{len(expected)} "
                          f"elapsed_minutes={(time.monotonic()-started)/60:.1f} log={log}", flush=True)
                    update = time.monotonic()
                time.sleep(1)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            save(directory / (label + ".timings.json"), records)
            save(directory / (label + ".process.json"), dict(rc=proc.returncode, wall_seconds=time.monotonic()-started))
    text = log.read_text(errors="replace")
    if arm == "kpack":
        sys.path.insert(0, str(args.llama_dir / "tests"))
        from quactlize_native import model_selection
        evidence = model_selection(SimpleNamespace(
            manifest=args.manifest, bundle=args.bundle, jit_cache=args.jit_cache,
            jit_helper=ROOT / "tools/kpack_jit.py", jit_python=Path(sys.executable)), text)
        if not evidence["plans"] and not evidence["fallbacks"]:
            raise ValueError("no selected or legacy K-pack compute plan; not a K-pack timing")
        q8_plans=[p for p in evidence["plans"] if p.get("q")=="8"]
        if inv["q8"] and (not q8_plans or any(p.get("activation")!="FP16" or
                p.get("route")!="sf" or p.get("scale_resident")!="1" for p in q8_plans)):
            raise ValueError("Q8_0 W8A16 compute evidence missing or activation/scale contract differs")
        evidence["q8_w8a16_plans"] = len(q8_plans)
        evidence["moe_chains"] = [dict(re.findall(r"([a-z_]+)=([^\s]+)",line))
            for line in text.splitlines() if "[quactlize-moe]" in line]
        evidence["paired_weights"] = [line for line in text.splitlines() if "[kpack-pair]" in line]
        evidence["cache"] = [line for line in text.splitlines() if "[kpack-cache]" in line]
        save(directory / (label + ".selection.json"), evidence)
        coverage = "PARTIAL_NATIVE" if evidence["fallbacks"] else "SELECTED_PLANS"
    else:
        if "[quactlize-plan]" in text or "native policy miss" in text:
            raise ValueError("reference unexpectedly selected K-pack compute")
        coverage = "REFERENCE"
    return dict(arm=arm, label=label, coverage=coverage, records=records)


def summarize(plan, runs, model):
    result = []
    for pp, tg in itertools.product(plan["prompts"], plan["generations"]):
        row = dict(model=model["name"], pp=pp, tg=tg, npl=1, devices=model["devices"])
        for arm in ("reference", "kpack"):
            values = [r for run in runs if run["arm"] == arm for r in run["records"]
                      if r["phase"] == "measured" and (r["pp"], r["tg"]) == (pp, tg)]
            row[arm + "_samples"] = len(values)
            for metric in ("prefill", "decode"):
                row[arm + "_" + metric + "_us"] = statistics.median(
                    v[metric + "_us_per_token"] for v in values) if values else None
        for metric in ("prefill", "decode"):
            ref, cur = row["reference_" + metric + "_us"], row["kpack_" + metric + "_us"]
            row[metric + "_delta_pct"] = (cur / ref - 1)*100 if ref and cur else None
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=ROOT / "tools/kpack_batched_models.json")
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--llama-dir", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--jit-cache", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("/workspace"))
    p.add_argument("--output",type=Path,help="fresh explicit run directory")
    p.add_argument("--repeats", type=int, default=1, help="measured passes after one full excluded pass per PP")
    p.add_argument("--model", action="append", help="select model name(s); default is all five")
    p.add_argument("--model-root", type=Path, help="override the plan's model root")
    p.add_argument("--device",help="override the ordinal for single-device models only")
    a = p.parse_args()
    plan = validate_plan(json.loads(a.plan.read_text()))
    if a.device is not None:
        if not re.fullmatch(r"\d+",a.device):p.error('one device ordinal required')
        for model in plan['models']:
            if model['split']=='none':model['devices']=a.device
    if a.repeats < 1 or not a.output_root.is_dir() or not os.access(a.binary, os.X_OK):
        p.error("positive repeats, executable benchmark and existing output root required")
    try:
        plan = resolve_plan(plan, a.model, a.model_root)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    a.manifest = verify(a.bundle)
    a.bundle = a.bundle.resolve(strict=True)
    a.jit_cache = a.jit_cache.resolve()
    a.jit_cache.mkdir(parents=True, exist_ok=True)
    for key in ("PPU_SDK", "QUACTLIZE_PPU_BUNDLE", "QUACTLIZE_PPU_PACK_LIBRARY"):
        if not os.environ.get(key) or not Path(os.environ[key]).exists():
            p.error("missing deployment input: " + key)
    a.binary, a.llama_dir = a.binary.resolve(strict=True), a.llama_dir.resolve(strict=True)
    out = a.output.resolve() if a.output else Path(tempfile.mkdtemp(prefix="kpack-batched.", dir=a.output_root))
    if a.output: out.mkdir(parents=True,exist_ok=False)
    results = out / "results"
    results.mkdir()
    save(results / "protocol.json", dict(plan=plan, repeats=a.repeats, first_pass_excluded=True,
        binary=str(a.binary), binary_sha256=sha(a.binary), policy_mode="auto",
        scope="SYNTHETIC_RANDOM_TOKENS_LLAMA_BATCHED_BENCH_NO_PROFILER",
        production_fusion="MOE_CHAIN_AND_GPU_GATE_UP_PAIR", kernel_execution_evidence="NOT_COLLECTED"))
    print(f"BATCHED_RESULTS run={out}", flush=True)
    boards, statuses = [], []
    try:
        for model in plan["models"]:
            directory = results / model["name"]
            directory.mkdir()
            runs = []
            try:
                inv = inventory(Path(model["path"]))
                save(directory / "inventory.json", inv)
                reason = ("KPACK_TP_NOT_ADMITTED" if model["split"] == "tensor" else
                          "NO_SUPPORTED_MATRICES" if not inv["eligible"] else None)
                arms = [] if reason else ["reference", "kpack", "kpack"]
                if reason:
                    statuses.append(dict(model=model["name"], arm="kpack", status="NOT_TESTED", reason=reason))
                    print(f"BATCHED_MODEL_SCOPE model={model['name']} kpack=NOT_TESTED reason={reason}", flush=True)
                for index, arm in enumerate(arms):
                    try:
                        run = run_arm(a, model, plan, inv, arm, directory, index)
                        runs.append(run)
                        statuses.append(dict(model=model["name"], arm=arm, status="PASS", coverage=run["coverage"]))
                    except Exception as e:
                        statuses.append(dict(model=model["name"], arm=arm, status="FAIL", reason=str(e)))
                        print(f"BATCHED_MODEL_FAIL model={model['name']} arm={arm} error={e} remaining_continue=1", flush=True)
            except Exception as e:
                statuses.append(dict(model=model["name"], status="FAIL", reason=str(e)))
                print(f"BATCHED_MODEL_FAIL model={model['name']} error={e} remaining_continue=1", flush=True)
            boards.extend(summarize(plan, runs, model))
            save(results / "status.json", statuses)
            save(results / "summary.json", boards)
            with (results / "summary.tsv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(boards[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(boards)
    finally:
        with tarfile.open(str(out) + ".results.tgz", "w:gz") as archive:
            archive.add(results, arcname="results")
        print(f"results={out}.results.tgz", flush=True)
    return int(any(s["status"] == "FAIL" for s in statuses))


if __name__ == "__main__":
    sys.exit(main())
