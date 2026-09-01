#!/usr/bin/env python3
"""Fail-closed validation and summary for the Q4 K-pack policy-v2 pilot."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys
import tempfile

from plan_fq_kquant_policy_v2 import MAPPING_ID, SCHEMA, materialize, validate


RESULT_SCHEMA = "quactlize.fq-kquant-kpack-policy-result.v2"


class AnalysisError(ValueError): pass


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line.strip()))


def parse_samples(text: str, iterations: int) -> list[float]:
    if not (text.startswith("[") and text.endswith("]")):
        raise AnalysisError("sample vector syntax differs")
    values = [float(x) for x in text[1:-1].split(",") if x]
    if len(values) != iterations:
        raise AnalysisError(f"sample count differs: {len(values)} != {iterations}")
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise AnalysisError("samples must be finite and positive")
    if values != sorted(values):
        raise AnalysisError("samples are not sorted")
    return values


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def analyze(plan_path: pathlib.Path, runs: pathlib.Path, output: pathlib.Path,
            rounds: int, iterations: int, warmups: int) -> dict:
    plan = json.loads(plan_path.read_text()); validate(plan)
    logs = sorted(runs.glob("q12-round*.log"))
    if len(logs) != rounds: raise AnalysisError(f"expected {rounds} logs, got {len(logs)}")
    by_m: dict[int, dict[str, list[float]]] = {m: {} for m in plan["m_values"]}
    expected_candidates = set(plan["candidate_names"])
    expected_rounds = set(range(1, rounds + 1))
    observed_rounds: set[int] = set()
    for log in logs:
        text = log.read_text()
        markers = [fields(x) for x in text.splitlines() if x.startswith("FQ_KQUANT_POLICY_RUN ")]
        if len(markers) != 1: raise AnalysisError(f"{log.name}: completion marker count differs")
        marker = markers[0]
        required = {"schema": "kpack-policy-v2", "q": "12", "layout": "kpack",
                    "order": "kpack-first",
                    "iterations": str(iterations), "all_configs": "1",
                    "warmups": str(warmups),
                    "dense_cases": "64", "grouped_cases": "0", "status": "PASS"}
        if any(marker.get(k) != v for k, v in required.items()):
            raise AnalysisError(f"{log.name}: completion identity differs")
        round_id = int(marker.get("round", "0"))
        if round_id not in expected_rounds or round_id in observed_rounds or \
           log.name != f"q12-round{round_id}.log":
            raise AnalysisError(f"{log.name}: round identity differs")
        observed_rounds.add(round_id)
        rows = [fields(x) for x in text.splitlines() if x.startswith("FQ_KQUANT_LAYOUT_DENSE ")]
        seen: set[tuple[int, str]] = set()
        for row in rows:
            if row.get("q") != "12" or row.get("layout") != "kpack" or \
               row.get("order") != "kpack-first" or \
               row.get("mapping_id") != MAPPING_ID or row.get("raw_bad") != "0" or \
               row.get("iterations") != str(iterations) or \
               row.get("provider") != "standard-aiu":
                raise AnalysisError(f"{log.name}: dense row identity/correctness differs")
            shape = tuple(map(int, row["shape"].split("x")))
            m, n, k = shape
            if m not in by_m or (n, k) != (plan["family"]["n"], plan["family"]["k"]):
                raise AnalysisError(f"{log.name}: shape outside plan")
            if int(row.get("round", "0")) != round_id:
                raise AnalysisError(f"{log.name}: row round differs")
            config = row["config"]
            if config not in expected_candidates:
                raise AnalysisError(f"{log.name}: unknown candidate {config}")
            key = (m, config)
            if key in seen: raise AnalysisError(f"{log.name}: duplicate candidate {key}")
            samples = parse_samples(row["samples"], iterations)
            median = statistics.median(samples)
            if not close(float(row["median_us"]), median) or \
               not close(float(row["min_us"]), samples[0]) or \
               not close(float(row["max_us"]), samples[-1]):
                raise AnalysisError(f"{log.name}: reported timing statistics differ")
            seen.add(key); by_m[m].setdefault(config, []).extend(samples)
        if not rows: raise AnalysisError(f"{log.name}: no dense rows")
        for m in plan["m_values"]:
            found = {config for seen_m, config in seen if seen_m == m}
            if found != expected_candidates:
                raise AnalysisError(f"{log.name}: M={m} candidate set differs")
    if observed_rounds != expected_rounds:
        raise AnalysisError("round denominator differs")
    candidates = tuple(plan["candidate_names"])
    for m, rows in by_m.items():
        if set(rows) != expected_candidates or \
           any(len(values) != rounds * iterations for values in rows.values()):
            raise AnalysisError(f"M={m}: aggregate sample coverage differs")
    summary_rows = []
    for m, rows in by_m.items():
        ranked = sorted((statistics.median(values), name) for name, values in rows.items())
        summary_rows.append({"m": m, "winner": ranked[0][1], "median_us": ranked[0][0],
                             "runner": ranked[1][1] if len(ranked) > 1 else None,
                             "runner_us": ranked[1][0] if len(ranked) > 1 else None,
                             "candidates": len(ranked),
                             "boundary_control": m in (1, 8, 9, 64)})
    result = {"schema": RESULT_SCHEMA, "plan_schema": SCHEMA,
              "profile": "kpack-policy-v2", "qtype": 12,
              "layout": "q4-kpack4", "mapping_id": MAPPING_ID,
              "rounds": rounds, "iterations": iterations,
              "candidate_names": list(candidates), "rows": summary_rows,
              "verdict": "PILOT_COMPLETE"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "summary.tsv").open("w") as stream:
        stream.write("M\twinner\tmedian_us\trunner\trunner_us\tcandidates\tboundary_control\n")
        for row in summary_rows:
            stream.write(f"{row['m']}\t{row['winner']}\t{row['median_us']:.9f}\t"
                         f"{row['runner'] or 'NONE'}\t{row['runner_us'] or 'NONE'}\t"
                         f"{row['candidates']}\t{int(row['boundary_control'])}\n")
    print(f"FQ_KQUANT_POLICY_V2 verdict=PILOT_COMPLETE shapes=64 candidates={len(candidates)}")
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary); runs = root / "runs"; runs.mkdir()
        plan = root / "plan.json"; plan.write_text(json.dumps(materialize()))
        for round_id in (1, 2):
            lines = []
            for m in range(1, 65):
                for index, config in enumerate(materialize()["candidate_names"]):
                    timing = 9 + index + m/100
                    samples = [timing, timing + .1, timing + .2]
                    lines.append(
                        f"FQ_KQUANT_LAYOUT_DENSE q=12 round={round_id} order=kpack-first "
                        f"layout=kpack mapping_id={MAPPING_ID} shape={m}x1024x5120 "
                        f"config={config} provider=standard-aiu iterations=3 raw_bad=0 "
                        f"median_us={samples[1]} min_us={samples[0]} max_us={samples[2]} "
                        f"samples=[{','.join(map(str,samples))}]")
            lines.append(
                f"FQ_KQUANT_POLICY_RUN schema=kpack-policy-v2 q=12 round={round_id} "
                "layout=kpack order=kpack-first iterations=3 warmups=1 all_configs=1 dense_cases=64 "
                "grouped_cases=0 status=PASS")
            (runs / f"q12-round{round_id}.log").write_text("\n".join(lines)+"\n")
        result = analyze(plan, runs, root / "out", 2, 3, 1)
        if len(result["rows"]) != 64 or result["rows"][-1]["m"] != 64:
            raise AssertionError("synthetic denominator differs")
        poisoned = runs / "q12-round2.log"
        poisoned.write_text(poisoned.read_text().replace(MAPPING_ID, "0x0", 1))
        try: analyze(plan, runs, root / "red", 2, 3, 1)
        except AnalysisError: pass
        else: raise AssertionError("wrong-map negative stayed green")
        # Independent timing and marker negatives cover the raw-sample authority.
        original = (runs / "q12-round2.log").read_text().replace("mapping_id=0x0", f"mapping_id={MAPPING_ID}", 1)
        plants = (
            original.replace("samples=[9.01,9.11,9.209999999999999]", "samples=[9.11,9.01,9.209999999999999]", 1),
            original.replace("samples=[9.01,9.11,9.209999999999999]", "samples=[9.01,9.11]", 1),
            original.replace("samples=[9.01,9.11,9.209999999999999]", "samples=[9.01,nan,9.209999999999999]", 1),
            original.replace("warmups=1", "warmups=2", 1),
            original.replace("median_us=9.11", "median_us=99", 1),
            original.replace("round=2", "round=1", 1),
            original.replace("order=kpack-first", "order=xplane-first", 1),
            original.replace("provider=standard-aiu", "provider=packed-row", 1),
            original.replace("kpack4:8x32x256:8x16:s3:S4", "unknown-config", 1),
            original.replace("kpack4:8x64x256:8x16:s2:S4",
                             "kpack4:8x32x256:8x16:s3:S4", 1),
        )
        for index, broken in enumerate(plants):
            (runs / "q12-round2.log").write_text(broken)
            try: analyze(plan, runs, root / f"red{index}", 2, 3, 1)
            except AnalysisError: pass
            else: raise AssertionError(f"timing/marker negative {index} stayed green")
    print("[fq-kquant-policy-v2-analysis:self-test] PASS exact schema/layout/mapping, "
          "64-shape denominator, raw samples/statistics and round coverage; "
          "wrong-map/order/provider/unknown/duplicate/timing plants RED")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("self-test")
    run=sub.add_parser("analyze"); run.add_argument("--plan",type=pathlib.Path,required=True); run.add_argument("--runs",type=pathlib.Path,required=True); run.add_argument("--output",type=pathlib.Path,required=True); run.add_argument("--rounds",type=int,required=True); run.add_argument("--iterations",type=int,required=True); run.add_argument("--warmups",type=int,required=True)
    args=parser.parse_args()
    try:
        if args.cmd=="self-test": self_test()
        else: analyze(args.plan,args.runs,args.output,args.rounds,args.iterations,args.warmups)
        return 0
    except (AnalysisError,AssertionError,OSError,ValueError) as error:
        print(f"[fq-kquant-policy-v2-analysis] FAIL: {error}",file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())
