from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kpack_tuning_plan as plan
import build_kpack_tuner as build
import run_kpack_tuner as run
import replay_kpack_tuning_history as history


def test_generator_membership_and_soft_budget():
    source = plan.candidates(12, "fq-dense")
    p = {"m": 1, "n": 1024, "k": 5120}
    selected, reasons = plan.choose(source, p, 32, 72)
    assert set(selected) <= set(source)
    assert len(selected) == len(reasons)
    anchors = {
        c
        for c in source
        if c.geometry in plan.historical_geometries(12, "fq-dense")
        and plan.admissible(c, p)
    }
    assert anchors <= set(selected)
    # A budget never ejects a known winner, even if its analytic rank is poor.
    winner = source[-1]
    while not plan.admissible(winner, p):
        source = source[:-1]
        winner = source[-1]
    assert winner in plan.choose(source, p, 4, 72, {winner.symbol})[0]


def test_missing_or_inadmissible_historical_winner_is_red():
    source = plan.candidates(12, "fq-dense")
    with pytest.raises(ValueError, match="missing"):
        plan.choose(source, {"m": 1, "n": 1024, "k": 5120}, 32, 72, {"missing"})
    ap1 = next(c for c in source if c.ap == 1)
    with pytest.raises(ValueError, match="admissible"):
        plan.choose(source, {"m": 2, "n": 1024, "k": 5120}, 32, 72, {ap1.symbol})


def test_decode_history_and_grouped_m8_not_lost():
    m8 = next(c for c in plan.candidates(12, "fq-dense") if c.tm == 8 and c.ap == 0)
    assert plan.admissible(m8, {"m": 8, "n": 1024, "k": 5120})
    g8 = next(c for c in plan.candidates(12, "fq-grouped") if c.tm == 8)
    assert plan.admissible(
        g8, {"total_rows": 528, "max_rows": 129, "experts": 256, "n": 3072, "k": 512}
    )


@pytest.mark.parametrize(
    "q,geometries",
    (
        (12, {(32, 128, 256, 32, 32, 3)}),
        (13, {(32, 128, 256, 32, 32, 3)}),
        (
            14,
            {
                (16, 128, 128, 16, 32, 3),
                (32, 128, 128, 32, 32, 3),
                (128, 64, 128, 64, 32, 2),
            },
        ),
    ),
)
def test_measured_grouped_winners_beyond_default_are_retained(q, geometries):
    p = {"total_rows": 16384, "max_rows": 239, "experts": 256, "n": 512, "k": 2048}
    selected, reasons = plan.choose(plan.candidates(q, "fq-grouped"), p, 4, 72)
    assert geometries <= {c.geometry for c in selected}
    for geometry in geometries:
        replay = [c for c in selected if c.geometry == geometry]
        assert {c.persistent for c in replay} == {0, 1}
        assert all(reasons[c.symbol] == "historical-geometry" for c in replay)


def test_module_emits_original_function_wrappers():
    for route in plan.ROUTES:
        row = plan.candidates(12, route)[0]
        source = build.module_source([row], "test-contract")
        assert build.SPEC[route][2] in source
        assert "kpack_tuner_module_v1" in source
        assert row.symbol in source
        assert "sizeof(rows[0])" in source
        assert "mma.sync" not in source


def test_historical_summary_replay_and_missing_winner():
    c = next(
        c
        for c in plan.candidates(12, "fq-grouped")
        if c.geometry == (32, 128, 256, 32, 32, 3)
    )
    p = {
        "schema": plan.SCHEMA,
        "candidates": [plan.asdict(c)],
        "requests": [
            {
                "qtype": 12,
                "route": "fq-grouped",
                "workload_key": "g",
                "symbols": [c.symbol],
            }
        ],
    }
    summary = {
        "rows": [
            {
                "qtype": 12,
                "operator": "grouped",
                "key": "g",
                "candidates": {
                    "kpack": [
                        {"config": "16x128:16x16:s2", "median_us": 110.0},
                        {"config": "32x128:32x32:s3", "median_us": 100.0},
                    ]
                },
            }
        ]
    }
    assert history.replay(p, summary)["winner_geometry_covered"] == 1
    p["requests"][0]["symbols"] = []
    report = history.replay(p, summary)
    assert report["winner_geometry_covered"] == 0
    assert report["missing"][0]["reason"] == "MISSING_GEOMETRY"
    p["requests"] = []
    assert history.replay(p, summary)["missing"][0]["reason"] == "MISSING_WORKLOAD"
    summary["rows"][0]["candidates"]["kpack"][0]["median_us"] = float("nan")
    with pytest.raises(ValueError, match="timings"):
        history.replay(p, summary)


def request(route="fq-dense"):
    return {
        "qtype": 12,
        "route": route,
        "problem": {"m": 1, "n": 1024, "k": 5120},
        "workload_key": "dense_m1_n1024_k5120",
        "symbols": ["row"],
    }


def fq_log(symbol="row", samples=(10.0, 11.0)):
    lines = []
    for split in (1, 2, 4, 8):
        lines.append(
            f'FQ_TC_CELL q=12 shape=1x1024x5120 symbol={symbol} S={split} scope=FULL_OUTPUT state=MEASURED raw_bad=0 samples={json.dumps(samples,separators=(",",":"))}'
        )
    return "\n".join(lines) + "\nFQ_SHAPE_DONE status=PASS\n"


def test_fq_full_e2e_parser():
    cells = run.parse_cells(fq_log(), request(), {"row"}, 2)
    assert len(cells) == 4 and all(c["median_us"] == 10.5 for c in cells)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda x: x.replace("FULL_OUTPUT", "PRODUCER_ONLY"),
        lambda x: x.replace("raw_bad=0", "raw_bad=1", 1),
        lambda x: x.replace("q=12", "q=11", 1),
        lambda x: x.replace("1x1024x5120", "2x1024x5120", 1),
        lambda x: x.replace("[10.0,11.0]", "[10.0]", 1),
        lambda x: x.replace("[10.0,11.0]", "[NaN,11.0]", 1),
        lambda x: x.replace("FQ_SHAPE_DONE status=PASS", ""),
        lambda x: x.replace("S=8", "S=4"),
    ),
)
def test_parser_negatives(mutation):
    with pytest.raises(ValueError):
        run.parse_cells(mutation(fq_log()), request(), {"row"}, 2)


def test_scalefirst_complete_not_pass():
    lines = []
    for algorithm in ("NONPERSISTENT", "PERSISTENT"):
        for sample in (0, 1):
            lines.append(
                "SF_CELL "
                + json.dumps(
                    {
                        "shape": "1x1024x5120",
                        "qtype": 12,
                        "symbol": "row",
                        "status": "MEASURED",
                        "algorithm": algorithm,
                        "split": 1,
                        "grid": 72,
                        "sample": sample,
                        "sample_us": 10.0 + sample,
                        "metric_scope": "FULL_OUTPUT",
                        "raw_bad": 0,
                    }
                )
            )
    text = "\n".join(lines) + "\nSF_COMPLETE status=COMPLETE roundtrip=PASS\n"
    assert len(run.parse_cells(text, request("sf-dense"), {"row"}, 2)) == 2
    with pytest.raises(ValueError):
        run.parse_cells(text + lines[0] + "\n", request("sf-dense"), {"row"}, 2)


def test_resume_rederives_results(tmp_path):
    path = tmp_path / "raw.log"
    path.write_text(fq_log())
    cells = run.parse_cells(fq_log(), request(), {"row"}, 2)
    record = {
        "cells": cells,
        "logs": [
            {"path": str(path), "sha256": build.sha(path), "rc": 0, "symbols": ["row"]}
        ],
    }
    run.verify_result(record, request(), 2)
    record["cells"][0]["median_us"] = 0.01
    with pytest.raises(ValueError, match="raw logs"):
        run.verify_result(record, request(), 2)


def test_cli_runtime_variants_and_no_shell(tmp_path):
    args = run.request_argv(request(), tmp_path / "symbols", tmp_path, 2, 42)
    assert "--tm8-max-m=64" in args and "--correctness-repeats=1" in args
    assert not any("only-split" in x for x in args)
    p = request()
    p["problem"]["m"] = 2048
    assert "--only-split=1" in run.request_argv(
        p, tmp_path / "symbols", tmp_path, 2, 42
    )


def test_cpu_module_loader_abi_and_cache(tmp_path):
    # This is a real dlopen smoke test with CPU-only mock rows, independent of
    # the PPU numerical gate. It checks ABI rejection and function lifetimes.
    header = tmp_path / "row.hpp"
    header.write_text("struct Row { const char* symbol; int (*run)(int); };\n")
    source = tmp_path / "module.cpp"
    source.write_text(
        '#include "row.hpp"\n#include "kpack_tuner_registry.hpp"\n'
        'int plus(int x){return x+1;}\nstatic Row const rows[]={{"a",plus}};\n'
        'extern "C" kpack_tuner::Module const* kpack_tuner_module_v1(){\n'
        'static kpack_tuner::Module const m{1,sizeof(Row),1,"right",rows};return &m;}\n'
    )
    so = tmp_path / "module.so"
    common = ["c++", "-std=c++17", f"-I{ROOT/'benchmarks'}", f"-I{tmp_path}"]
    subprocess.run(
        common + ["-shared", "-fPIC", str(source), "-ldl", "-o", str(so)], check=True
    )
    main = tmp_path / "main.cpp"
    main.write_text(
        '#include "row.hpp"\n#include "kpack_tuner_registry.hpp"\n#include <cassert>\n'
        'int main(int argc,char**argv){setenv("KPACK_TUNER_MODULES",argv[1],1);\n'
        'auto r=kpack_tuner::load_registry<Row>("right");assert(r[0].run(2)==3);\n'
        'try{kpack_tuner::load_registry<Row>("wrong");return 1;}catch(std::exception const&){}\n'
        "kpack_tuner::HostWeightCache<float> c;assert(!c.matches(16,256));c.bind(16,256);\n"
        "assert(c.matches(16,256));assert(!c.matches(16,512));assert(!c.matches(16,256,2));return 0;}\n"
    )
    binary = tmp_path / "test"
    subprocess.run(common + [str(main), "-ldl", "-o", str(binary)], check=True)
    modules = tmp_path / "modules"
    modules.write_text(str(so) + "\n")
    subprocess.run([str(binary), str(modules)], check=True)


def test_source_keeps_legacy_timing_default():
    source = (ROOT / "benchmarks/fully_quantized_splitk_producer_bench.hpp").read_text()
    assert "#if defined(KPACK_TUNER_E2E) && KPACK_TUNER_E2E" in source
    assert "result.full_output = true;" in source
    assert "result.reducer_correctness_untimed = false;" in source


def test_candidate_failure_restarts_only_unfinished_work(tmp_path, monkeypatch):
    run.STOP.clear()
    starts = []

    class FakeDriver:
        def __init__(self, binary, device, sdk):
            starts.append(device)

        def close(self):
            pass

        def run(self, request_id, modules, argv, log, timeout):
            path = Path(
                next(
                    x.split("=", 1)[1] for x in argv if x.startswith("--symbols-file=")
                )
            )
            symbols = path.read_text().splitlines()
            if "bad" in symbols:
                text = "KPACK_TUNER_ROW_BEGIN symbol=a\n" + fq_log("a")
                text += "KPACK_TUNER_ROW_BEGIN symbol=bad\n"
                rc = 1
            else:
                text = "KPACK_TUNER_ROW_BEGIN symbol=c\n" + fq_log("c")
                rc = 0
            log.write_text(text)
            return rc, text

    monkeypatch.setattr(run, "Driver", FakeDriver)
    for directory in ("results", "inputs", "logs"):
        (tmp_path / directory).mkdir()
    r = request()
    r.update(id="a" * 64, cell_key="q12/dense/test", symbols=["a", "bad", "c"])
    bundle = {
        "pairs": {
            "q12-fq-dense": {
                "driver": "fake",
                "modules": [{"path": "/fake.so", "symbols": r["symbols"]}],
            }
        }
    }
    run.run_group([r], 0, bundle, tmp_path, tmp_path, 2, 10, False, lambda _: None)
    value = json.loads((tmp_path / "results" / (r["id"] + ".json")).read_text())
    assert starts == [0, 0]
    assert {c["symbol"] for c in value["cells"]} == {"a", "c"}
    assert set(value["rejected"]) == {"bad"}
    assert value["status"] == "MEASURED"
    run.run_group([r], 0, bundle, tmp_path, tmp_path, 2, 10, False, lambda _: None)
    assert starts == [0, 0]  # Verified success/rejection receipts need no launch.
