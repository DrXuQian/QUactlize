#!/usr/bin/env python3
"""Host-only audit and ACU re-import of the pinned real-model GEMV experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_model.plan import POINTS, candidates, inventory
from dev.gemv_model.run import profile_records, summarize
from quactlize.runtime.compiler import sha

SOURCE = "2d8e5030afbbe757ab6759c0d2a37440310f1a32"
MANIFEST = "f54f6f627cc62f60f81664372d31b7e29c3bf9dedd406f88db331488d64301db"
METRICS = (
    "ppu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "derived__l2_bytes_pipe_l1_total", "derived__kvd_transactions_pipe_l2_total",
    "ws__inst_executed_op_vmem_ld.sum", "pu__inst_executed.sum",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "cu__warps_active.avg.pct_of_peak_sustained_active",
    "cu__inst_executed_pipe_falu_fp32.avg", "cu__inst_executed_pipe_salu.avg",
    "derived__tsm_bank_conflicts_total",
    "pu__we_average_warps_issue_stalled_vmem_pipe_busy_per_issue_active.ratio",
    "pu__we_average_warps_issue_stalled_vmem_dependency_per_issue_active.ratio",
    "pu__we_average_warps_issue_stalled_inst_fetch_per_issue_active.ratio",
    "pu__we_average_warps_issue_stalled_compute_dependency_per_issue_active.ratio",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)


def raw_rows(text):
    start = text.find('"ID"')
    require(start >= 0, "ACU raw header absent")
    return [r for r in csv.DictReader(io.StringIO(text[start:])) if r.get("Kernel Name")]


def check_selection(point, arm, receipt):
    config = candidates(point)[0 if arm == "incumbent" else int(arm)]
    tc = arm == "incumbent" and bool(point.tc)
    expected = dict(arm=arm, name="incumbent" if arm == "incumbent" else config.name,
                    kind="tc" if tc else "simt",
                    config=point.parent | dict(split=point.tc[-1]) if tc else asdict(config),
                    scope="PAIRED_GATE_UP_PLUS_SWIGLU" if point.paired else "GEMV_INCLUDING_SPLITK_REDUCTION")
    require(all(receipt.get(key) == value for key, value in expected.items()), "selection differs: " + arm)


def check_finalists(point, data):
    eligible = [str(i) for i, c in enumerate(candidates(point)) if c.name not in ("clone", "generic")]
    finalists = sorted(eligible, key=lambda arm: statistics.median(data["screen_us"][arm]))[:2]
    require(list(data["summary"]) == ["incumbent", *finalists], "screen finalists differ")
    best = min(finalists, key=lambda arm: data["summary"][arm]["median_us"])
    require(data["best_candidate"] == best, "confirmation winner differs")


def audit(args):
    source, output = args.results.resolve(strict=True), args.output.resolve()
    require(not output.exists(), "review output already exists")
    output.mkdir(parents=True)
    sweep = source / "sweep"
    require((source / "source.txt").read_text().strip() == SOURCE, "runner source differs")
    require(sha(source / "manifest.json") == MANIFEST, "bundle manifest differs")
    manifest = json.loads((source / "manifest.json").read_text())
    identity = json.loads((sweep / "identity.json").read_text())
    require(identity["manifest"] == MANIFEST, "identity manifest differs")
    require(manifest["inventory"] == json.loads(json.dumps(inventory())), "inventory differs")
    for path, digest in {**manifest["source_hashes"], **identity["runner"]}.items():
        require(sha(ROOT / path) == digest, "source input differs: " + path)
    require(sha(source / "native-inspection.json") == manifest["payloads"]["native-inspection.json"], "ISA receipt differs")
    campaign = json.loads((sweep / "summary.json").read_text())
    require(campaign["complete"] and campaign["status"] == "PASS", "campaign incomplete")
    require([r["point"] for r in campaign["records"]] == [p.name for p in POINTS], "point denominator/order differs")
    jobs, records = [], []
    numeric_count = provider_count = sample_count = 0
    devices = []
    for p, row in zip(POINTS, campaign["records"]):
        file = sweep / (p.name + ".json")
        d = json.loads(file.read_text())
        require(d["status"] == "PASS" and row["status"] == "PASS" and row["rc"] == 0, "point failed: " + p.name)
        require(d["point"] == json.loads(json.dumps(asdict(p))), "shape/compute differs")
        require(not d["failures"] and d["manifest_sha256"] == MANIFEST, "point identity/numerics failed")
        require(d["runtime"] == identity["runtime"], "runtime changed between points")
        devices.append(d["device"])
        expected = {"incumbent", *map(str, range(len(candidates(p))))}
        require(set(d["numerics"]) == expected == set(d["screen_us"]), "numeric/screen denominator differs")
        error_max = 0.
        for arm, n in d["numerics"].items():
            require(n["status"] == "PASS", "failed numeric arm")
            check_selection(p, arm, n["selection"])
            actual = [(v["tokens"], v["repeat"]) for v in n["controls"]]
            wanted = [(m, r) for m in (1, 2, 8) for r in ((0, 1, "bf16-range") if p.compute else (0, 1))]
            require(actual == wanted, "missing numeric control")
            for v in n["controls"]:
                require(math.isfinite(v["error"]) and 0 <= v["error"] < .005, "invalid numeric error")
                error_max = max(error_max, v["error"])
            require(len(d["screen_us"][arm]) == 3 and all(math.isfinite(x) and x > 0 for x in d["screen_us"][arm]), "invalid screen")
            provider_count += 1
            numeric_count += len(n["controls"])
        fixture = d["fixture"]
        weight_bytes = fixture["physical_weight_bytes_per_call"]
        require(fixture["l2_bytes"] == 67108864 == d["l2"]["l2_bytes"], "L2 differs")
        require(fixture["copies"] >= 2 and fixture["rotating_active_bytes"] == weight_bytes * fixture["copies"] >= 2.25 * 67108864, "cold ring too small")
        target = 60 if weight_bytes >= 16 * 1024 * 1024 else 40
        require(len(d["summary"]) == 3 and d["summary"] == row["summary"], "confirmation/summary differs")
        check_finalists(p, d)
        for arm, value in d["summary"].items():
            check_selection(p, arm, value)
            recomputed = summarize(value["samples_us"], weight_bytes, target)
            for name in ("median_us", "modeled_weight_MBU_pct"):
                require(close(value[name], recomputed[name]), "recomputed " + name + " differs")
            require(value["round_medians_us"] == recomputed["round_medians_us"], "round median differs")
            require(value["target_MBU_pct"] == target and value["target_met"] == recomputed["target_met"], "target differs")
            baseline = d["summary"]["incumbent"]["median_us"]
            require(close(value["delta_pct"], 100 * (value["median_us"] / baseline - 1)), "delta differs")
            sample_count += sum(map(len, value["samples_us"]))
        winner = d["summary"][d["best_candidate"]]
        incumbent = d["summary"]["incumbent"]
        ratios = [100 * (a / b - 1) for a, b in zip(winner["round_medians_us"], incumbent["round_medians_us"])]
        records.append(dict(point=p.name, input_output="F32", compute="BF16" if p.compute else "F16",
                            incumbent_us=incumbent["median_us"], candidate_us=winner["median_us"],
                            candidate=winner["name"], arm=d["best_candidate"], config=winner["config"],
                            delta_pct=winner["delta_pct"], round_deltas_pct=ratios,
                            candidate_MBU_pct=winner["modeled_weight_MBU_pct"], target_MBU_pct=target,
                            max_numeric_error=error_max, point_sha256=sha(file),
                            decision="EXACT_M1_INTEGRATION_CANDIDATE" if all(x < 0 for x in ratios) else "RETAIN_INCUMBENT",
                            fixture=fixture, profiles=[]))
        require(len(row["profiles"]) == 2 and [x["arm"] for x in row["profiles"]] == ["incumbent", d["best_candidate"]], "profile denominator differs")
        for profile in row["profiles"]:
            require(profile["status"] == "PASS", "profile failed")
            report = sweep / profile["report"]
            require(report.parent == sweep and sha(report) == profile["sha256"], "report hash differs")
            require((sweep / profile["csv"]).parent == sweep, "profile CSV outside sweep")
            receipt = json.loads(report.with_suffix(".json").read_text())
            require(receipt["status"] == "PASS" and receipt["arm"] == profile["arm"], "profile proof differs")
            require(receipt["device"] == d["device"] and receipt["runtime"] == d["runtime"], "profile device/runtime differs")
            require(receipt["fixture"]["raw_sha256"] == fixture["raw_sha256"], "profile weight bytes differ")
            jobs.append((p, profile, report))

    def reimport(job):
        p, profile, report = job
        raw = subprocess.check_output([str(args.acu), "--import", str(report), "--page", "raw", "--csv"], text=True, stderr=subprocess.STDOUT)
        received = raw_rows(raw)
        original = raw_rows((sweep / profile["csv"]).read_text())
        require(received == original, "reimported counters differ: " + report.name)
        geometry = profile_records(raw, p, profile["arm"])
        require(geometry == profile["kernels"], "reimported geometry differs")
        (output / profile["csv"]).write_text(raw)
        result = dict(arm=profile["arm"], report_sha256=sha(report), csv_sha256=sha(sweep / profile["csv"]),
                      components=[g | dict(metrics={m: float(row[m]) for m in METRICS}) for row, g in zip(received, geometry)])
        print(f"MODEL_GEMV_REIMPORT point={p.name} arm={profile['arm']} kernels={len(received)} PASS", flush=True)
        return p.name, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, profile in pool.map(reimport, jobs):
            next(r for r in records if r["point"] == name)["profiles"].append(profile)
    require(all(d == devices[0] for d in devices), "physical device changed")
    require((provider_count, numeric_count, sample_count, len(jobs)) == (76, 510, 2160, 16), "audit total differs")
    evidence = dict(status="PASS", scope="ROTATING_COMPLETE_M1_CALL_NOT_MODEL_TPOT", source=SOURCE, manifest_sha256=MANIFEST,
                    device=devices[0], runtime=identity["runtime"], providers=provider_count,
                    numeric_checks=numeric_count, event_samples=sample_count, reimported_reports=len(jobs),
                    reimported_kernels=sum(len(x["components"]) for r in records for x in r["profiles"]),
                    idle_admission="NOT_INDEPENDENTLY_PROVEN", production_changed=False, records=records)
    (output / "review.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "output", "acu"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = audit(parser.parse_args())
    print("MODEL_GEMV_REVIEW PASS", result["providers"], result["numeric_checks"], result["event_samples"], result["reimported_reports"])
