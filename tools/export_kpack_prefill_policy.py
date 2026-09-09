#!/usr/bin/env python3
"""Export per-call FQ versus (GPU metadata prepass + SF) comparisons."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def median(values):
    if len(values) < 3 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("invalid prefill samples")
    return statistics.median(values)


def export(summary):
    if (
        summary.get("status") != "PASS"
        or summary.get("failures")
        or len(summary.get("results", [])) != 28
    ):
        raise ValueError("selected device gate is incomplete")
    grouped = {}
    for r in summary["results"]:
        if (
            r.get("status") != "PASS"
            or r.get("graph_replays") != 3
            or len(r["profiles"]) != 3
            or r.get("rows_host") is not False
            or r.get("rows_device") is not False
        ):
            raise ValueError("native context lacks device-only replay evidence")
        key = tuple(r[k] for k in ("q", "n", "k", "experts", "m", "max_rows")) + (
            r["route"] // 2 * 2,
        )
        sf = r["route"] % 2
        if r.get("sf_metadata_mode") != ("PER_CALL_GPU_PREPASS" if sf else "PACKED_UNITS"):
            raise ValueError("resident-only timings cannot select a per-call SF route; rerun native gate")
        if sf in grouped.setdefault(key, {}):
            raise ValueError("duplicate native context")
        times = []
        for profile in r["profiles"]:
            rows = profile["rows"]
            if (
                len(rows) != r["experts"]
                or any(type(x) is not int or x < 0 for x in rows)
                or sum(rows) != r["m"]
                or max(rows) > r["max_rows"]
            ):
                raise ValueError("invalid native router receipt")
            if not math.isfinite(profile["error"]) or not 0 <= profile["error"] < 0.005:
                raise ValueError("native oracle failed")
            times.append(median(profile["samples_us"]))
        grouped[key][sf] = (median(times), r)
    expected = {(q, 1024, 5120, 1, t, t, 0) for q in range(10, 15) for t in (1, 128)}
    expected.update(
        (q, n, k, 256, t * 8, t, 2)
        for q, n, k in ((12, 512, 2048), (13, 2048, 512))
        for t in (1, 128)
    )
    if set(grouped) != expected:
        raise ValueError("native workload coverage differs")
    lines = ["KPACK_PREFILL_POLICY_V2_PER_CALL"]
    records = []
    for key, pair in sorted(grouped.items()):
        if set(pair) != {0, 1}:
            raise ValueError("missing FQ/SF pair")
        fq, fqr = pair[0]
        sf, sfr = pair[1]
        if [p["rows"] for p in fqr["profiles"]] != [p["rows"] for p in sfr["profiles"]]:
            raise ValueError("FQ/SF router profiles differ")
        preparation = median(sfr["prepass_samples_us"])
        selected = int(sf < fq * 0.98)
        lines.append("\t".join(map(str, key + (selected,))))
        records.append(
            dict(
                key=key,
                selected="SF" if selected else "FQ",
                fq_us=fq,
                sf_us=sf,
                prepass_us=preparation,
                prepass_first_us=sfr["prepass_samples_us"][0],
                fq_selection=fqr["selection"],
                sf_selection=sfr["selection"],
                scope="PER_CALL_PREPASS_PLUS_GEMM_2PCT_MARGIN_ALLOCATION_EXCLUDED",
            )
        )
    if len(records) != 14:
        raise ValueError("native context denominator differs")
    return "\n".join(lines) + "\n", records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--native-bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    s = json.loads(a.results.read_text())
    manifest = hashlib.sha256(
        (a.native_bundle / "manifest.json").read_bytes()
    ).hexdigest()
    if s.get("manifest_sha256") != manifest:
        raise ValueError("measured native package differs")
    text, records = export(s)
    with a.output.open("x") as f:
        f.write(text)
    with a.output.with_suffix(".json").open("x") as f:
        json.dump(
            dict(
                native_manifest_sha256=manifest,
                source_sha256=hashlib.sha256(a.results.read_bytes()).hexdigest(),
                policy_sha256=hashlib.sha256(text.encode()).hexdigest(),
                entries=records,
            ),
            f,
            indent=2,
        )
        f.write("\n")
    print(f"KPACK_PREFILL_POLICY PASS entries={len(records)} output={a.output}")


if __name__ == "__main__":
    main()
