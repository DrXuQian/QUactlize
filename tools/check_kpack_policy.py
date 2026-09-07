#!/usr/bin/env python3
"""Host-only Python/C++ selector parity at measured points and boundaries."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile

import generate_kpack_policy_header as header
import kpack_policy as policy

PROBE = r"""
#include "kpack_policy.hpp"
#include <iostream>
#include <string>
int main() {
  using namespace quactlize_kpack_policy;
  Query q; int route; std::string device;
  while (std::cin >> q.qtype >> route >> q.n >> q.k >> q.group_size >> q.m
                 >> q.total_rows >> q.max_rows >> q.experts >> q.compute_units
                 >> q.mapping_id >> device) {
    q.route = static_cast<Route>(route); q.device_name = device.c_str();
    auto r = select(q);
    std::cout << int(r.status) << ' ' << (r.config ? r.config->id : "-") << ' ' << r.grid << '\n';
  }
}
"""


def queries(model):
    result = []
    for family in model["families"]:
        q, route, n, k, gs, experts = family["key"]
        observed = family["observed"] + [b["features"] for b in family["blocked"]]
        points = {tuple(p) for p in observed}
        if route.endswith("dense"):
            values = sorted(p[0] for p in points)
            points.update(((a + b) // 2,) for a, b in zip(values, values[1:]))
            points.update(
                (max(1, m + delta),) for m in (1, 8, 64, *values) for delta in (-1, 1)
            )
        else:
            for x, y in list(points):
                points.update(
                    (x + delta, y)
                    for delta in (-1, 1)
                    if x + delta >= y and x + delta <= y * experts
                )
        for point in sorted(points):
            problem = dict(qtype=q, n=n, k=k, group_size=gs)
            problem.update(zip(policy.axes(route), point))
            if route.endswith("grouped"):
                problem["experts"] = experts
            result.append((route, problem, "PPU-ZW810", 72, policy.mapping(q)))
        example = result[-1]
        result.extend(
            (example[:2] + (name, cu, mapping))
            for name, cu, mapping in (
                ("other-device", 72, policy.mapping(q)),
                ("PPU-ZW810", 1, policy.mapping(q)),
                ("PPU-ZW810", 72, "0x0"),
            )
        )
    return result


def check(model):
    inputs, expected, statuses = [], [], Counter()
    for route, p, name, cu, mapping in queries(model):
        r = policy.select(
            model, route, p, device_name=name, compute_units=cu, mapping_id=mapping
        )
        status = (
            "NO_MEASURED_POLICY",
            "MEASURED_POLICY",
            "INTERPOLATED_PROPOSAL",
        ).index(r["status"])
        statuses[r["status"]] += 1
        expected.append(
            f"{status} {r.get('config_id', '-')} {r.get('config', {}).get('grid', 0)}"
        )
        inputs.append(
            " ".join(
                map(
                    str,
                    (
                        p["qtype"],
                        policy.ROUTES.index(route),
                        p["n"],
                        p["k"],
                        p["group_size"],
                        p.get("m", 0),
                        p.get("total_rows", 0),
                        p.get("max_rows", 0),
                        p.get("experts", 1),
                        cu,
                        int(mapping, 16),
                        name,
                    ),
                )
            )
        )
    with tempfile.TemporaryDirectory(prefix="kpack-policy-host-") as scratch:
        path = Path(scratch)
        (path / "kpack_policy.hpp").write_text(header.generate(model))
        build = subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-x",
                "c++",
                "-",
                "-I",
                str(path),
                "-o",
                str(path / "probe"),
            ],
            input=PROBE,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if build.returncode:
            raise ValueError(f"host selector compilation failed: {build.stderr}")
        run = subprocess.run(
            [str(path / "probe")],
            input="\n".join(inputs) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
        )
        if run.returncode or run.stdout.splitlines() != expected:
            actual = run.stdout.splitlines()
            first = next(
                (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                min(len(actual), len(expected)),
            )
            raise ValueError(
                f"Python/C++ selector mismatch at query {first}: {inputs[first:first+1]}"
            )
    return {
        "status": "PASS",
        "queries": len(inputs),
        "query_status": dict(statuses),
        "policy_sha256": policy.digest(model),
        "device_execution": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    try:
        result = check(json.loads(args.policy.read_text()))
    except (OSError, KeyError, ValueError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    print("KPACK_POLICY_HOST_CHECK " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
