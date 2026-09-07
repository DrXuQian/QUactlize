"""Measured-family seeds for a bounded warmup, not a universal time model."""

import json
import math
from pathlib import Path

from .compiler import validate_parent
from .tuning import ROUTES, Tactic, digest

PARENT_FIELDS = (
    "symbol",
    "route",
    "qtype",
    "tm",
    "tn",
    "tk",
    "wm",
    "wn",
    "stages",
    "ap",
    "dn",
    "persistent",
)


def from_config(config):
    return Tactic(
        config["symbol"],
        config["algorithm"],
        config["split"],
        config["grid_mode"],
        config["grid_b"],
    )


class MeasuredCandidates:
    def __init__(self, policy):
        data = json.loads(Path(policy).read_text())
        if data.get("schema") != "quactlize.kpack-runtime-policy.v1" or data.get(
            "policy_digest"
        ) != digest({k: v for k, v in data.items() if k != "policy_digest"}):
            raise ValueError("seed policy schema/digest differs")
        self.data = data
        self.parents = {
            c["symbol"]: {k: c[k] for k in PARENT_FIELDS}
            for c in data["configurations"].values()
        }
        for p in self.parents.values():
            validate_parent(p)

    def exact_entry(self, request):
        """Historical measurement identity, not a loaded-module admission."""
        for entry in self.data["entries"]:
            key = entry["key"]
            if key[:4] != [
                request.qtype,
                ROUTES.index(request.route),
                request.n,
                request.k,
            ]:
                continue
            if key[5] != request.m:
                continue
            if request.grouped:
                if tuple(self.data["row_vectors"][entry["row_vector"]]) != request.rows:
                    continue
            elif key[4] != 1:
                continue
            return entry
        return None

    @staticmethod
    def usable(c, request):
        if (
            c["route"] != request.route
            or c["qtype"] != request.qtype
            or request.k % c["tk"]
            or request.k // (c["tk"] * c["split"]) < c["stages"] - 1
            or (c["ap"] and request.m != 1)
        ):
            return False
        if (
            not request.grouped
            and c["tm"] == 8
            and request.m > (64 if request.route == "fq-dense" else 7)
        ):
            return False
        if c["split"] > 1 and (
            request.m >= 64
            or request.k % (c["tk"] * c["split"])
            or request.k // (c["tk"] * c["split"]) < c["stages"] - 1
        ):
            return False
        return True

    def shortlist(self, request, max_parents=5, runtimes_per_parent=3):
        if not 1 <= max_parents <= 8 or not 1 <= runtimes_per_parent <= 4:
            raise ValueError("unbounded shortlist")
        scored = []
        for entry in self.data["entries"]:
            c = self.data["configurations"][entry["config_id"]]
            if not self.usable(c, request):
                continue
            key = entry["key"]
            # N/K are weight-family coordinates, not tuning bucket dimensions.
            # Nearby families are proposals only, still gated on actual inputs.
            family = abs(math.log2(request.n / key[2])) + abs(
                math.log2(request.k / key[3])
            )
            if request.grouped:
                old_rows = self.data["row_vectors"][entry["row_vector"]]
                old_expected = key[5] / key[4]
                expected = request.m / len(request.rows)
                load = abs(math.log2(expected / old_expected)) + abs(
                    math.log2(max(request.rows) / max(old_rows))
                )
                exact = tuple(old_rows) == request.rows and family == 0
            else:
                load = abs(math.log2(request.m / key[5]))
                exact = request.m == key[5] and family == 0
            scored.append(
                (
                    (
                        not exact,
                        family,
                        load,
                        entry["status"] != "MEASURED_WITHIN_5PCT",
                    ),
                    c,
                )
            )
        result = []
        counts = {}
        seen = set()
        for _, c in sorted(
            scored, key=lambda pair: (pair[0], pair[1]["symbol"], pair[1]["split"])
        ):
            tactic = from_config(c)
            if tactic in seen:
                continue
            seen.add(tactic)
            count = counts.get(tactic.parent, 0)
            if count >= runtimes_per_parent or (
                not count and len(counts) >= max_parents
            ):
                continue
            counts[tactic.parent] = count + 1
            result.append(tactic)
        return result

    def parent_union(self, tactics):
        return [self.parents[name] for name in dict.fromkeys(t.parent for t in tactics)]
