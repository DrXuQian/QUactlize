"""Bounded real-shape cohort, preserving exact shipping policy choices."""

from copy import deepcopy
from quactlize.fusion.native import Config


def points():
    return [
        dict(
            q=q,
            n=512,
            k=2048,
            tokens=t,
            mode=mode,
            experts=e,
            compute=compute,
            rounding=compute,
            channels=1,
            key=f"q{q}-{'shared' if mode == 0 else 'routed'}-t{t}",
        )
        for q, mode, e, compute in ((8, 0, 1, 0), (12, 2, 256, 1))
        for t in range(1, 9)
    ]


def incumbent(point, matched, vector):
    p = point
    key = [
        p["q"],
        p["mode"],
        p["n"] * (2 if p["mode"] else 1),
        p["k"],
        p["experts"],
        8 if p["mode"] else 1,
        p["channels"],
        p["tokens"],
        p["compute"],
    ]
    rows = [r for r in matched["exact"] if r["key"] == key]
    if len(rows) != 1:
        raise ValueError(f"missing/duplicate exact incumbent: {key}")
    choice = deepcopy(rows[0]["config"])
    overrides = [r for r in vector["rows"] if r["key"] == key]
    if len(overrides) > 1:
        raise ValueError("duplicate Q8 vector override")
    if overrides:
        old = choice.get("kind") == "simt" and [
            choice[k] for k in ("variant", "columns", "warps", "values", "split")
        ]
        if old != overrides[0]["baseline"]:
            raise ValueError("Q8 override does not match incumbent")
        choice = dict(
            kind="simt",
            **dict(
                zip(
                    ("variant", "columns", "warps", "values", "split"),
                    overrides[0]["candidate"],
                )
            ),
        )
    if choice["kind"] not in ("simt", "tc"):
        raise ValueError("unsupported incumbent, do not substitute generic SIMT")
    return dict(
        request_key=key,
        config=choice,
        vector_override=bool(overrides),
        source="PINNED_EXACT_MATCHED_POLICY_PLUS_Q8_OVERRIDE",
    )


def parent(choice):
    c = choice["config"]
    route = {0: "fq-dense", 1: "sf-dense", 2: "fq-grouped", 3: "sf-grouped"}[c["route"]]
    return {
        k: c[k]
        for k in ("symbol", "qtype", "tm", "tn", "tk", "wm", "wn", "stages", "ap", "dn")
    } | dict(route=route, persistent=c["parent_persistent"])


def candidates():
    return {
        f"{backend}-s{s}-{axis}{v}": Config(
            int(backend == "tc"),
            s,
            v if backend == "tc" else 0,
            v if backend == "simt" else 0,
        )
        for backend, axis, values in (("simt", "w", (4, 8)), ("tc", "tm", (8, 16)))
        for s in (1, 2, 4, 8)
        for v in values
    }


def ring_copies(l2_bytes, active_bytes):
    if l2_bytes <= 0 or active_bytes <= 0:
        raise ValueError("positive verified L2 and active bytes required")
    return max(2, (9 * l2_bytes + 4 * active_bytes - 1) // (4 * active_bytes))
