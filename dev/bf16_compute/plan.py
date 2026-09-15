"""Bounded capability coverage, not a Cartesian configuration search."""
from dataclasses import asdict, dataclass

QTYPES = (10, 11, 12, 13, 14, 8)
SPLITS = (1, 2, 4, 8)


@dataclass(frozen=True, order=True)
class Parent:
    qtype: int
    route: str
    tm: int = 8
    persistent: int = 0
    compute: str = "bf16"
    tk: int = 256

    @property
    def key(self):
        return f"q{self.qtype}-{self.route}-tm{self.tm}-tk{self.tk}-p{self.persistent}-{self.compute}"

    def record(self):
        return dict(qtype=self.qtype, route=self.route, tm=self.tm, tn=64,
                    tk=self.tk, wm=min(self.tm, 16), wn=16, stages=2, ap=0, dn=16,
                    persistent=self.persistent if self.route == "fq-grouped" else -1,
                    symbol="bf16_gate_" + self.key.replace("-", "_"))


def routes(q):
    return ("sf",) if q == 8 else ("fq", "sf")


def cases():
    result = []
    for q in QTYPES:
        for quant in routes(q):
            for profile, tm in (("small", 8), ("large", 64), ("ordinary", 64)):
                for algorithm in ((0,) if profile == "ordinary" else (0, 1)):
                    parent = Parent(q, quant + "-grouped", tm, algorithm if quant == "fq" else 0)
                    for split in SPLITS:
                        result.append(dict(family="grouped", q=q, quant=quant,
                            profile=profile, algorithm=algorithm, split=split,
                            parent=parent, compute="bf16"))
        # One independent legacy-compute control per format and TC family.
        quant = "sf" if q == 8 else "fq"
        result.append(dict(family="grouped", q=q, quant=quant, profile="small",
            algorithm=0, split=1, parent=Parent(q, quant + "-grouped", compute="f16"),
            compute="f16"))
        for tokens in range(1, 9):
            for storage in (1, 2):
                for mode, channels in ((0, 1), (1, 1), (2, 1), (2, 8)):
                    result.append(dict(family="simt", q=q, tokens=tokens,
                        storage=storage, mode=mode, channels=channels, compute="bf16",
                        split=1 if tokens % 2 else 2))
        for mode in (0, 2):
            result.append(dict(family="simt", q=q, tokens=1, storage=1,
                mode=mode, channels=1, compute="f16", split=1))
        for tokens in (1, 4, 8):
            for merged in (False, True):
                for kind in ("tc", "simt", "mixed"):
                    result.append(dict(family="moe", q=q, tokens=tokens, merged=merged,
                        kind=kind, parent=Parent(q, quant + "-grouped"), compute="bf16"))
    # Real model combinations: Q4 merged gate/up with Q5 or Q6 down.
    for down_q in (13, 14):
        for tokens in (1, 8):
            for kind in ("tc", "mixed"):
                result.append(dict(family="moe", q=12, down_q=down_q, tokens=tokens,
                    merged=True, kind=kind, parent=Parent(12, "fq-grouped"), compute="bf16"))
    for compute in ("bf16", "f16"):
        for split in SPLITS:
            result.append(dict(family="outlier", q=14, split=split, compute=compute,
                parent=Parent(14, "fq-dense", compute=compute, tk=128)))
    for index, row in enumerate(result):
        fields = "-".join(f"{k}{v}" for k, v in row.items() if k not in ("parent", "family"))
        row["id"] = f"{index:04d}-{row['family']}-{fields}"
    return result


def grouped_rows(profile, repeat=0):
    import numpy as np
    experts = 1025 if profile == "ordinary" else 256
    rows = np.zeros(experts, dtype="<i4")
    counts = (1, 8, 9, 17) if profile == "small" else (1, 63, 64, 65, 129, 257)
    owners = (np.arange(len(counts)) * 47 + repeat * 11) % experts
    owners[-1] = (experts - 1 + repeat * 11) % experts
    rows[owners] = counts
    return rows


def modules():
    return sorted({case["parent"] for case in cases() if "parent" in case})


def serializable():
    result = []
    for case in cases():
        row = dict(case)
        if "parent" in row:
            row["parent"] = asdict(row["parent"])
        result.append(row)
    return result
