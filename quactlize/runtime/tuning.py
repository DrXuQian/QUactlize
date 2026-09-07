"""Small, explicit initialization-time tuner; inference only reads the cache.

Grouped cache buckets describe expected rows per expert, as in DeepGEMM,
with active/max-row bands to distinguish empty/skewed routers. They are
performance hints, not an assertion that different row vectors have equal
cost. The backend must resolve grids and validate the actual request again.
"""

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time

ROUTES = ("fq-dense", "sf-dense", "fq-grouped", "sf-grouped")
GROUPS = {10: 16, 11: 16, 12: 32, 13: 32, 14: 16}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def band(value):
    return 0 if value == 0 else 1 << (value.bit_length() - 1)


@dataclass(frozen=True)
class Request:
    route: str
    qtype: int
    n: int
    k: int
    m: int
    rows: tuple = ()

    def __post_init__(self):
        if (
            self.route not in ROUTES
            or self.qtype not in GROUPS
            or any(
                type(x) is not int or not 0 < x <= 2147483647
                for x in (self.n, self.k, self.m)
            )
            or self.n % 256
            or self.k % (512 if self.qtype in (11, 14) else 256)
        ):
            raise ValueError("unsupported K-pack request")
        if self.grouped:
            if (
                not isinstance(self.rows, tuple)
                or not self.rows
                or any(type(x) is not int or x < 0 for x in self.rows)
                or sum(self.rows) != self.m
            ):
                raise ValueError("grouped request requires actual expert rows")
        elif self.rows:
            raise ValueError("dense request cannot have expert rows")

    @property
    def grouped(self):
        return self.route.endswith("grouped")

    @property
    def mapping(self):
        return 0x51344B5034540001 if self.qtype == 12 else 0x514B504B54000001

    @property
    def exact_key(self):
        return digest(asdict(self))

    @property
    def bucket(self):
        if self.grouped:
            expected = (self.m + len(self.rows) - 1) // len(self.rows)
            load = (
                len(self.rows),
                band(expected),
                band(max(self.rows)),
                band(sum(x > 0 for x in self.rows)),
            )
        else:
            # Exact tiny decode bands preserve AP1 and M8 admission boundaries.
            load = ("tiny", self.m) if self.m <= 8 else ("band", band(self.m))
        return digest((self.route, self.qtype, self.n, self.k, self.mapping, load))


@dataclass(frozen=True)
class Tactic:
    parent: str
    algorithm: str
    split: int = 1
    grid_mode: str = "implicit"
    grid_b: int = 0

    @property
    def key(self):
        return digest(asdict(self))


class UnsupportedTactic(Exception):
    """Pre-launch rejection, safe to try another already-admitted candidate."""


class TuningCache:
    """Versioned timing cache. Identity must bind device, SDK and kernel code."""

    schema = "quactlize.kpack-warmup-cache.v1"

    def __init__(self, identity, path=None):
        required = {"device", "compute_units", "sdk", "kernel", "inventory"}
        if not required <= identity.keys() or any(not identity[k] for k in required):
            raise ValueError("cache identity lacks device/SDK/kernel/inventory binding")
        self.identity = dict(identity)
        self.path = Path(path) if path else None
        self.entries = {}
        self.lock = threading.RLock()
        if self.path and self.path.exists():
            self.entries = self._read()

    def _read(self):
        value = json.loads(self.path.read_text())
        if (
            value.get("schema") != self.schema
            or value.get("identity") != self.identity
            or value.get("digest")
            != digest({k: v for k, v in value.items() if k != "digest"})
        ):
            raise ValueError(
                "stale or corrupt timing cache; use a cache for this build/device"
            )
        return value["entries"]

    def get(self, request):
        with self.lock:
            entry = self.entries.get(request.bucket)
            if entry is None:
                return None
            return dict(
                entry,
                status=(
                    "MEASURED_EXACT"
                    if entry["exact"] == request.exact_key
                    else "BUCKET_HINT"
                ),
                performance_bound=False,
            )

    def put(self, request, tactic, measured_us, candidates, elapsed_ms):
        if not math.isfinite(measured_us) or measured_us <= 0:
            raise ValueError("invalid timing")
        entry = dict(
            exact=request.exact_key,
            tactic=asdict(tactic),
            us=measured_us,
            candidates=candidates,
            tuning_ms=elapsed_ms,
        )
        with self.lock:
            self.entries[request.bucket] = entry
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.with_suffix(self.path.suffix + ".lock").open(
                    "a"
                ) as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    merged = self._read() if self.path.exists() else {}
                    merged[request.bucket] = entry
                    value = dict(
                        schema=self.schema, identity=self.identity, entries=merged
                    )
                    value["digest"] = digest(value)
                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=self.path.parent, delete=False
                    ) as tmp:
                        json.dump(value, tmp, sort_keys=True, allow_nan=False)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                    os.replace(tmp.name, self.path)
                    self.entries = merged


class Tuner:
    """Backend operations are prepare, check, run, measure, synchronize, close.

    prepare must not compile: call compile_only first. check is a correctness
    callback against an independent/reference output, not a timer. A runtime
    failure propagates; only UnsupportedTactic is a safe candidate rejection.
    """

    def __init__(
        self,
        cache,
        max_candidates=15,
        budget_ms=100,
        warmups=2,
        repeats=5,
        samples=3,
        improve_pct=5,
    ):
        if (
            not 1 <= max_candidates <= 32
            or budget_ms <= 0
            or warmups < 0
            or not 1 <= repeats <= 1024
            or not 1 <= samples <= 11
            or improve_pct < 0
        ):
            raise ValueError("invalid tuning budget")
        self.cache = cache
        self.max_candidates = max_candidates
        self.budget_ms = budget_ms
        self.warmups, self.repeats, self.samples = warmups, repeats, samples
        self.improve_pct = improve_pct
        self.lock = threading.RLock()

    def select(self, request, backend, fallback=None):
        # No compilation, timing or hidden stream synchronization here.
        if backend.identity != self.cache.identity:
            raise ValueError("backend differs from timing-cache identity")
        entry = self.cache.get(request)
        if entry:
            tactic = Tactic(**entry["tactic"])
            if backend.admissible(request, tactic):
                return dict(entry, tactic=tactic)
        if fallback is not None and backend.admissible(request, fallback):
            return dict(
                status="ADMITTED_FALLBACK", tactic=fallback, performance_bound=False
            )
        return dict(status="FALLBACK_REQUIRED", performance_bound=False)

    def warmup(self, request, candidates, backend, *, force=False):
        # Serialization avoids two host threads profiling against one another.
        # Cross-process/device contention is the embedding application's policy.
        with self.lock:
            if backend.is_capturing():
                raise ValueError("tuning must finish before graph capture")
            cached = self.select(request, backend)
            if not force and cached["status"] != "FALLBACK_REQUIRED":
                return cached
            candidates = list(dict.fromkeys(candidates))
            if not candidates or len(candidates) > self.max_candidates:
                raise ValueError("candidate list must be nonempty and bounded")
            start = time.monotonic()
            measured, rejected = [], []
            for tactic in candidates:
                if measured and (time.monotonic() - start) * 1000 >= self.budget_ms:
                    break
                handle = None
                try:
                    handle = backend.prepare(request, tactic)
                    backend.check(handle)
                    for _ in range(self.warmups):
                        backend.run(handle)
                    probe = backend.measure(handle, 1)
                    if not math.isfinite(probe) or probe <= 0:
                        raise RuntimeError("nonfinite/zero candidate timing")
                    # Long prefill kernels need fewer repetitions than tiny
                    # decode kernels. This is a budget, not a precision proof.
                    repeats = max(1, min(self.repeats, math.ceil(1000 / probe)))
                    times = [
                        backend.measure(handle, repeats) for _ in range(self.samples)
                    ]
                    if any(not math.isfinite(t) or t <= 0 for t in times):
                        raise RuntimeError("nonfinite/zero candidate timing")
                    measured.append((statistics.median(times), tactic))
                except UnsupportedTactic:
                    rejected.append(tactic.key)
                finally:
                    if handle is not None:
                        backend.synchronize()
                        backend.close(handle)
            if not measured:
                return dict(
                    status="FALLBACK_REQUIRED",
                    rejected=rejected,
                    performance_bound=False,
                )
            # First candidate is the caller's incumbent. A small measured gain
            # does not justify a noisy tactic change.
            best = min(measured, key=lambda x: x[0])
            incumbent = measured[0]
            if incumbent[0] <= best[0] * (1 + self.improve_pct / 100):
                best = incumbent
            elapsed = (time.monotonic() - start) * 1000
            self.cache.put(request, best[1], best[0], len(measured), elapsed)
            return dict(
                self.cache.get(request),
                tactic=best[1],
                rejected=rejected,
                budget_exhausted=elapsed >= self.budget_ms,
                timing_scope="RESIDENT_FULL_OUTPUT_SELECTED_CANDIDATES",
            )
