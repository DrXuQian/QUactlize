#!/usr/bin/env python3
"""The C++ that WRITES attempt/sample records and the Python that READS them, checked against each other.

WHY THIS IS NOT COVERED BY THE TWO SELF-TESTS IT SITS BETWEEN. benchmarks/bench_samples.hpp emits the records;
benchmarks/analyse.py parses them and reports an attempt with no matching sample as where a sweep died. Both have
self-tests, and neither test can catch the failure that matters: analyse.py's fixtures are JSON strings written
by hand in analyse.py. They assert that the reader agrees with what its own author believed the writer emits. A
renamed field, a dropped key, a `rec` tag spelled differently -- every one of those leaves BOTH sides green while
the pair is broken, and the symptom is that a real sweep's failing candidate is silently never reported. That is
the same two-spellings defect the config table and the dispatch chain were merged to remove.

So this compiles the REAL header, runs it, and feeds its ACTUAL BYTES to the REAL parser.

It needs only a host C++ compiler: bench_samples.hpp is <cstdio> and <cstring>, no CUDA, which is why the check
belongs in the device-free tier.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROBE = r'''
#include "bench_samples.hpp"
int main() {
  bench_samples::Sample s{};
  s.fixture = "dense-m2048-n4096-k4096-gs32"; s.dist = "dense-v1"; s.schema = "i4";
  s.n = 4096; s.k = 4096; s.gs = 32;
  s.experts = 0; s.rows = 2048; s.mmax = 2048;
  s.tm = 64; s.tn = 64; s.tk = 64; s.wm = 64; s.wn = 32; s.st = 3;
  s.pass = 0;
  // The pair that MUST cancel: a candidate that launched and finished.
  bench_samples::attempt(s);
  s.us = 209.27;
  bench_samples::emit(s);
  // The one that must NOT: a candidate that launched and died. Nothing follows it, exactly as a device assert
  // would leave the file.
  bench_samples::Sample d = s;
  d.tm = 16; d.tn = 16; d.wm = 16; d.wn = 16; d.st = 2; d.us = 0.0;
  bench_samples::attempt(d);
  // And the THIRD outcome: tried, and the bench said no. It must resolve its attempt exactly as a sample does,
  // or a healthy sweep containing one unsupported config reports a dead run.
  bench_samples::Sample e = s;
  e.tm = 32; e.tn = 32; e.wm = 32; e.wn = 32; e.st = 4; e.us = 0.0;
  bench_samples::attempt(e);
  bench_samples::excluded(e, "planted: unsupported for this shape");
  return 0;
}
'''


def fail(msg: str) -> int:
    print(f"[attempt-roundtrip] FAIL: {msg}")
    return 1


def main() -> int:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyse", ROOT / "benchmarks" / "analyse.py")
    analyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyse)

    with tempfile.TemporaryDirectory() as td:
        src, exe, out = Path(td) / "p.cpp", Path(td) / "p", Path(td) / "r.jsonl"
        src.write_text(PROBE)
        cc = subprocess.run(["c++", "-std=c++17", f"-I{ROOT / 'benchmarks'}", str(src), "-o", str(exe)],
                            capture_output=True, text=True)
        if cc.returncode:
            return fail(f"bench_samples.hpp does not compile as host C++: {cc.stderr.strip().splitlines()[-1:]}")
        run = subprocess.run([str(exe)], capture_output=True, text=True, env={"BENCH_JSONL": str(out), "PATH": "/usr/bin"})
        if run.returncode or not out.is_file():
            return fail(f"probe did not write {out} (rc={run.returncode})")

        text = out.read_text()
        runs, samples, attempts, excludeds, bad = analyse.load(text)

        # THE PARSER MUST NOT REJECT ITS OWN WRITER'S OUTPUT. This is the check that a renamed field trips.
        if bad:
            return fail("the reader rejected lines the writer produced -- the two have drifted apart:\n"
                        + "\n".join(f"    {b}" for b in bad)
                        + f"\n  raw:\n" + "".join(f"    {l}\n" for l in text.splitlines()))
        if (len(attempts), len(samples), len(excludeds)) != (3, 1, 1):
            return fail(f"expected 3 attempts, 1 sample, 1 exclusion; parsed "
                        f"{len(attempts)}, {len(samples)}, {len(excludeds)}")

        stopped = analyse.unfinished(samples, attempts, excludeds)
        if len(stopped) != 1:
            return fail(f"expected exactly 1 unfinished attempt (the excluded one must NOT count), "
                        f"got {len(stopped)}: {stopped}")
        # And it must be the RIGHT one: matching must key on the identity, not merely count.
        if (stopped[0].get("tm"), stopped[0].get("st")) != (16, 2):
            return fail(f"the wrong attempt was reported as unfinished: {stopped[0]}")

        # The completed pair must agree field for field apart from `us`, or an attempt and its sample could
        # describe different runs and the matching would be luck.
        a_done = next(a for a in attempts if a.get("tm") == 64)
        s_done = samples[0]
        differing = {k for k in set(a_done) | set(s_done)
                     if k not in ("rec", "us") and a_done.get(k) != s_done.get(k)}
        if differing:
            return fail(f"attempt and its sample disagree on {sorted(differing)} -- they may describe different runs")
        if "us" in a_done:
            return fail("the attempt record carries a `us`; nothing is measured before the launch")

    print(f"[attempt-roundtrip] PASS -- the writer's actual bytes parse, the finished pair cancels, "
          f"the dead candidate is named (tm=16 st=2), and attempt/sample agree on every field but `us`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
