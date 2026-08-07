"""THE TWO SELECTION PROCEDURES MUST AGREE, or the C++ one cannot be deleted.

benchmarks/bench_select.hpp decides inside the bench; benchmarks/analyse.py decides outside it. The plan is to
delete the first (docs/BENCH_DESIGN.md step 3). This test is what makes that safe: it feeds one planted sample
file to both and compares the verdicts field by field.

WHY PLANTED AND NOT RECORDED DATA. A recorded run cannot exercise the cases that matter -- a candidate whose
band just overlaps the leader's, a candidate that just misses, a one-pass file. Those are exactly the boundaries
where the two implementations could differ and where a real sweep is unlikely to land. The fixtures below place
a candidate on each side of the boundary deliberately.

Runs with no device and no PPU SDK: bench_select.hpp is host C++ and analyse.py is Python.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
XCHECK_SRC = ROOT / "benchmarks" / "xcheck_select.cpp"
ANALYSE = ROOT / "benchmarks" / "analyse.py"


def _sample(fixture, cfg, pass_i, us, **over):
    tm, tn, tk, wm, wn, st = cfg
    rec = dict(rec="s", fixture=fixture, dist="planted-v1", schema="i4",
               n=512, k=2048, gs=32, experts=256, rows=128, mmax=420,
               tm=tm, tn=tn, tk=tk, wm=wm, wn=wn, st=st, bc=0, bc_eff=0)
    rec.update(over)
    rec["pass"] = pass_i
    rec["us"] = us
    # SEPARATORS MATTER. bench_samples.hpp writes compact JSON with no space after the colon, and the C++
    # cross-checker parses that shape by hand. json.dumps' default `": "` produced a file the C++ side silently
    # read as zero samples -- so the first version of this test compared a real verdict against an EMPTY one and
    # would have "agreed" the moment the assertions were looser. Planted data has to be byte-compatible with
    # what the producer writes, or the check is over a format nothing emits.
    return json.dumps(rec, separators=(",", ":"))


LEADER = (64, 128, 64, 64, 64, 3)
OVERLAP = (32, 64, 64, 32, 32, 3)     # min 101.5 lands inside the leader's [100, 102] -> TIE
CLEAR = (16, 32, 64, 16, 16, 2)       # min 300 is far outside -> not a tie

# The boundary is the point of the fixture: OVERLAP's median (103.0) is WORSE than the leader's (101.0), so a
# procedure that compares point estimates calls it beaten. Both procedures must instead call it unresolved.
PLANTED = "\n".join(
    ['{"rec":"run","bench":"planted","build":"PPU_PACKED_FORMAT=0","reps":3}']
    + [_sample("f", LEADER, i, us) for i, us in enumerate((100.0, 101.0, 102.0))]
    + [_sample("f", OVERLAP, i, us) for i, us in enumerate((101.5, 103.0, 104.0))]
    + [_sample("f", CLEAR, i, us) for i, us in enumerate((300.0, 301.0, 302.0))]
) + "\n"


@pytest.fixture(scope="module")
def xcheck_bin(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no host C++ compiler")
    out = tmp_path_factory.mktemp("xcheck") / "xcheck"
    r = subprocess.run([cxx, "-std=c++17", "-I", str(ROOT / "benchmarks"), str(XCHECK_SRC), "-o", str(out)],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.fail(f"xcheck_select.cpp does not build:\n{r.stderr}")
    return out


def _cpp_verdict(binary, path):
    r = subprocess.run([str(binary), str(path)], capture_output=True, text=True, check=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        key, rest = parts[0], (parts[1] if len(parts) > 1 else "")
        if key == "ties":
            out["ties"] = int(rest)
        elif key == "leader":
            out["leader"] = rest
        elif key == "median":
            out["median"] = float(rest)
        elif key == "band":
            out["band"] = [float(x) for x in rest.split()]
        elif key == "tie":
            out.setdefault("tie_names", []).append(rest)
        elif key == "passes":
            out["passes"] = int(rest)
    return out


def _py_verdict(path):
    r = subprocess.run(["python3", str(ANALYSE), str(path), "--json"], capture_output=True, text=True, check=True)
    vs = json.loads(r.stdout)
    assert len(vs) == 1, "the planted file has one fixture"
    return vs[0]


def test_cpp_and_python_selection_agree(tmp_path, xcheck_bin):
    p = tmp_path / "planted.jsonl"
    p.write_text(PLANTED)
    c, y = _cpp_verdict(xcheck_bin, p), _py_verdict(p)

    assert c["leader"] == y["leader"], "the two procedures disagree about which candidate leads"
    assert c["median"] == pytest.approx(y["median"]), "medians differ"
    assert c["band"] == pytest.approx(y["band"]), "bands differ"
    assert c["ties"] == len(y["ties"]), "tie COUNT differs"
    assert sorted(c.get("tie_names", [])) == sorted(t["config"] for t in y["ties"]), "tie MEMBERSHIP differs"
    assert c["passes"] == y["passes"]


def test_the_fixture_actually_exercises_the_boundary(tmp_path, xcheck_bin):
    """A cross-check over data where nothing ties would pass with both procedures broken in the same direction.
    So assert the planted file produces exactly one tie, and that it is the candidate with the WORSE median --
    the case a point-estimate comparison gets wrong."""
    p = tmp_path / "planted.jsonl"
    p.write_text(PLANTED)
    y = _py_verdict(p)
    assert y["leader"].startswith("i4 64x128")
    assert len(y["ties"]) == 1, "the boundary candidate must tie, or this test proves nothing"
    tie = y["ties"][0]
    assert tie["config"].startswith("i4 32x64")
    assert tie["median"] > y["median"], "the tie must have the worse median, or it is not the interesting case"


def test_one_pass_is_not_a_ranking(tmp_path):
    p = tmp_path / "one.jsonl"
    p.write_text('{"rec":"run","bench":"planted","build":"b","reps":1}\n'
                 + _sample("f", LEADER, 0, 100.0) + "\n"
                 + _sample("f", OVERLAP, 0, 200.0) + "\n")
    assert _py_verdict(p)["ranked"] is False


def test_bc_request_is_a_config_axis_but_bc_eff_is_export_only(tmp_path, xcheck_bin):
    """bc0/bc1 are different candidates even when the collective rejects bc1 and both execute the same kernel.

    Conversely bc_eff describes what happened; it must not split repeated samples for one requested row.
    """
    lines = ['{"rec":"run","bench":"planted","build":"b","reps":3}']
    for pass_i in range(3):
        lines.append(_sample("f", LEADER, pass_i, 100.0 + pass_i, bc=0, bc_eff=0))
        lines.append(_sample("f", LEADER, pass_i, 200.0 + pass_i, bc=1, bc_eff=pass_i % 2))
    p = tmp_path / "bc.jsonl"
    p.write_text("\n".join(lines) + "\n")
    c, y = _cpp_verdict(xcheck_bin, p), _py_verdict(p)
    assert y["candidates"] == 2, "bc requests collapsed together, or bc_eff incorrectly became a key"
    assert c["leader"] == y["leader"]
    assert y["leader"].endswith("bc0")


# ---------------------------------------------------------------------------------------------------------
# THE FORMAT CONTRACT between bench_samples.hpp and analyse.py. The tests above plant JSON from Python, so they
# check the DECISION and not the WIRE. That gap already bit once: json.dumps' default `": "` produced a file the
# C++ cross-checker silently read as zero samples. So this one writes with the real emitter and reads with the
# real analyser -- nothing between them is hand-written.

EMITTER = r'''
#include "bench_samples.hpp"
int main() {
  bench_samples::run_header("roundtrip", "bits=4 TSK=64", 3);
  const struct { int tm, tn, wm, wn, st; double base; } cs[] = {{64,128,64,64,3,100.0},{32,64,32,32,3,101.5}};
  for (int p = 0; p < 3; ++p) for (auto& c : cs) {
    bench_samples::Sample s{};
    s.fixture = "rt"; s.dist = "dense-v1"; s.schema = "i4";
    s.n = 4096; s.k = 4096; s.gs = 128; s.experts = 0; s.rows = 2048; s.mmax = 2048;
    s.tm = c.tm; s.tn = c.tn; s.tk = 64; s.wm = c.wm; s.wn = c.wn; s.st = c.st;
    s.pass = p; s.us = c.base + p * 1.2;
    bench_samples::emit(s);
  }
  bench_samples::flush();
}
'''


def test_emitter_output_is_readable_by_the_analyser(tmp_path):
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no host C++ compiler")
    src = tmp_path / "emit.cpp"
    src.write_text(EMITTER)
    exe = tmp_path / "emit"
    r = subprocess.run([cxx, "-std=c++17", "-I", str(ROOT / "benchmarks"), str(src), "-o", str(exe)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"bench_samples.hpp does not compile standalone:\n{r.stderr}"

    out = tmp_path / "run.jsonl"
    subprocess.run([str(exe)], env={"BENCH_JSONL": str(out), "PATH": "/usr/bin:/bin"}, check=True)
    assert out.is_file() and out.read_text().count("\n") == 7, "expected a run header and six samples"

    v = _py_verdict(out)
    assert v["passes"] == 3
    assert v["candidates"] == 2
    assert v["leader"].startswith("i4 64x128"), "the analyser must read the emitter's field names"
    # The two candidates were planted to overlap, so a round trip that reports SEPARATED means the analyser is
    # reading numbers it did not get from this file.
    assert len(v["ties"]) == 1


def test_unset_bench_jsonl_writes_nothing(tmp_path):
    """The emission is additive: with BENCH_JSONL unset the bench must behave exactly as before. A header that
    wrote to a default path would change every existing run's side effects."""
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no host C++ compiler")
    src = tmp_path / "emit.cpp"
    src.write_text(EMITTER)
    exe = tmp_path / "emit"
    subprocess.run([cxx, "-std=c++17", "-I", str(ROOT / "benchmarks"), str(src), "-o", str(exe)], check=True)
    before = set(p.name for p in tmp_path.iterdir())
    subprocess.run([str(exe)], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, check=True)
    assert set(p.name for p in tmp_path.iterdir()) == before, "something was written with BENCH_JSONL unset"
