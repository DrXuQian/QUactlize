#!/usr/bin/env python3
"""Compile a fake runtime around the production GEMV event/JSONL protocol."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
TIMING = ROOT / "benchmarks/gemv_perf_timing.hpp"
SAMPLES = ROOT / "benchmarks/gemv_perf_samples.hpp"
DRIVER = ROOT / "benchmarks/sweep_gemv_perf.py"

FAKE_RUNTIME = r'''#pragma once
typedef int hggcError_t; typedef int hggcEvent_t; typedef int hggcStream_t;
static constexpr int hggcSuccess = 0;
inline hggcError_t hggcEventCreate(hggcEvent_t*) { return 0; }
inline hggcError_t hggcEventDestroy(hggcEvent_t) { return 0; }
inline hggcError_t hggcEventRecord(hggcEvent_t, hggcStream_t=0) { return 0; }
inline hggcError_t hggcEventSynchronize(hggcEvent_t) { return 0; }
inline hggcError_t hggcDeviceSynchronize() { return 0; }
inline hggcError_t hggcEventElapsedTime(float*,hggcEvent_t,hggcEvent_t){return 0;}
inline hggcError_t hggcPeekAtLastError(){return 0;}
inline const char* hggcGetErrorName(hggcError_t){return "fake";}
inline const char* hggcGetErrorString(hggcError_t){return "fake";}
'''

ORACLE = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <string>
#include "benchmarks/gemv_perf_samples.hpp"

struct Fake {
  using Event=int; using Stream=int; using Status=int;
  static inline int creates=0,destroys=0,records=0,event_syncs=0,device_syncs=0;
  static inline int elapsed_calls=0,launches=0; static inline bool inject_zero=false;
  static Status create(Event* e){*e=++creates;return 0;}
  static Status destroy(Event){++destroys;return 0;}
  static Status record(Event,Stream){++records;return 0;}
  static Status synchronize_event(Event){++event_syncs;return 0;}
  static Status synchronize_device(){++device_syncs;return 0;}
  static Status elapsed(float* ms,Event,Event){
    ++elapsed_calls; *ms=(inject_zero&&elapsed_calls==2)?0.0f:0.002048f*elapsed_calls; return 0;}
  static Status launch_status(){return 0;}
  static bool success(Status s){return s==0;}
  static std::string describe(Status){return "fake";}
  static void reset(){creates=destroys=records=event_syncs=device_syncs=elapsed_calls=launches=0;}
};

int main(int argc,char** argv){
  assert(argc==2); Fake::reset();
  auto good=gemv_perf_timing::measure_raw_launches<Fake>([]{++Fake::launches;},3,0);
  assert(good.complete() && good.samples.size()==3);
  assert(Fake::creates==8 && Fake::destroys==8 && Fake::records==8);
  assert(Fake::event_syncs==1 && Fake::device_syncs==1 && Fake::launches==4);
  assert(good.samples[0].event_ms_bits!=0 && good.samples[0].event_us>0);

  gemv_perf_samples::JsonlWriter writer(argv[1]);
  gemv_perf_samples::Run run{"r","build","space",false}; assert(writer.write_run(run));
  gemv_perf_samples::Candidate c{"r","shape",R"({"active":1,"experts":0,"k":2048,"m":1,"n":2048})",
    "int4","cfg",R"({"chunk":2,"cta_m":1,"cta_n":8,"format":"int4","layout":"native","route":"dense","step_k":16,"threads":128,"tile_size_k":0})"};
  gemv_perf_samples::Attempt a{c,"0",0}; assert(writer.write_attempt(a,3));
  assert(writer.write_samples(a,good));
  auto c2=c; c2.config_id="excluded"; c2.config_json=R"({"chunk":4,"cta_m":1,"cta_n":8,"format":"int4","layout":"native","route":"dense","step_k":16,"threads":128,"tile_size_k":0})";
  gemv_perf_samples::Attempt x{c2,"0",0}; assert(writer.write_attempt(x,3));
  assert(writer.write_excluded(x,"shape legality")); assert(writer.ok());

  Fake::reset(); Fake::inject_zero=true;
  auto bad=gemv_perf_timing::measure_raw_launches<Fake>([]{++Fake::launches;},3,0);
  assert(!bad.ok && bad.samples.empty() && !bad.error.empty());
  return 0;
}
'''


def main() -> int:
    missing = [p for p in (TIMING, SAMPLES, DRIVER) if not p.is_file()]
    if missing:
        print("[gemv-event-protocol] FAIL missing " + ", ".join(map(str, missing)))
        return 1
    with tempfile.TemporaryDirectory(prefix="qz-gemv-event-") as td:
        root = Path(td)
        (root / "hggc_runtime.h").write_text(FAKE_RUNTIME)
        source = root / "oracle.cpp"
        source.write_text(ORACLE)
        exe, raw = root / "oracle", root / "raw.jsonl"
        build = subprocess.run(
            ["g++", "-std=c++17", "-I", str(root), "-I", str(ROOT),
             str(source), "-o", str(exe)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            print("[gemv-event-protocol] FAIL compile:\n" + build.stdout)
            return 1
        run = subprocess.run([str(exe), str(raw)], cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if run.returncode:
            print("[gemv-event-protocol] FAIL oracle:\n" + run.stdout)
            return 1
        probe = subprocess.run(
            [sys.executable, "-c",
             "import pathlib,sys;sys.path.insert(0,str(pathlib.Path('.').resolve()));"
             "from benchmarks.sweep_gemv_perf import load_raw_lines;"
             "d=load_raw_lines(pathlib.Path(sys.argv[1]).read_text().splitlines(),sys.argv[1]);"
             "assert len(d.complete_attempts())==2 and len(d.exclusions)==1 and not d.complaints,"
             "(len(d.complete_attempts()),len(d.exclusions),d.complaints)",
             str(raw)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if probe.returncode:
            print("[gemv-event-protocol] FAIL reader:\n" + probe.stdout)
            return 1

    text = TIMING.read_text()
    required = (
        "std::vector<Pair> pairs(measured_launches + 1)",
        "Api::synchronize_event(pairs[0].end)",
        "Api::synchronize_device()",
        "out.samples.clear()",
        "event_ms_bits",
    )
    absent = [x for x in required if x not in text]
    if absent:
        print(f"[gemv-event-protocol] FAIL protocol seams missing: {absent}")
        return 1
    print("[gemv-event-protocol] PASS: warmup+N distinct event pairs; one final sync; "
          "raw float words; incomplete batch excluded; JSONL roundtrip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
