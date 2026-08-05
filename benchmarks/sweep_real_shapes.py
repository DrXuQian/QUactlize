#!/usr/bin/env python3
"""Run the real-model sweep: every projection of every target model, at every token count, resumably.

    python3 benchmarks/sweep_real_shapes.py --bin <path> --jsonl run.jsonl            # dense
    python3 benchmarks/sweep_real_shapes.py --bin <path> --jsonl run.jsonl --kind moe
    python3 benchmarks/sweep_real_shapes.py --jsonl run.jsonl --dry-run               # what it WOULD run

WHY A DRIVER AND NOT A SHELL LOOP OVER `fixtures.py --emit`.

  1. ONE PROCESS PER INVOCATION IS THE BLAST-RADIUS BOUND. A device assert takes the whole process, so a single
     in-process sweep over every shape loses everything after the first bad config. codex put it exactly: per-
     sample flushing "cannot make successors run -- a poisoned context still requires restarting the process. An
     external per-candidate process driver can complete the remainder." This is that driver. One shape dying
     costs that shape.
  2. RESUME, because the dense list alone is 66 invocations x 293 compiled configs ~ 19,000 timings, and nobody
     restarts that from zero to recover one crash.
  3. The shapes are IMPORTED from fixtures.py, which derives them from workloads.py, which read them off the
     models' config.json. Nothing here transcribes a number -- a second spelling of a model shape is a second
     thing to be wrong, and this repo has paid for that pattern twice this week already.

WHAT COUNTS AS DONE, and why it is not just "exit 0". A child can exit cleanly having written nothing -- the
obvious way is BENCH_JSONL not reaching it. So an invocation counts as complete only when it returned zero AND
the sample file grew. Recording "done" for a run that measured nothing would make resume skip it forever, which
is the silent-loss shape this file exists to avoid.

THE PROGRESS LOG IS THE DRIVER'S OWN, deliberately separate from the sample file. The sample file is
measurements; this is a log of what was launched and what happened to it. Mixing them would put a
non-measurement record into a file whose whole contract is "every line is a measurement" -- and analyse.py would
have to learn a record type that says nothing about performance.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fixtures import DEFAULT_GS, dedup, fixtures            # noqa: E402
from workloads import MODELS                                # noqa: E402


def invocations(kind: str, gs: int, model_filter: str):
    """-> [(label, argv_tail)] in a FIXED order, so resume and a fresh run agree on what 'the third one' is."""
    out = []
    for model, cfg in MODELS.items():
        if model_filter and model_filter != model:
            continue
        for _, label, n, k, t, extra in dedup([f for f in fixtures(model, cfg) if f[0] == kind]):
            if kind == "moe":
                # Positional, and the order is the MoE bench's: L Rows N K gs mode topk. The bench derives Mmax
                # and the expert histogram from the pinned router mode; nothing here asserts rows/expert.
                argv = [str(extra["experts"]), str(extra["tokens"]), str(n), str(k),
                        str(gs), str(extra["mode"]), str(extra["topk"])]
            else:
                argv = [f"--m={t}", f"--n={n}", f"--k={k}", f"--g={gs}", "--search_configs"]
            out.append((f"{label} m={t}", argv))
    return out


def load_done(progress: pathlib.Path) -> set:
    """-> the set of argv tuples recorded complete. Unreadable or absent means nothing is done, which re-runs
    work rather than skipping it: the safe direction when the progress log itself is in doubt."""
    done = set()
    if not progress.is_file():
        return done
    for line in progress.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("rec") == "inv" and r.get("done"):
            done.add(tuple(r.get("argv", [])))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", help="the bench binary (build.sh prints it)")
    ap.add_argument("--jsonl", required=True, help="sample file; passed to the bench as BENCH_JSONL and appended to")
    ap.add_argument("--kind", default="dense", choices=("dense", "moe"))
    ap.add_argument("--gs", type=int, default=DEFAULT_GS["i4"])
    ap.add_argument("--model", default="", help="restrict to one model from workloads.MODELS")
    ap.add_argument("--progress", default="", help="driver log (default: <jsonl>.progress)")
    ap.add_argument("--reps", type=int, default=1, help="BENCH_REPS for each invocation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-run invocations a previous pass recorded as FAILED (default: only never-run ones)")
    a = ap.parse_args()

    jsonl = pathlib.Path(a.jsonl).resolve()
    progress = pathlib.Path(a.progress).resolve() if a.progress else pathlib.Path(str(jsonl) + ".progress")
    todo = invocations(a.kind, a.gs, a.model)
    done = set() if a.retry_failed and not progress.is_file() else load_done(progress)

    pending = [(lbl, argv) for lbl, argv in todo if tuple(argv) not in done]
    print(f"[sweep] {a.kind}: {len(todo)} invocation(s), {len(todo) - len(pending)} already complete, "
          f"{len(pending)} to run")
    print(f"[sweep] samples -> {jsonl}")
    print(f"[sweep] progress -> {progress}")

    if a.dry_run:
        for lbl, argv in pending:
            print(f"  {lbl:<34} {' '.join(argv)}")
        print("\n--dry-run: nothing launched")
        return 0
    if not a.bin:
        print("[sweep] --bin is required unless --dry-run", file=sys.stderr)
        return 2
    binary = pathlib.Path(a.bin).resolve()
    if not binary.is_file():
        print(f"[sweep] no such binary: {binary}", file=sys.stderr)
        return 2

    env = dict(os.environ, BENCH_JSONL=str(jsonl), BENCH_REPS=str(a.reps))
    failed = []
    with progress.open("a") as plog:
        for i, (lbl, argv) in enumerate(pending, 1):
            before = jsonl.stat().st_size if jsonl.is_file() else 0
            # WRITTEN BEFORE THE LAUNCH, same reason bench_samples::attempt() is: if this process is killed with
            # the child, the log still names what was in flight.
            plog.write(json.dumps({"rec": "inv", "label": lbl, "argv": argv, "done": False}) + "\n")
            plog.flush()
            print(f"\n[sweep] {i}/{len(pending)}  {lbl}\n         {binary.name} {' '.join(argv)}", flush=True)
            t0 = time.time()
            rc = subprocess.run([str(binary), *argv], env=env).returncode
            dt = time.time() - t0
            after = jsonl.stat().st_size if jsonl.is_file() else 0
            grew = after > before
            ok = rc == 0 and grew
            plog.write(json.dumps({"rec": "inv", "label": lbl, "argv": argv, "done": ok,
                                   "rc": rc, "grew": grew, "seconds": round(dt, 1)}) + "\n")
            plog.flush()
            if not ok:
                why = f"rc={rc}" if rc else "exited 0 but wrote no samples"
                failed.append((lbl, why))
                print(f"[sweep] ✗ {lbl}: {why} -- continuing; this shape is lost, the rest are not", flush=True)

    print(f"\n[sweep] {len(pending) - len(failed)}/{len(pending)} completed")
    if failed:
        print(f"[sweep] {len(failed)} FAILED:")
        for lbl, why in failed:
            print(f"    {lbl}: {why}")
        print("[sweep] Re-run this command to retry only the incomplete ones. For the config that killed a")
        print("        crashed invocation, run analyse.py over the sample file: an attempt with no matching")
        print("        sample is the row, and it can be reproduced with --config.")
    print(f"[sweep] next: python3 benchmarks/analyse.py {jsonl} --coverage")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
