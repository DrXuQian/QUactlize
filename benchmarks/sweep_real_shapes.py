#!/usr/bin/env python3
"""Run the real-model sweep: every projection of every target model, at every token count, resumably.

    python3 benchmarks/sweep_real_shapes.py --bin <path> --jsonl run.jsonl            # dense
    python3 benchmarks/sweep_real_shapes.py --bin <path> --jsonl run.jsonl --kind moe
    python3 benchmarks/sweep_real_shapes.py --jsonl run.jsonl --dry-run               # what it WOULD run
    python3 benchmarks/sweep_real_shapes.py --bin <path> --jsonl run.jsonl --devices 0,1,2,3   # four cards at once

WHY A DRIVER AND NOT A SHELL LOOP OVER `fixtures.py --emit`.

  1. ONE PROCESS PER INVOCATION IS THE BLAST-RADIUS BOUND. A device assert takes the whole process, so a single
     in-process sweep over every shape loses everything after the first bad config. codex put it exactly: per-
     sample flushing "cannot make successors run -- a poisoned context still requires restarting the process. An
     external per-candidate process driver can complete the remainder." This is that driver. One shape dying
     costs that shape.
  2. RESUME, because the dense list alone is 66 invocations x 632 compiled configs ~ 42,000 timings, and nobody
     restarts that from zero to recover one crash. (293 was the TileK=64-only table; TileK became a row field on
     2026-08-05 and the count is read off benchmarks/lowbit_dense_configs.inc, not remembered.)
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

N CARDS AT ONCE: --workers N --devices 0,1,2,3. The default is one worker with no device set, which is the old
behaviour unchanged -- every recorded number was taken that way, and a driver that quietly changes what it
measures is worse than a slow one.

  A DYNAMIC QUEUE, NOT A STATIC SHARD. Invocation cost spans more than an order of magnitude (m=1 against
  m=4096, each dense invocation sweeping ~1164 configs), so round-robin sharding ends with three workers idle
  watching the fourth finish its share. Instead there is one shared list and workers pop the next undone item.
  The residual tail is one invocation long, which is the floor for any scheme that does not split an invocation.

  EVERY WORKER GETS ITS OWN SAMPLE FILE -- and NOT for the reason usually given. The usual worry is torn lines
  from concurrent appenders, and that was MEASURED rather than assumed (2026-08-07, this box): 8 processes
  reproducing bench_samples.hpp's writer exactly (fopen(p,"a"), one fprintf per record, fflush after EVERY
  record) wrote 80,000 records to one file with 0 torn lines. The negative control -- the identical program with
  the per-record fflush removed -- tore 3,197 of 80,000. The mechanism is the flush: it empties the buffer, so
  each 225-byte record starts at buffer offset 0, never straddles BUFSIZ (8192 here), and reaches the kernel as
  ONE write() on an O_APPEND fd, which is atomic against the file offset. Drop the flush and records straddle
  the boundary and interleave. So tearing is not the hazard TODAY, and this note exists so nobody "fixes" it
  twice; it WOULD return if a record ever grew past BUFSIZ or a writer stopped flushing per record.

  The decisive reason is THE `grew` RULE. Complete means rc==0 AND the sample file grew. With one shared file
  those bytes are everyone's: an invocation that measured NOTHING sees the file grown by a concurrent worker and
  records itself complete, and resume then skips it forever. That is precisely the silent loss the rule exists
  to prevent, so the rule needs a file that exactly one process can grow. Per-worker files are what make the
  existing rule keep meaning what it says under concurrency.

  RESUME READS ALL OF THEM. `done` is the union over <progress> and every <progress>.w<i>, so a 4-worker run
  resumed at 2 workers -- or at 1, or after --devices changed -- does not redo the other workers' share. The
  glob is anchored to `\\.w\\d+$` rather than `.w*` because the per-worker child-output log is <progress>.w<i>.out
  and it matches `.w*`: feeding kernel stdout to a JSON reader is how a log line becomes a fake completion.

  NEVER ATTEMPTED IS A THIRD OUTCOME, reported separately from completed and from failed. A worker that dies
  takes no coverage with it -- its unpopped items stay in the shared queue for the others -- but if the run is
  interrupted, items nobody popped must not read as either "done" (resume would skip them) or "failed" (which
  claims a launch that never happened). The final line names all three counts every time, including zeros,
  because a count that only appears when it is non-zero is a count nobody checks.
"""
import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fixtures import DEFAULT_GS, dedup, fixtures            # noqa: E402
from workloads import MODELS                                # noqa: E402

# Popped, no result recorded yet. A distinct object and not a string, so it can never collide with a failure
# reason -- failure reasons ARE strings and one of them reading as "still running" is the confusion this whole
# accounting exists to prevent.
INFLIGHT = object()

# `<anything>.w<digits>` and nothing else. See the module docstring: `.w*` also matches `.w0.out`.
WORKER_SUFFIX = re.compile(r"\.w\d+$")


def invocations(kind: str, gs: int, model_filter: str, formats=("",)):
    """-> [(label, argv_tail, fmt)] in a FIXED order, so resume and a fresh run agree on what 'the third one' is.

    FORMAT IS AN AXIS, NOT A PROPERTY OF THE BINARY YOU HAPPENED TO BUILD. The five low-bit formats can no longer
    share one executable -- all of them in one link overflows the small code model -- so a full MoE sweep is
    shapes x formats, with a different binary per format. Crossing it here rather than looping outside keeps ONE
    ordered list, which is what resume, --dry-run and the dynamic queue all read.

    `fmt` is "" for the single-binary case (every dense run, and any MoE run given a plain --bin). That value is
    load-bearing for resume: progress logs written before this axis existed carry no fmt, and `.get("fmt", "")`
    reads them as "" too, so an old log still cancels the work it recorded instead of silently re-running it."""
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
            for fmt in formats:
                out.append((f"{label} m={t}" + (f" [{fmt}]" if fmt else ""), argv, fmt))
    return out


def shard(base: pathlib.Path, w: int, workers: int) -> pathlib.Path:
    """Where worker `w` writes. ONE worker means the unsuffixed path: a lone run must land where it always did,
    or every command in docs/ and .coord/BOX.md that names run.jsonl starts pointing at nothing."""
    return base if workers == 1 else pathlib.Path(f"{base}.w{w}")


def shards_on_disk(base: pathlib.Path):
    """-> [base, base.w0, base.w1, ...], restricted to what EXISTS, for reading back a previous run.

    Every shard of every PREVIOUS worker count, not just this run's: the whole point is that a 4-worker run can
    be resumed at 1 (or 8, or with different cards) without redoing 3/4 of the sweep."""
    found = [base] if base.is_file() else []
    for p in sorted(glob.glob(glob.escape(str(base)) + ".w*")):
        q = pathlib.Path(p)
        if WORKER_SUFFIX.search(q.name) and q.is_file():
            found.append(q)
    return found


def load_done(progress: pathlib.Path) -> set:
    """-> the set of (fmt, argv tuple) recorded complete, UNIONED over the single-worker and per-worker logs.

    THE KEY CARRIES fmt BECAUSE argv CANNOT. A MoE invocation's argv is positional and names the SHAPE only --
    L Rows N K gs mode topk -- while which format ran is a property of the binary. Keying on argv alone would let
    the q3 run cancel the q5 one: four fifths of a sweep skipped, resume reporting it complete, and no error
    anywhere. Old logs have no fmt and read as "", which is exactly the single-binary key.

    Unreadable or absent means nothing is done, which re-runs work rather than skipping it: the safe direction
    when the progress log itself is in doubt. That applies PER SHARD -- one unreadable worker log costs that
    worker's completions and nothing else."""
    done = set()
    for p in shards_on_disk(progress):
        try:
            text = p.read_text()
        except (OSError, UnicodeDecodeError) as e:
            print(f"[sweep] cannot read {p} ({e}) -- treating its work as NOT done, which re-runs it",
                  file=sys.stderr)
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("rec") == "inv" and r.get("done"):
                done.add((r.get("fmt", ""), tuple(r.get("argv", []))))
    return done


def parse_devices(spec: str):
    """'0,1,2,3' -> [0, 1, 2, 3]. Raises ValueError with a usable message on anything else.

    DUPLICATES ARE REFUSED. `--devices 0,0` reads as two-way parallelism and delivers two processes contending
    for one card: half the throughput, no error anywhere, and a sweep that reports it ran on two cards. That is
    the exact failure bench_device.hpp was written to close, arriving through the driver instead."""
    spec = spec.strip()
    if not spec:
        return []
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok.isdigit():                       # rejects "", "-1", "0x1", " 1 2" -- isdigit() is not int()
            raise ValueError(f"--devices: {tok!r} is not a device index; want a comma-separated list like 0,1,2,3")
        out.append(int(tok))
    if len(set(out)) != len(out):
        raise ValueError(f"--devices {spec}: repeats a card. Two workers on one PPU is not two-way parallelism.")
    return out


def summarise(status):
    """-> (completed, failed, never, inflight, unknown) over the queue's status list.

    FIVE buckets for four states plus a catch-all, and the catch-all is the point: a status value nobody
    anticipated must surface as UNKNOWN rather than fall into whichever branch happens to be last. Counting by
    `if/elif/else` over one list is also what makes the buckets add up to len(status) by construction, so the
    report cannot claim more completions than there were items."""
    completed = failed = never = inflight = unknown = 0
    for s in status:
        if s is True:
            completed += 1
        elif s is None:
            never += 1
        elif s is INFLIGHT:
            inflight += 1
        elif isinstance(s, str):
            failed += 1
        else:
            unknown += 1
    return completed, failed, never, inflight, unknown


class WorkQueue:
    """One shared list; workers pop the next item. See the docstring: this is the dynamic queue, not a shard.

    `status[i]` is the accounting, and it has four states because three of them get conflated:
        None      -- NEVER POPPED. Not attempted, not failed, still owed. A resume must run it.
        INFLIGHT  -- popped, no final status. Reaching the REPORT in this state means the wait was cut short
                     (a second interrupt) or a worker died outright; a normal exit leaves none.
        True      -- rc == 0 AND its own sample file grew.
        str       -- failed, and the string says how.
    """

    def __init__(self, items):
        self.items = items
        self.status = [None] * len(items)
        self._next = 0
        self._lock = threading.Lock()
        self.stopping = False

    def pop(self):
        """-> index, or None when the queue is drained or stopping. The lock covers read-and-increment together;
        two workers reading the same index would run one invocation twice and skip another entirely."""
        with self._lock:
            if self.stopping or self._next >= len(self.items):
                return None
            i = self._next
            self._next += 1
            self.status[i] = INFLIGHT
            return i

    def record(self, i, st):
        with self._lock:
            self.status[i] = st

    def stop(self):
        with self._lock:
            self.stopping = True


def run_worker(w, dev, q, binaries, jsonl, progress, workers, reps, say, finished):
    """Pop, launch, judge, log -- until the queue is drained. One of these per card.

    `finished` is this worker's own "I am done" flag, set in the finally below. The interrupt path waits on
    THAT and not on Thread.is_alive(); see the comment there for the run that made the difference visible."""
    sample = shard(jsonl, w, workers)
    plog_path = shard(progress, w, workers)
    env = dict(os.environ, BENCH_JSONL=str(sample), BENCH_REPS=str(reps))
    if dev is not None:
        env["PPU_BENCH_DEVICE"] = str(dev)
    # Empty for a lone unbound worker, so its console output is what it has always been. Present otherwise,
    # because a line that cannot be attributed to a card is the defect bench_device.hpp opens by naming.
    tag = ""
    if workers > 1 or dev is not None:
        tag = f"w{w}" + (f"/dev{dev}" if dev is not None else "") + "  "

    # THE CHILD'S OWN OUTPUT NEEDS A FILE ONCE THERE IS MORE THAN ONE CHILD. Four benches inheriting this
    # terminal interleave line-by-line at the whim of their flushes, and a device assert -- the message that
    # matters most here -- lands in that soup with nothing saying which card produced it. Inherited for a lone
    # worker (unchanged, and it stays live), redirected per worker otherwise. buffering=0 because nothing on the
    # Python side ever writes to this handle; it exists to be the child's fd, and the byte offsets read back on
    # failure must be the child's, not a stale view through a Python buffer.
    child_out = None if workers == 1 else open(str(plog_path) + ".out", "ab", buffering=0)
    try:
        with plog_path.open("a") as plog:
            while True:
                i = q.pop()
                if i is None:
                    return
                lbl, argv, fmt = q.items[i]
                binary = binaries[fmt]
                try:
                    before = sample.stat().st_size if sample.is_file() else 0
                    # WRITTEN BEFORE THE LAUNCH, same reason bench_samples::attempt() is: if this process is
                    # killed with the child, the log still names what was in flight. `worker`/`device` ride
                    # along so a completed record says WHERE it ran; `device` is null when nothing was bound.
                    plog.write(json.dumps({"rec": "inv", "label": lbl, "argv": argv, "fmt": fmt,
                                           "done": False, "worker": w, "device": dev}) + "\n")
                    plog.flush()
                    say(f"\n[sweep] {i + 1}/{len(q.items)}  {tag}{lbl}\n         {binary.name} {' '.join(argv)}")
                    t0 = time.time()
                    out_at = os.path.getsize(str(plog_path) + ".out") if child_out else 0
                    kw = {"stdout": child_out, "stderr": subprocess.STDOUT} if child_out else {}
                    rc = subprocess.run([str(binary), *argv], env=env, **kw).returncode
                    dt = time.time() - t0
                    after = sample.stat().st_size if sample.is_file() else 0
                    grew = after > before
                    ok = rc == 0 and grew
                    plog.write(json.dumps({"rec": "inv", "label": lbl, "argv": argv, "fmt": fmt,
                                           "done": ok, "worker": w, "device": dev,
                                           "rc": rc, "grew": grew, "seconds": round(dt, 1)}) + "\n")
                    plog.flush()
                    # ONE spelling of the reason, used for both the status and the console line. Two spellings
                    # is how the summary and the message that scrolled past come to disagree.
                    why = "" if ok else (f"rc={rc}" if rc else "exited 0 but wrote no samples")
                    q.record(i, True if ok else why)
                    if not ok:
                        msg = f"[sweep] ✗ {tag}{lbl}: {why} -- continuing; this shape is lost, the rest are not"
                        if child_out:
                            # The tail of what THIS child said, from the offset its own launch started at.
                            # Without it the one message that explains the failure is in a file the reader has
                            # not been told about yet.
                            with open(str(plog_path) + ".out", "rb") as fh:
                                fh.seek(out_at)
                                tail = fh.read().decode("utf-8", "replace").splitlines()[-8:]
                            for ln in tail:
                                msg += f"\n         | {ln}"
                            msg += f"\n         (full output: {plog_path}.out)"
                        say(msg)
                except Exception as e:                              # noqa: BLE001
                    # A worker must not take the queue down with it. Anything unexpected here -- the binary
                    # vanished, the log filled the disk -- is this ITEM's failure; the remaining items stay in
                    # the shared queue and the other workers keep taking them.
                    q.record(i, f"driver error: {type(e).__name__}: {e}")
                    say(f"[sweep] ✗ {tag}{lbl}: driver error: {type(e).__name__}: {e} -- continuing")
    finally:
        if child_out:
            child_out.close()
        # LAST, and after every q.record() this worker will ever make. That ordering is the whole guarantee the
        # interrupt path leans on: flag set => this worker's items all have a final status.
        finished.set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", help="the bench binary (build.sh prints it); one format, or dense")
    ap.add_argument("--bin-map", default="",
                    help="per-format binaries for --kind moe: 'q3=PATH,q5=PATH,q6=PATH,i2=PATH,i4=PATH'. "
                         "The five formats no longer fit one executable, so a full MoE sweep needs one binary "
                         "each; the sweep then runs shapes x formats. Mutually exclusive with --bin.")
    ap.add_argument("--jsonl", required=True, help="sample file; passed to the bench as BENCH_JSONL and appended to")
    ap.add_argument("--kind", default="dense", choices=("dense", "moe"))
    ap.add_argument("--gs", type=int, default=DEFAULT_GS["i4"])
    ap.add_argument("--model", default="", help="restrict to one model from workloads.MODELS")
    ap.add_argument("--progress", default="", help="driver log (default: <jsonl>.progress)")
    ap.add_argument("--reps", type=int, default=1, help="BENCH_REPS for each invocation")
    ap.add_argument("--workers", type=int, default=None,
                    help="run this many invocations at once (default: 1, or len(--devices) when that is given)")
    ap.add_argument("--devices", default="",
                    help="PPUs to spread the workers over, e.g. 0,1,2,3; worker i gets devices[i] in "
                         "PPU_BENCH_DEVICE. Unset leaves the device to the runtime, as before.")
    ap.add_argument("--dry-run", action="store_true")
    # NOT A NO-OP THAT LOOKS LIKE ONE, but not what its old help said either. load_done() only ever collects
    # records with done:true, and a failed invocation is written done:false -- so failures have ALWAYS been
    # retried by a plain re-run, with or without this flag, and the old help ("default: only never-run ones")
    # described a behaviour that has never existed. Corrected rather than implemented: retrying what did not
    # finish is what the closing message already tells the reader to do, and changing the default would be a
    # change to what a resume covers. Kept so existing command lines and scripts do not break.
    ap.add_argument("--retry-failed", action="store_true",
                    help="accepted and has no effect: a re-run ALREADY retries everything not recorded complete")
    a = ap.parse_args()

    jsonl = pathlib.Path(a.jsonl).resolve()
    progress = pathlib.Path(a.progress).resolve() if a.progress else pathlib.Path(str(jsonl) + ".progress")

    try:
        devices = parse_devices(a.devices)
    except ValueError as e:
        ap.error(str(e))
    workers = a.workers if a.workers is not None else (len(devices) if devices else 1)
    if workers < 1:
        ap.error(f"--workers {workers}: want at least 1")
    if devices and workers > len(devices):
        # LOUD, not silent. A silent clamp is how a run reports N-way parallelism and delivers less -- and the
        # number the reader remembers is the one they typed.
        print(f"[sweep] --workers {workers} but only {len(devices)} device(s) given: running {len(devices)}. "
              f"Add cards to --devices to go wider.", file=sys.stderr)
        workers = len(devices)
    if workers > 1 and not devices:
        if os.environ.get("PPU_BENCH_DEVICE", "") != "":
            # Refused, not warned: every worker would inherit the SAME card, which is the pile-up bench_device
            # exists to prevent, wearing the disguise of a deliberate setting.
            print(f"[sweep] PPU_BENCH_DEVICE={os.environ['PPU_BENCH_DEVICE']} is set in the environment and "
                  f"--workers is {workers}: all {workers} workers would land on that one card. Pass --devices "
                  f"to spread them, or unset it.", file=sys.stderr)
            return 2
        print(f"[sweep] ⚠ --workers {workers} with no --devices: nothing binds a card, so the runtime picks -- "
              f"probably the same one {workers} times. Pass --devices to spread them.", file=sys.stderr)

    # ONE BINARY OR A MAP, NEVER BOTH -- "both" has no meaning and the two answers would silently disagree about
    # which executable ran the q3 rows.
    if a.bin and a.bin_map:
        print("[sweep] pass --bin OR --bin-map, not both", file=sys.stderr)
        return 2
    binaries, formats = {}, ("",)
    if a.bin_map:
        for tok in a.bin_map.split(","):
            tok = tok.strip()
            if not tok or "=" not in tok:
                print(f"[sweep] --bin-map entry {tok!r} is not FMT=PATH", file=sys.stderr)
                return 2
            f, path = tok.split("=", 1)
            f = f.strip()
            if f in binaries:
                print(f"[sweep] --bin-map names {f} twice", file=sys.stderr)
                return 2
            binaries[f] = pathlib.Path(path.strip()).resolve()
        formats = tuple(binaries)          # the map's ORDER, so --dry-run and the run agree

    todo = invocations(a.kind, a.gs, a.model, formats)
    done = load_done(progress)

    pending = [it for it in todo if (it[2], tuple(it[1])) not in done]
    print(f"[sweep] {a.kind}: {len(todo)} invocation(s), {len(todo) - len(pending)} already complete, "
          f"{len(pending)} to run")
    if workers == 1:
        print(f"[sweep] samples -> {jsonl}")
        print(f"[sweep] progress -> {progress}")
    else:
        print(f"[sweep] samples -> {jsonl}.w0 .. .w{workers - 1}  (one per worker; analyse.py takes them all)")
        print(f"[sweep] progress -> {progress}.w0 .. .w{workers - 1}   child output -> the same, .out")
        print(f"[sweep] {workers} worker(s) over "
              f"{'device(s) ' + ','.join(str(d) for d in devices[:workers]) if devices else 'whatever the runtime picks'}")

    if a.dry_run:
        for j, (lbl, argv, _fmt) in enumerate(pending):
            # ROUND-ROBIN HERE AND NOWHERE ELSE. The real queue is dynamic, so which worker takes an item
            # depends on how long its predecessors ran; this column is what the plan LOOKS like, not a
            # prediction. Absent entirely for a lone unbound worker, so the old plan output is unchanged.
            tag = ""
            if workers > 1 or devices:
                d = devices[j % workers] if devices else None
                tag = f"w{j % workers}" + (f"/dev{d}" if d is not None else "") + "  "
            print(f"  {tag}{lbl:<34} {' '.join(argv)}")
        if workers > 1:
            print("\n  (w/dev above is round-robin: the queue is DYNAMIC, so the worker that actually takes an"
                  " item\n   depends on how long the items before it took. Shape of the plan, not a prediction.)")
        print("\n--dry-run: nothing launched")
        return 0
    if not a.bin and not a.bin_map:
        print("[sweep] --bin or --bin-map is required unless --dry-run", file=sys.stderr)
        return 2
    if a.bin:
        binaries = {"": pathlib.Path(a.bin).resolve()}
    # EVERY mapped binary is checked BEFORE the first launch. Discovering the fifth path is wrong after four
    # formats have swept is the same waste as the link failure that made the map necessary.
    for f, b in sorted(binaries.items()):
        if not b.is_file():
            print(f"[sweep] no such binary{f and ' for ' + f}: {b}", file=sys.stderr)
            return 2

    q = WorkQueue(pending)
    console = threading.Lock()

    def say(s):
        # One write per message, under a lock. print() emits the text and the newline as two writes, which four
        # workers can interleave into a line that names one invocation and the binary of another.
        with console:
            sys.stdout.write(s + "\n")
            sys.stdout.flush()

    sys.stdout.flush()          # the prologue above, ahead of anything a worker flushes
    finished = [threading.Event() for _ in range(workers)]
    threads = [threading.Thread(target=run_worker,
                                args=(w, devices[w] if devices else None, q, binaries, jsonl, progress,
                                      workers, a.reps, say, finished[w]),
                                name=f"w{w}", daemon=True)
               for w in range(workers)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Stop handing out work and let the in-flight children finish being waited on. The items nobody popped
        # stay None and are reported as never attempted -- which is the state a resume needs them in.
        q.stop()
        say("\n[sweep] interrupted: no new invocations will start; waiting for the running ones")
        # WAIT ON THE WORKERS' OWN FLAGS, not on Thread.join()/is_alive(). Measured here 2026-08-07 over three
        # interrupted runs: after the signal, `for t in threads: t.join()` returned -- and so did a follow-up
        # `while any(t.is_alive()): t.join(timeout=1)` -- while one worker was still waiting on its child. Twice
        # out of three the child then finished and wrote its sample, so the file held 23 samples while the
        # summary said 22 completed and called the last one "in flight". Nothing was lost (it is not recorded
        # done, so a resume re-runs it, which is the safe direction) but the REPORT was wrong, and this file's
        # whole job is a report that can be trusted about what was measured. The flag is set by the worker
        # itself, after its last q.record(), so it cannot be early. With it: six interrupted runs, six times
        # `completed` equal to the number of samples on disk and nothing left in flight.
        try:
            for e in finished:
                while not e.wait(timeout=1.0):
                    pass
        except KeyboardInterrupt:
            # A second interrupt is the reader saying they will not wait. Then the items ARE still running, the
            # report says so, and a resume re-runs them.
            say("[sweep] second interrupt: no longer waiting for the running invocation(s)")

    completed, failed_n, never, inflight, unknown = summarise(q.status)
    print(f"\n[sweep] {completed}/{len(pending)} completed, {failed_n} failed, {never} never attempted"
          + (f", {inflight} STILL IN FLIGHT AT EXIT" if inflight else "")
          + (f", {unknown} in an UNKNOWN state" if unknown else ""))
    if never or inflight:
        # Named, not just counted. These are the items a resume has to pick up, and "3 never attempted" with no
        # names is a number the reader has to reconstruct the queue to act on.
        print("[sweep] NOT MEASURED -- these were never launched (or were still running when this exited).")
        print("        They are NOT recorded complete, so re-running this command picks them up:")
        for j, st in enumerate(q.status):
            if st is None or st is INFLIGHT:
                print(f"    {'never attempted' if st is None else 'in flight'}: {pending[j][0]}")
    if failed_n:
        print(f"[sweep] {failed_n} FAILED:")
        for j, st in enumerate(q.status):
            if isinstance(st, str):
                print(f"    {pending[j][0]}: {st}")
        print("[sweep] Re-run this command to retry only the incomplete ones. For the config that killed a")
        print("        crashed invocation, run analyse.py over the sample file: an attempt with no matching")
        print("        sample is the row, and it can be reproduced with --config.")
    # EVERY shard that exists, not the path that was typed. After a 4-worker run `analyse.py run.jsonl` reads a
    # file nobody wrote, reports nothing, and looks like a sweep that measured nothing rather than one whose
    # samples are next door.
    files = shards_on_disk(jsonl)
    print(f"[sweep] next: python3 benchmarks/analyse.py {' '.join(str(f) for f in files) or jsonl} --coverage")
    return 1 if (failed_n or never or inflight or unknown) else 0


if __name__ == "__main__":
    raise SystemExit(main())
