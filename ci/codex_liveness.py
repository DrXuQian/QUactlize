#!/usr/bin/env python3
"""IS CODEX WORKING, WAITING, OR HUNG -- decided from outside codex, because a hung codex reports nothing.

WHY HEARTBEATS CANNOT ANSWER THIS. .coord/STATUS.md carries `last-heartbeat`, written by codex at its own
checkpoints. That makes progress visible while codex is healthy, and says exactly nothing while it is not: a
process blocked mid-turn writes no checkpoint, so the instrument goes silent in precisely the case it exists to
detect. On 2026-08-04 that cost 95 minutes -- STATUS said 047, four dispatched items sat unread, and "reading a
lot of source" and "dead" produced identical evidence.

WHAT DOES ANSWER IT, and neither needs codex's cooperation:

  1. CPU TIME. /proc/<pid>/stat fields 14+15 (utime+stime) sampled twice. Advancing means it is computing.
     Frozen means it is not -- which is normal for a process waiting on a reply, and is the first half only.

  2. THE SOCKET'S TIMER. A TCP connection in ESTABLISHED with empty send and receive queues AND timer_active
     == 0 has nothing pending and nothing scheduled: no retransmit, no keepalive, no zero-window probe. The
     kernel has no reason to ever wake the reader again. If the far end vanished silently, this socket stays
     ESTABLISHED forever and the read() behind it never returns. That is a hang, and it is distinguishable from
     waiting -- a normal in-flight request has bytes queued, or a timer armed, or both.

The states this reports:

     COMPUTING   CPU advanced within one window
     WAITING     CPU frozen this window, but a connection has queued bytes or an armed timer -- OR every window
                 looked frozen while cumulative CPU still moved across them
     IDLE        no established connection: between calls, not mid-request
     HUNG        CPU has not moved AT ALL across `confirm` windows, and every connection is idle and untimered

AND THE CORRECTION THAT MATTERS MOST HERE. The first version of this file called HUNG on ONE frozen window, and
that is wrong: the codex mcp-server was measured accumulating ~4.5 seconds of CPU across twenty minutes of a
LIVE call. It wakes per streamed chunk and sleeps between them, so a five-second window landing in a gap sees
exactly what death looks like. Persistence across windows, plus cumulative CPU over the whole span, is what
separates them -- a gap cannot survive the span, because the next chunk lands inside it. The eager version
reported HUNG against a healthy session within minutes of being written.

A SECOND DEFECT WORTH NAMING, because it is funnier and just as fatal: matching "codex" anywhere in the cmdline
made this script match ITSELF -- python3 .../ci/codex_liveness.py -- so its own advancing CPU counted as
evidence that the thing it was probing was alive. A liveness check that cannot fail.

AND THE SETTING THAT MAKES A REAL HANG PERMANENT. CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=0 was set deliberately, to
stop long codex calls being killed at the 30-minute default. The cost is that a genuinely hung call is never
reaped either: 0 does not mean "be patient", it means "no detector". A bounded value longer than the longest
legitimate silence keeps the patience and restores the detector.

    python3 ci/codex_liveness.py                    # one verdict; exit 3 on HUNG
    python3 ci/codex_liveness.py --confirm 5        # stricter: five frozen windows before HUNG
    python3 ci/codex_liveness.py --watch            # sample until the state changes
"""
import argparse
import os
import pathlib
import sys
import time



PROC = pathlib.Path("/proc")
PID = os.getpid()


# The names the codex distribution actually runs. Matching the WHOLE cmdline for "codex" instead caught this
# script itself -- python3 .../ci/codex_liveness.py contains the word -- and a probe that counts its own CPU as
# evidence of the thing it is probing is a liveness check that cannot fail. It also caught the shell that
# launched it. Neither displaced a real process here, but the top-3 cut means either could have.
CODEX_ARGV0 = {"codex", "codex-code-mode-host", "codex-exec"}


def codex_pids():
    """The codex processes, most-CPU first."""
    me = {str(pathlib.Path().resolve()), str(pathlib.Path(__file__).name)}
    out = []
    for p in PROC.iterdir():
        if not p.name.isdigit() or int(p.name) == PID:
            continue
        try:
            raw = (p / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        argv = raw.split("\x00")
        if not argv or not argv[0]:
            continue
        name = argv[0].rsplit("/", 1)[-1]
        # The node wrapper is `node /usr/bin/codex mcp-server`: argv[0] is node, argv[1] names codex. Accept it,
        # but never accept a process merely MENTIONING codex in a later argument -- that is this script.
        if name not in CODEX_ARGV0 and not (name.startswith("node") and len(argv) > 1
                                            and argv[1].rsplit("/", 1)[-1] in CODEX_ARGV0):
            continue
        if any(m in raw for m in ("codex_liveness", "local_gates")):
            continue
        try:
            f = (p / "stat").read_text().split()
            out.append((int(p.name), int(f[13]) + int(f[14]), name))
        except (OSError, IndexError, ValueError):
            continue
    return sorted(out, key=lambda t: -t[1])


def ticks(pid):
    try:
        f = (PROC / str(pid) / "stat").read_text().split()
        return int(f[13]) + int(f[14])
    except (OSError, IndexError, ValueError):
        return None


def sockets(pid):
    """-> [ {state, txq, rxq, timer, retr, remote} ] for this process's TCP fds.

    /proc/<pid>/net/tcp is the whole network namespace, so it is filtered by the inodes the process actually
    holds -- otherwise every unrelated connection in the container would count as evidence of life."""
    try:
        inodes = set()
        for fd in (PROC / str(pid) / "fd").iterdir():
            try:
                t = fd.readlink() if hasattr(fd, "readlink") else pathlib.Path(fd).resolve()
            except OSError:
                continue
            s = str(t)
            if s.startswith("socket:["):
                inodes.add(s[8:-1])
    except OSError:
        return []
    out = []
    for name in ("net/tcp", "net/tcp6"):
        try:
            lines = (PROC / str(pid) / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for l in lines:
            f = l.split()
            if len(f) < 10 or f[9] not in inodes:
                continue
            tx, rx = f[4].split(":")
            tmr, when = f[5].split(":")
            out.append(dict(state=f[3], txq=int(tx, 16), rxq=int(rx, 16),
                            timer=int(tmr, 16), when=int(when, 16), retr=int(f[6], 16), remote=f[2]))
    return out


def _snapshot(watched, sample_seconds):
    before = {p: ticks(p) for p in watched}
    time.sleep(sample_seconds)
    after = {p: ticks(p) for p in watched}
    moved = {p: (after[p] or 0) - (before[p] or 0) for p in watched}
    live, idle = [], []
    for p in watched:
        for s in sockets(p):
            if s["state"] != "01":                    # only ESTABLISHED can be the one being read
                continue
            (live if (s["txq"] or s["rxq"] or s["timer"]) else idle).append(dict(s, pid=p))
    return moved, live, idle, after


def verdict(sample_seconds=6.0, confirm=3):
    """HUNG REQUIRES PERSISTENCE, and this is the correction that matters most in this file.

    A single frozen sample does NOT mean hung. Measured on 2026-08-04: the codex mcp-server accumulated ~4.5
    seconds of CPU across twenty minutes of a live call -- it wakes to process each streamed chunk and sleeps
    between them. A five-second window landing in one of those gaps sees zero ticks and an idle socket, which is
    indistinguishable from death at that timescale. The first version of this file said HUNG on exactly that
    evidence and would have had me killing healthy sessions.

    So HUNG is only reported when the frozen-and-untimered state holds across `confirm` separate windows AND the
    process's cumulative CPU has not moved at all between the first and last -- the second condition being the
    one a between-chunks gap cannot satisfy, because the next chunk lands inside the span."""
    pids = codex_pids()
    if not pids:
        return "ABSENT", "no codex process is running", {}
    watched = [p for p, _, _ in pids[:3]]
    detail = {}
    first_total = {p: ticks(p) for p in watched}
    for attempt in range(max(1, confirm)):
        moved, live, idle, totals = _snapshot(watched, sample_seconds)
        detail = dict(pids=[dict(pid=p, name=n, delta_ticks=moved.get(p, 0)) for p, _, n in pids[:3]],
                      established=len(live) + len(idle), idle_no_timer=len(idle),
                      windows=attempt + 1, window_seconds=sample_seconds)
        if any(v > 0 for v in moved.values()):
            return "COMPUTING", f"CPU advanced over {sample_seconds:.0f}s ({max(moved.values())} ticks)", detail
        if live:
            return "WAITING", (f"CPU frozen this window, but {len(live)} connection(s) have queued bytes or an "
                               f"armed timer -- a reply is in flight"), detail
        if not idle:
            return "IDLE", "no established connection -- codex is between calls, not mid-request", detail
        # frozen + untimered: keep looking rather than concluding.

    span = sum((totals.get(p) or 0) - (first_total.get(p) or 0) for p in watched)
    detail["cumulative_ticks_over_span"] = span
    if span > 0:
        return "WAITING", (f"every window looked frozen, but cumulative CPU moved {span} ticks across them -- "
                           f"this is a process waking between streamed chunks, not a dead one"), detail
    return "HUNG", (f"CPU has not moved at all across {confirm} windows of {sample_seconds:.0f}s, and all "
                    f"{len(idle)} established connection(s) are idle with no timer armed. Nothing will wake the "
                    f"reader"), detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true", help="sample until the verdict changes")
    ap.add_argument("--seconds", type=float, default=6.0, help="one CPU sampling window")
    ap.add_argument("--confirm", type=int, default=3,
                    help="consecutive frozen windows required before saying HUNG (1 cannot tell a "
                         "between-chunks gap from a dead session)")
    a = ap.parse_args()

    st, why, detail = verdict(a.seconds, a.confirm)
    print(f"{st}: {why}")
    for p in detail.get("pids", []):
        print(f"    pid {p['pid']:<8} {p['name']:<22} +{p['delta_ticks']} ticks")
    if "established" in detail:
        print(f"    established connections: {detail['established']}, "
              f"of which idle with no timer: {detail['idle_no_timer']}")
    if a.watch:
        print("\nwatching for a change...")
        while True:
            s2, w2, _ = verdict(a.seconds, a.confirm)
            if s2 != st:
                print(f"CHANGED: {st} -> {s2}: {w2}")
                st = s2
                if s2 in ("HUNG", "ABSENT", "IDLE"):
                    break
    return 3 if st == "HUNG" else 0


if __name__ == "__main__":
    sys.exit(main())
