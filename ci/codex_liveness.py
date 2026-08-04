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

The three states this reports are therefore:

     COMPUTING   CPU advancing
     WAITING     CPU frozen, and some connection has queued bytes or an armed timer
     HUNG        CPU frozen, and every connection is idle with no timer armed

AND THE SETTING THAT MAKES THE THIRD STATE PERMANENT. CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT=0 was set deliberately,
to stop long codex calls being killed at the 30-minute default. The cost is that a genuinely hung call is never
reaped either: 0 does not mean "be patient", it means "no detector". A bounded value longer than the longest
legitimate silence keeps the patience and restores the detector.

    python3 ci/codex_liveness.py            # one verdict, exit 0 computing/waiting, 3 hung
    python3 ci/codex_liveness.py --watch    # sample until the state changes
"""
import argparse
import pathlib
import sys
import time

PROC = pathlib.Path("/proc")


def codex_pids():
    """The codex processes, most-CPU first. Matching on the vendored binary name rather than 'codex' avoids
    counting the node wrapper, which never does the work."""
    out = []
    for p in PROC.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if "codex" not in cmd:
            continue
        try:
            f = (p / "stat").read_text().split()
            out.append((int(p.name), int(f[13]) + int(f[14]), cmd.split("\x00")[0].rsplit("/", 1)[-1]))
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


def verdict(sample_seconds=6.0):
    pids = codex_pids()
    if not pids:
        return "ABSENT", "no codex process is running", {}
    watched = [p for p, _, _ in pids[:3]]
    before = {p: ticks(p) for p in watched}
    time.sleep(sample_seconds)
    after = {p: ticks(p) for p in watched}
    moved = {p: (after[p] or 0) - (before[p] or 0) for p in watched}
    detail = dict(pids=[dict(pid=p, name=n, delta_ticks=moved.get(p, 0)) for p, _, n in pids[:3]])

    if any(v > 0 for v in moved.values()):
        return "COMPUTING", f"CPU advanced over {sample_seconds:.0f}s ({max(moved.values())} ticks)", detail

    live, idle = [], []
    for p in watched:
        for s in sockets(p):
            if s["state"] != "01":                    # only ESTABLISHED can be the one being read
                continue
            (live if (s["txq"] or s["rxq"] or s["timer"]) else idle).append(dict(s, pid=p))
    detail["established"] = len(live) + len(idle)
    detail["idle_no_timer"] = len(idle)
    if live:
        return "WAITING", (f"CPU frozen, but {len(live)} connection(s) have queued bytes or an armed timer -- "
                           f"a reply is in flight"), detail
    if idle:
        return "HUNG", (f"CPU frozen AND all {len(idle)} established connection(s) are idle with no timer armed. "
                        f"Nothing will wake the reader; this will not resolve on its own"), detail
    return "IDLE", "CPU frozen and no established connection -- codex is between calls, not mid-request", detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true", help="sample until the verdict changes")
    ap.add_argument("--seconds", type=float, default=6.0, help="CPU sampling window")
    a = ap.parse_args()

    st, why, detail = verdict(a.seconds)
    print(f"{st}: {why}")
    for p in detail.get("pids", []):
        print(f"    pid {p['pid']:<8} {p['name']:<22} +{p['delta_ticks']} ticks")
    if "established" in detail:
        print(f"    established connections: {detail['established']}, "
              f"of which idle with no timer: {detail['idle_no_timer']}")
    if a.watch:
        print("\nwatching for a change...")
        while True:
            s2, w2, _ = verdict(a.seconds)
            if s2 != st:
                print(f"CHANGED: {st} -> {s2}: {w2}")
                st = s2
                if s2 in ("HUNG", "ABSENT", "IDLE"):
                    break
    return 3 if st == "HUNG" else 0


if __name__ == "__main__":
    sys.exit(main())
