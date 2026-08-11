#!/usr/bin/env bash
# WATCH WHAT CODEX IS ACTUALLY DOING, live, in its own terminal.
#
#   tools/watch_codex.sh              follow the newest session
#   tools/watch_codex.sh -n 200       replay the last 200 events first, then follow
#   tools/watch_codex.sh -f           include reasoning (very verbose)
#   tools/watch_codex.sh <threadId>   a specific session
#
# WHY THIS EXISTS AND NOT `codex` ITSELF. The MCP call is the only channel between claude and codex, and it has
# been observed to COMPLETE ON CODEX'S SIDE AND NEVER DELIVER: turn 019fee47 reached `task_complete` at 01:02:30
# with a full answer that the caller never received, and was then killed as a suspected hang. The rollout JSONL
# is written by codex regardless, so it is the honest source of truth for "is it working" and "what did it say".
#
# READ `task_complete` RATHER THAN CPU. A codex process at 0% is not idle -- it can be waiting on sub-agents
# (`sub_agent_activity` below) or on a model response. The only reliable "this turn is finished" signal is a
# `task_complete` event, and its `last_agent_message` holds the full reply even when MCP dropped it.
set -Eeuo pipefail

FULL=0; TAIL=0; THREAD=""
while [ $# -gt 0 ]; do
  case "$1" in
    -f|--full) FULL=1; shift ;;
    -n) TAIL="$2"; shift 2 ;;
    *)  THREAD="$1"; shift ;;
  esac
done

if [ -n "$THREAD" ]; then
  S=$(ls -t "$HOME"/.codex/sessions/*/*/*/rollout-*"$THREAD"*.jsonl 2>/dev/null | head -1)
else
  S=$(ls -t "$HOME"/.codex/sessions/*/*/*/rollout-*.jsonl 2>/dev/null | head -1)
fi
[ -n "${S:-}" ] || { echo "no codex session found under ~/.codex/sessions" >&2; exit 1; }
echo "== $S" >&2
echo "== $(stat -c '%s bytes, last written %y' "$S")" >&2

# `tail -f` from the end by default; -n replays that many lines first.  Python does the rendering because one
# event is one very long JSON line and the interesting fields are nested.
tail -n "${TAIL}" -f "$S" | FULL=$FULL python3 -u -c '
import sys, os, json, textwrap

FULL = os.environ.get("FULL") == "1"
C = {"you": "\033[36m", "codex": "\033[32m", "tool": "\033[33m",
     "sub": "\033[35m", "turn": "\033[1;37m", "off": "\033[0m"}

OFF = C["off"]

def show(kind, head, body="", width=110):
    c = C.get(kind, "")
    print(c + head + OFF)
    if body:
        for line in body.rstrip().splitlines():
            for w in textwrap.wrap(line, width) or [""]:
                print(f"    {w}")

for line in sys.stdin:
    try:
        d = json.loads(line)
    except Exception:
        continue
    p = d.get("payload") or {}
    t, ts = p.get("type"), d.get("timestamp", "")[11:19]

    if t == "task_started":
        show("turn", f"\n=== {ts}  TURN START  {p.get('turn_id','')[:18]}")
    elif t == "task_complete":
        show("turn", f"=== {ts}  TURN COMPLETE  {p.get('turn_id','')[:18]}")
        show("codex", "--- final answer ---", p.get("last_agent_message", ""))
    elif t == "user_message":
        # THIS IS WHAT CLAUDE SENT. The whole point of the window.
        show("you", f"{ts}  >>> SENT BY CLAUDE", p.get("message", ""))
    elif t == "agent_message" and d.get("type") == "event_msg":
        show("codex", f"{ts}  <<< CODEX", p.get("message", ""))
    elif t == "sub_agent_activity":
        show("sub", f"{ts}  ~~~ sub-agent  {json.dumps(p, ensure_ascii=False)[:160]}")
    elif t in ("custom_tool_call", "function_call"):
        name = p.get("name") or p.get("tool_name") or "?"
        arg = json.dumps(p.get("arguments") or p.get("input") or "", ensure_ascii=False)
        show("tool", f"{ts}  [tool] {name}  {arg[:180]}")
    elif t == "reasoning":
        # A HEARTBEAT, NOT SILENCE. Codex spends most of a long turn here, and a window that renders nothing
        # during it is indistinguishable from a dead one -- which is how a working call got killed as a hang.
        if FULL:
            show("sub", ts + "  (reasoning)", json.dumps(p, ensure_ascii=False)[:600])
        else:
            txt = json.dumps(p, ensure_ascii=False)
            print(C["sub"] + ts + "  . thinking  " + txt[:90].replace(chr(10), " ") + OFF)
'
