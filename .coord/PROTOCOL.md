# How Claude and codex work together

Written 2026-08-03 after two failures of the same shape: instructions sat unread for an hour, and an
unblocking piece of evidence sat unread through three heartbeats. Neither was a disagreement. Both were the
same structural fact -- **Claude is notified of codex; codex is not notified of Claude** -- and the cost was
paid in polling: every "how is codex doing" meant grepping a log and correlating it against git.

## The three files

| file | written by | read by | shape |
|---|---|---|---|
| `INBOX.md`  | Claude | codex | numbered items, append-only, never rewritten |
| `STATUS.md` | codex  | Claude, the user | OVERWRITTEN each time, always current, never a history |
| `BOX.md`    | either | the user | commands that need ppu001, batched |

`codex_gemv_progress.log` stays as the narrative heartbeat. It is a diary, not a channel.

## The four rules

-1. **codex IS A REQUEST/RESPONSE WORKER, NOT A RESIDENT PEER**, and this file was built on the wrong assumption
   for a whole day. Each `mcp__codex__codex` call is ONE TURN: it works, returns a summary, and ends. Between
   turns it is not running, so anything written to INBOX.md after a turn ends is unread BY CONSTRUCTION until the
   next invocation. That is not flakiness. Of the four sessions on 2026-08-03, exactly one died -- an MCP
   1800s idle kill -- and the other three RETURNED COMPLETION SUMMARIES. "codex stopped again" was, three times
   out of four, "codex finished".

   Two consequences, and the second is the one that was being paid for daily:

   * **The PROMPT must be self-contained.** INBOX.md is the durable record and the place to put things between
     turns; it is not a mailbox that gets delivered. Every invocation has to carry what that turn needs. This
     was already happening by accident -- each prompt restated the INBOX items -- which is the only reason the
     arrangement worked at all.
   * **Continue a thread with `codex-reply` and its threadId, do not start a new one.** A fresh `codex` call
     loses everything: the formats, the measurements, the corrections it made an hour ago. Starting fresh four
     times meant explaining Q3/Q5/Q6 and the five-format matrix four times, and paying for codex to rebuild
     context it already had. threadIds seen so far: 019fc758-5e8c-79a0-8ff7-ffc0747a947a,
     019fca98-bf43-72d2-b0fb-8572779d4ad2. RECORD THE threadId FROM EVERY COMPLETION.

0. **STATUS.md carries a WALL-CLOCK timestamp**, not just a heartbeat number. `last-heartbeat: 082` says what
   codex last did; it does not say whether that was a minute ago or an hour. Three times on 2026-08-03 codex's
   session simply ENDED and instructions sat unread — and the monitor could not show it, because a monitor
   reports what someone said, never that they stopped. `inbox-consumed` made unread instructions visible;
   this makes an absent agent visible. Both failures look identical without it.

1. **codex re-reads INBOX.md at every checkpoint** -- before starting a queue item, and after every commit --
   and stamps the highest item number it has consumed into STATUS.md. That stamp is the whole point: Claude can
   then SEE whether the latest instruction has landed instead of inferring it from behaviour.

2. **Claude writes to INBOX at checkpoints, not on realisation.** Twelve of forty-four dispatches on 2026-08-03
   were retractions of the previous one. Most would have been caught by waiting for the next checkpoint and
   re-reading before sending. A correction that arrives after the work has started costs more than a slower
   instruction.

3. **Nobody blocks on the box.** ppu001 is reachable only by the user. Anything needing it goes into BOX.md as a
   runnable command with the exact output required, and work continues on something else. Blocking silently on
   evidence that cannot arrive without a third party is how an hour disappears.

4. **Ownership is by artifact, not by task.** codex: kernels and kernel-adjacent C++/CUDA, plus schemes.py's
   cells and notes. Claude: tests/, ci/, benchmarks/, docs/, pyproject.toml, quactlize/*.py. Need a change on
   the other side of the line? Put it in the other's file. This one has held since it was stated -- there have
   been no collisions -- so it is recorded rather than revisited.

## STATUS.md is answerable in one read

The question "how is codex doing" must not require reading a log. STATUS.md answers it: what is being worked
on, which INBOX item that is, what is blocked and on whom, and the last commit. If it is stale, that is itself
the answer.
