# Repository-owned PPU skills

These development instructions ship with the Quactlize source checkout. They
do not depend on one engineer's global agent directory or private chat memory.

| Skill | Use |
|---|---|
| [ppu-kernel-iteration](ppu-kernel-iteration/SKILL.md) | Resumable overnight task, build/test entrypoints, curated technical handoff |
| [ppu-cold-gemv-tuning](ppu-cold-gemv-tuning/SKILL.md) | Cold-weight performance, address patterns and ACU interpretation |
| [ppu-cute-numeric-debug](ppu-cute-numeric-debug/SKILL.md) | Isolate a failing specialization and prove a semantic fix |
| [ppu-main-productization](ppu-main-productization/SKILL.md) | Explicit product/main admission, not ordinary development experiments |

The PPU adaptation of [KDA](https://github.com/DrXuQian/kda) carries a reviewed,
hash-bound snapshot of these four skills and their references. Quactlize is
the source of truth. See that repository's quickstart to create an isolated
box task, get its overnight prompt and inspect progress.

Update skills here, commit them, then run KDA's
`scripts/sync_quactlize_knowledge.py --quactlize /path/to/quactlize`.
Use `--check` to detect drift. Never export global settings, credentials,
sessions, raw conversations, uploaded models or unreviewed local files.

The active user's task and permission boundaries take precedence over skill
defaults. Read only the skills/references relevant to that task. These files
are development workflow assets and are excluded from product-main curation.
