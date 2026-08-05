#!/usr/bin/env python3
"""Every path docs/BACKTEST.md reports a number for must be callable from the shared library.

WHY THIS EXISTS, and it is one mistake made twice in one day rather than a hypothetical.

  * The BACKTEST header claimed the .so ships only `fully_quantized`, and concluded that reproducing the 65%
    dense figure "says nothing about what will ship". The library exports `quactlize_ppu_dense_lowbit*` too --
    scale_first, dense -- so section A WAS a shipping measurement and the whole back-test was mis-ranked
    against a sentence nobody had checked.
  * Corrected that, and immediately made the mirror error on the MoE side: reasoned about section C's 33%
    without looking, and would have called it fully_quantized. It is scale_first (the bench's LOWBIT_QMODE_SEL
    is QM::FinegrainedScaleOnly) -- but there is NO grouped scale_first export, so section C reports numbers for
    a path the product cannot call at all. The measured MoE path and the shippable MoE path do not overlap.

Both errors have one cause: nothing connects "what we measured" to "what we can call". The benches invoke the
collective directly; the .so is a separate surface; and the only link was a human remembering to grep. So this
gate makes the link mechanical.

HOW A SECTION DECLARES ITSELF. docs/BACKTEST.md carries an HTML comment per section:

    <!-- route: dense_lowbit -->            one route
    <!-- route: grouped_lowbit, grouped_fully_quantized -->
    <!-- route: none -- <reason> -->        deliberately not a shipping path, with the reason REQUIRED

`none` is not an escape hatch, it is a statement: section D is GEMV timings taken through a bench that has no
library entry, and saying so is the point. What must not exist is a section that quotes a figure and declares
nothing, because that is exactly the state both errors above were made from.

THE ROUTES COME FROM THE SOURCE, never from a list here. They are the `extern "C" quactlize_ppu_*` symbols in
quactlize/csrc/device/, with the ABI suffixes stripped -- so adding an export makes a route legal here with no
second edit, and removing one fails the section that depended on it.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKTEST = ROOT / "docs" / "BACKTEST.md"
DEVICE = ROOT / "quactlize" / "csrc" / "device"

# quactlize_ppu_<route>[_<abi suffix>]. The suffixes are the ABI's, not the route's: two entries differing only
# by _dev_v2 are one path with two calling conventions, and a section that cites the path should not have to
# know which conventions exist.
SUFFIXES = ("_config_valid_v1", "_config_v1", "_workspace_bytes_v1", "_dev_v1", "_dev_v2")
SYMBOL = re.compile(r"\bquactlize_ppu_([a-z0-9_]+)")
DECL = re.compile(r"<!--\s*route:\s*(.+?)\s*-->")
SECTION = re.compile(r"^##\s+(.+)$", re.M)


def exported_routes() -> set:
    out = set()
    for src in sorted(DEVICE.glob("*.cu")) + sorted(DEVICE.glob("*.cpp")):
        for m in SYMBOL.finditer(src.read_text()):
            name = m.group(1)
            for s in SUFFIXES:
                if name.endswith(s):
                    name = name[: -len(s)]
                    break
            out.add(name)
    return out


def sections(text: str):
    """-> [(title, declared_routes_or_None, reason)] for every '## ' section, in order."""
    marks = [(m.start(), m.group(1).strip()) for m in SECTION.finditer(text)]
    out = []
    for i, (pos, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end]
        d = DECL.search(body)
        if not d:
            out.append((title, None, ""))
            continue
        raw = d.group(1)
        if raw.startswith("none"):
            out.append((title, set(), raw[4:].lstrip(" -")))
        else:
            out.append((title, {r.strip() for r in raw.split(",") if r.strip()}, ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the exported routes and exit")
    a = ap.parse_args()

    routes = exported_routes()
    if not routes:
        print(f"[measured-ships] ERROR: no quactlize_ppu_* symbols found under {DEVICE} -- the parser is wrong, "
              f"not the library")
        return 2
    if a.list:
        for r in sorted(routes):
            print(f"  {r}")
        return 0
    if not BACKTEST.is_file():
        print(f"[measured-ships] ERROR: {BACKTEST} is missing")
        return 2

    problems, declared_none = [], []
    for title, decl, reason in sections(BACKTEST.read_text()):
        if decl is None:
            problems.append(f"section '{title}' declares no route. Add `<!-- route: ... -->` or "
                            f"`<!-- route: none -- why -->`; a section that quotes a figure and says nothing "
                            f"about whether it ships is how this file was misread twice.")
            continue
        if not decl:
            if not reason:
                problems.append(f"section '{title}' declares `none` with no reason. `none` is a claim and needs one.")
            else:
                declared_none.append((title, reason))
            continue
        for r in sorted(decl):
            if r not in routes:
                problems.append(f"section '{title}' reports numbers for route '{r}', which the library DOES NOT "
                                f"export. Either the path is unreachable from the product -- say so in the "
                                f"section -- or the export is missing. Exported: {', '.join(sorted(routes))}")

    for title, reason in declared_none:
        print(f"[measured-ships] not a shipping path, by declaration: '{title}' -- {reason}")
    if problems:
        print(f"\n[measured-ships] FAIL -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  * {p}")
        return 1
    print(f"[measured-ships] PASS -- every section's declared route is exported "
          f"({len(routes)} route(s): {', '.join(sorted(routes))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
