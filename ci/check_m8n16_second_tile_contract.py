#!/usr/bin/env python3
"""Source contract for #112's grouped TM8 second-tile device bisection."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests/test_ppu_m8n16_collective.cu"
RUNNER = ROOT / "tools/run_m8n16_112_box.sh"


def code_only(text: str) -> str:
    """Remove C++ comments before token checks so comments cannot satisfy them."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def audit(harness: str, runner: str) -> list[str]:
    bad: list[str] = []
    source = code_only(harness)
    shell = compact(runner)
    source_tokens = (
        "constexprintkMaxG4M=17;",
        "std::vector<half_t>a(std::size_t(kMaxG4M)*kK);",
        "constexprintkTagK=33;",
        "replay_first_tile&&m>=8?m%8:m",
        "autoblock=cute::make_coord(tile_m,0,cute::_,0);",
        "output[global_m*kN+int(cute::get<1>(coord))]=accum(i);",
        "EpiloguePtrArraySimtVectorized",
        "args.ptr_D=dPtrs.get();",
        "args.dD=dStrides.get();",
        "ReplayFirstTile&&tile_m>0?local_m:global_m",
        "epilogue(problem,Tile{},block,accum,tiled_mma,residue,tid,",
        "intconstrow8_red=half_row_bitdiff(replay.logical,golden,8);",
        "half_row_bitdiff(replay.logical,golden,16):-1;",
        "constexprintMs[]={1,2,3,7,8,9,15,16,17};",
        "errors+=run_g3_a_tag(M,dB,dScale,dZero,q,scales,zeros);",
        "errors+=run_g4_epilogue_tag(M);",
        "typenameContract::ElementB,void,false,false,false,0,UsePersistent>",
        "run_g4_arm<8,8,true>(",
        '"m8p",M,dDenseA.get(),dB.get(),dScale.get(),dZero.get());',
        'G4m8p-vs-m8M=%dbitdiff=%d/%zu%s\\n',
        "if(second_tile_only)",
        "M=9/15/16/17seams=mainloop-A+ptr-array-epilogue+",
        "nonpersistent+persistent==\\n",
    )
    for token in source_tokens:
        if source.count(token) != 1:
            bad.append(f"harness must contain exactly one {token!r}")

    early_match = re.search(r"if\(second_tile_only\)\{(.*?)\}", source)
    early = early_match.start() if early_match else -1
    g5 = source.find("for(introws:{1,8})")
    if early < 0 or g5 < 0 or early > g5:
        bad.append("--second-tile-only does not return before the expensive G5 sweep")
    elif "returnerrors?1:0;" not in early_match.group(1):
        bad.append("--second-tile-only branch does not return before G5")

    runner_tokens = (
        "check_m8n16_second_tile_contract.py",
        '"$BIN"--second-tile-only',
        "formin9151617;do",
        "G3A-TAGM=${m}raw-bitdiff=0/${count}MATCH",
        "G3A-TAG-NEGATIVEM=${m}replay-oracle-bitdiff=0/${count}",
        "G4EPILOGUE-TAGM=${m}raw-bitdiff=0/${count}MATCH",
        "G4EPILOGUE-TAG-NEGATIVEM=${m}observed-red=${red}",
        "G4m8p+M=${m}goldenbad=0/${count}",
        "G4m8p-vs-m8M=${m}bitdiff=0/${count}MATCH",
        "\\[112:SECOND-TILE\\]PASS:errors=0M=9/15/16/17seams="
        "mainloop-A+ptr-array-epilogue+nonpersistent+persistent",
    )
    for token in runner_tokens:
        if shell.count(token) != 1:
            bad.append(f"runner must contain exactly one {token!r}")
    if source.count("intconstexpected_red=(M-8)*kN;") != 2:
        bad.append("A and epilogue negatives must each bind the exact red denominator")
    return bad


def main() -> int:
    harness = HARNESS.read_text()
    runner = RUNNER.read_text()
    bad = audit(harness, runner)
    if bad:
        print("[m8n16-second-tile] FAIL: " + "; ".join(bad))
        return 1

    plants = (
        ("harness", "cute::make_coord(tile_m, 0, cute::_, 0)",
         "cute::make_coord(0, 0, cute::_, 0)", "A tile coordinate"),
        ("harness", "replay_first_tile && m >= 8 ? m % 8 : m",
         "m % 8", "A replay negative"),
        ("harness", "ReplayFirstTile && tile_m > 0 ? local_m : global_m",
         "local_m", "epilogue absolute coordinate"),
        ("harness",
         "int const row8_red = half_row_bitdiff(replay.logical, golden, 8);",
         "int const row8_red = 0;", "exact row-8 red witness"),
        ("harness", "if (second_tile_only) {",
         "if (false && second_tile_only) {", "G5 early return"),
        ("harness", "return errors ? 1 : 0;\n  }\n\n  // G5 runs",
         "// planted missing return\n  }\n\n  // G5 runs",
         "G5 branch return"),
        ("harness", "auto m8p = run_g4_arm<8, 8, true>(",
         "auto m8p = run_g4_arm<8, 8, false>(",
         "persistent composed arm"),
        ("runner", '"$BIN" --second-tile-only', '"$BIN"',
         "focused invocation"),
    )
    for target, old, new, label in plants:
        source = harness if target == "harness" else runner
        if source.count(old) != 1:
            print(f"[m8n16-second-tile] FAIL: cannot plant {label}")
            return 1
        planted_harness = source.replace(old, new, 1) if target == "harness" else harness
        planted_runner = source.replace(old, new, 1) if target == "runner" else runner
        if not audit(planted_harness, planted_runner):
            print(f"[m8n16-second-tile] FAIL: checker accepted planted {label}")
            return 1

    # A correct token preserved only in a comment must not rescue a real
    # coordinate regression.
    comment_bypass = harness.replace(
        "cute::make_coord(tile_m, 0, cute::_, 0)",
        "cute::make_coord(0, 0, cute::_, 0) /* "
        "cute::make_coord(tile_m, 0, cute::_, 0) */", 1)
    if not audit(comment_bypass, runner):
        print("[m8n16-second-tile] FAIL: comments can satisfy a source token")
        return 1

    print("[m8n16-second-tile] PASS M=9/15/16/17; exact A/epilogue tags; "
          "persistent composed arm; nine source negatives RED; focused mode "
          "returns before G5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
