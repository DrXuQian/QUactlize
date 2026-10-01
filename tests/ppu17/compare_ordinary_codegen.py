"""Compare encoded ordinary SM90a device bodies against the pre-PDL parent.

Only trailing disabled PDL/prefetch policy tags are matched to an existing
parent symbol. Instruction encodings and scheduling words must match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def bodies(inspector, obj):
    text = subprocess.check_output([str(inspector), "--dump-sass", str(obj)], text=True)
    chunks = re.split(r"Function\s*:\s*", text)[1:]
    symbols = [chunk.splitlines()[0].strip() for chunk in chunks]
    names = subprocess.check_output(["c++filt"], input="\n".join(symbols) + "\n", text=True).splitlines()
    result = {}
    for chunk, name in zip(chunks, names):
        if "ACQBULK" in chunk or "PREEXIT" in chunk:
            continue
        # c++filt separates adjacent closing template brackets with spaces.
        name = re.sub(r"\s+(?=>)", "", name)
        words = re.findall(r"/\* 0x([0-9a-f]{16}) \*/", chunk)
        if not words or name in result:
            raise ValueError("missing or duplicate encoded body: " + name)
        result[name] = hashlib.sha256("".join(words).encode()).hexdigest()
    return result


def match_parent(candidate, parent):
    """Accept only known disabled policy additions, never instruction changes."""
    matched = {}
    for name, digest in candidate.items():
        choices = [name]
        # TMA may add Pdl=false followed by PrefetchB=false. Other families
        # add at most Pdl=false. Each normalized name must actually exist in
        # the parent; do not blindly erase dtype/arithmetic boolean tags.
        for _ in range(2 if "::tma_wgmma<" in name else 1):
            shorter, count = re.subn(r", false>\(", ">(", choices[-1], count=1)
            if not count:
                break
            choices.append(shorter)
        present = [choice for choice in choices if choice in parent]
        if len(present) != 1 or present[0] in matched:
            raise ValueError("missing/ambiguous/duplicate parent symbol: " + name)
        matched[present[0]] = digest
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.baseline / "manifest.json").read_text())
    after = json.loads((args.candidate / "manifest.json").read_text())
    if before["target"] != "cuda-source" or after["target"] != "cuda-source":
        raise ValueError("this comparison is CUDA SM90a only")
    if before["compiler_sha256"] != after["compiler_sha256"]:
        raise ValueError("compiler differs")
    rows = []
    for q in (8, 10, 11, 12, 13, 14):
        parent = bodies(args.cuobjdump, args.baseline / f"q{q}.o")
        candidate = match_parent(bodies(args.cuobjdump, args.candidate / f"q{q}.o"), parent)
        if set(parent) != set(candidate):
            raise ValueError("ordinary inventory differs for q=" + str(q))
        expected = 86 if q == 12 else 26
        if len(parent) != expected:
            raise ValueError("incomplete ordinary inventory")
        changed = [name for name in parent if parent[name] != candidate[name]]
        rows.append(dict(q=q, bodies=expected, changed=changed))
    receipt = dict(status="PASS" if not any(r["changed"] for r in rows) else "CHANGED",
                   scope="CUDA_SM90A_ENCODED_ORDINARY_BODIES_NOT_PERFORMANCE",
                   baseline=before["build_key"], candidate=after["build_key"], results=rows)
    with args.output.open("x") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")
    print("PPU17_ORDINARY_CODEGEN " + receipt["status"] + " bodies=216")
    return int(receipt["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
