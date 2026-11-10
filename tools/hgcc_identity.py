#!/usr/bin/env python3
"""Read HGGC identity without assuming one SDK's version-option spelling."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def compiler_identity(compiler, env=None):
    compiler = Path(compiler).absolute()
    digest = hashlib.sha256(compiler.read_bytes()).hexdigest()
    probes = []
    # -V is accepted by HGGC SDKs that reject the long --version spelling.
    # Do not use -v: it enables verbose compilation and requires an input.
    for flag in ("-V", "--version"):
        argv = [str(compiler), flag]
        try:
            result = subprocess.run(
                argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", timeout=10, check=False,
            )
            output = result.stdout.strip()
            probes.append(dict(argv=argv, returncode=result.returncode, output=output))
            if (result.returncode == 0 and re.search(r"\bhgcc\b", output, re.I)
                    and re.search(r"\b(?:version|release)\s+\d", output, re.I)):
                return dict(compiler=str(compiler), sha256=digest, version=output,
                            version_flag=flag, probes=probes)
        except (OSError, subprocess.TimeoutExpired) as exc:
            probes.append(dict(argv=argv, error=str(exc)))
    raise RuntimeError(
        "cannot read an HGGC version banner; compiler_sha256=" + digest
        + "; probes=" + json.dumps(probes, ensure_ascii=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compiler_identity(args.compiler), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
