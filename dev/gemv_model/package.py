#!/usr/bin/env python3
"""Copy only verified runtime/inspection payloads into a new artifact path."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.gemv_model.run import verify
from quactlize.runtime.compiler import sha


def package(source, destination, cohort_name='model'):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    manifest = verify(source,cohort_name)
    if "native-inspection.json" not in manifest["payloads"]:
        raise ValueError("native inspection missing")
    destination.mkdir(parents=True, exist_ok=False)
    for file, digest in manifest["payloads"].items():
        shutil.copy2(source / file, destination / file)
        if sha(destination / file) != digest:
            raise ValueError("copied artifact differs: " + file)
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    verify(destination,cohort_name)
    print(json.dumps(dict(path=str(destination), manifest_sha256=sha(destination / "manifest.json"),
                          payloads=len(manifest["payloads"]),
                          bytes=sum((destination / p).stat().st_size for p in manifest["payloads"]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument('--cohort',choices=('model','tp2'),default='model')
    args = parser.parse_args()
    package(args.source, args.destination,args.cohort)
