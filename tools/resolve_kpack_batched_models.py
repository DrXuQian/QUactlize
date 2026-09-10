#!/usr/bin/env python3
"""Resolve benchmark GGUF paths without reading/hashing model payloads.

Only the selected model directories are searched, including nested split-file
directories. Multiple weight files/families are an error, not a first-file
choice. Vision mmproj files are not text model candidates. An explicit `path`
in a custom plan binds one file or directory instead of `model_root/directory`.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.resolve_internal_sweep_models import resolve_binding, split_group


def resolve_plan(plan, names=None, model_root=None):
    requested = set(names) if names else {m["name"] for m in plan["models"]}
    unknown = requested - {m["name"] for m in plan["models"]}
    if unknown:
        raise ValueError(f"unknown model names: {sorted(unknown)}")
    root = model_root or plan.get("model_root")
    models = []
    for model in plan["models"]:
        if model["name"] not in requested:
            continue
        if model.get("path"):
            source = Path(model["path"])
        else:
            directory = Path(model["directory"])
            if root is None or directory.is_absolute() or ".." in directory.parts:
                raise ValueError(f"{model['name']}: model_root and relative directory required")
            source = Path(root) / directory
        try:
            if source.is_dir():
                files = split_group(p for p in source.rglob("*")
                    if p.is_file() and p.suffix.lower() == ".gguf"
                    and not p.name.lower().startswith("mmproj"))
            else:
                files = resolve_binding(source)
            if len({p.parent for p in files}) != 1:
                raise ValueError("split GGUF files must share one directory")
            for path in files:
                if path.stat().st_size == 0:
                    raise ValueError(f"empty GGUF file: {path}")
        except (OSError, ValueError) as exc:
            raise ValueError(f"{model['name']}: {source}: {exc}; "
                             "use an exact GGUF path in MODEL_PLAN if ambiguous") from exc
        models.append(model | dict(path=str(files[0]), files=[str(p) for p in files]))
    resolved = plan | dict(models=models)
    if root is not None:
        resolved["model_root"] = str(root)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "tools/kpack_batched_models.json")
    parser.add_argument("--model", action="append", help="selected name; repeat or omit for all")
    parser.add_argument("--model-root", type=Path, help="replace the plan's single model root")
    parser.add_argument("--output", type=Path, help="write a fresh resolved plan for benchmark/trace")
    args = parser.parse_args()
    try:
        plan = resolve_plan(json.loads(args.plan.read_text()), args.model, args.model_root)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(plan, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"KPACK_MODEL_PATH FAIL: {exc}\n")
    for model in plan["models"]:
        print(f"KPACK_MODEL_PATH name={model['name']} shards={len(model['files'])} "
              f"path={model['path']}", flush=True)


if __name__ == "__main__":
    main()
