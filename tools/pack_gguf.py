#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``quactlize-pack-gguf`` command."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from quactlize.pack_gguf import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    raise SystemExit(main())
