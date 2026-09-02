from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compose_kpack_discovery_bundles as compose  # noqa: E402


def test_compose_kpack_discovery_bundles_self_test() -> None:
    compose.self_test()
