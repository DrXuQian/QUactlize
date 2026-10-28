"""Independent product numerical-test fixture."""
import hashlib
import json
import pathlib
from typing import Any
ROOT = pathlib.Path(__file__).resolve().parents[2]


from quactlize.gguf_roles import InventoryError
ROUTING_FIXTURE = "token-topk-hot16x4-wor-sm64-s44-v1"


ROUTING_SEED = 0x51554143544C0044


ROUTING_SOURCE = ROOT / "ci/support/moe_router_fixture.hpp"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _splitmix64(state: int) -> tuple[int, int]:
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return state, value & mask


def _routing_fixture(experts: int, top_k: int, tokens: int) -> dict[str, Any]:
    """Bit-for-bit Python transcription of moe_router_fixture.hpp v1."""
    if experts <= 0 or tokens <= 0 or top_k <= 0 or top_k > experts:
        raise InventoryError(
            f"invalid routing fixture E={experts} top_k={top_k} tokens={tokens}")
    counts = [0] * experts
    state = ROUTING_SEED
    hot_experts = min(16, experts)
    token_routes: list[list[int]] = []
    for _token in range(tokens):
        selected = [False] * experts
        remaining_weight = hot_experts * 4 + (experts - hot_experts)
        picks: list[int] = []
        for _pick in range(top_k):
            state, random_value = _splitmix64(state)
            lottery = random_value % remaining_weight
            chosen = -1
            for expert in range(experts):
                if selected[expert]:
                    continue
                weight = 4 if expert < hot_experts else 1
                if lottery < weight:
                    chosen = expert
                    break
                lottery -= weight
            if chosen < 0:
                raise InventoryError("versioned routing fixture failed to choose an expert")
            selected[chosen] = True
            counts[chosen] += 1
            picks.append(chosen)
            remaining_weight -= 4 if chosen < hot_experts else 1
        if len(picks) != len(set(picks)):
            raise InventoryError("versioned routing fixture selected one expert twice for a token")
        token_routes.append(picks)
    row_offsets = [0]
    for rows in counts:
        row_offsets.append(row_offsets[-1] + rows)
    fixture = {
        "fixture": ROUTING_FIXTURE,
        "fixture_source": str(ROUTING_SOURCE.relative_to(ROOT)),
        "fixture_source_sha256": sha256_file(ROUTING_SOURCE),
        "seed": f"0x{ROUTING_SEED:016x}",
        "experts": experts,
        "top_k": top_k,
        "tokens": tokens,
        "total_rows": tokens * top_k,
        "active": sum(rows > 0 for rows in counts),
        "zero": sum(rows == 0 for rows in counts),
        "min_rows": min(counts),
        "max_rows": max(counts),
        "rows_per_expert": counts,
        "row_offsets": row_offsets,
        "token_routes_sha256": _identity_sha256(token_routes),
        "row_offsets_sha256": _identity_sha256(row_offsets),
    }
    _validate_routing_fixture(fixture)
    return fixture


def _validate_routing_fixture(fixture: dict[str, Any]) -> None:
    experts = fixture["experts"]
    top_k = fixture["top_k"]
    tokens = fixture["tokens"]
    rows = fixture["rows_per_expert"]
    offsets = fixture["row_offsets"]
    if len(rows) != experts or len(offsets) != experts + 1:
        raise InventoryError("routing fixture rows/offsets cardinality mismatch")
    if offsets[0] != 0 or any(offsets[i + 1] - offsets[i] != rows[i]
                              for i in range(experts)):
        raise InventoryError("routing fixture row_offsets do not prefix-sum rows_per_expert")
    if offsets[-1] != tokens * top_k or fixture["total_rows"] != offsets[-1]:
        raise InventoryError("routing fixture total is not tokens*top_k")
    if fixture["active"] != sum(value > 0 for value in rows):
        raise InventoryError("routing fixture active count does not match its histogram")
    if fixture["row_offsets_sha256"] != _identity_sha256(offsets):
        raise InventoryError("routing fixture row_offsets hash mismatch")
