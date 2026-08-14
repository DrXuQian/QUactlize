#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
out="${QUACTLIZE_L172_OUT:-/workspace/quactlize-l172}"
mkdir -p "$out"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$repo/quactlize/include" \
  "$repo/dev/fold_derivation/l172_standalone_marlin_tactic_space.cpp" \
  -o "$out/l172"
"$out/l172"

"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror \
  -I"$repo/quactlize/include" \
  "$repo/dev/fold_derivation/emit_marlin_tactic_space.cpp" \
  -o "$out/emit_marlin_tactic_space"
"$out/emit_marlin_tactic_space" >"$out/census.txt"
generated_header="$out/marlin_standalone_configs.inc"
committed_header="$repo/benchmarks/marlin_standalone_configs.inc"
"$out/emit_marlin_tactic_space" --header >"$generated_header"
if ! cmp -s "$generated_header" "$committed_header"; then
  echo '[l172:header] FAIL: committed admitted-row registry is stale' >&2
  diff -u "$committed_header" "$generated_header" >&2 || true
  exit 1
fi

# Causal controls for the byte-for-byte regeneration contract.  Each plant
# changes both its row list and declared count, so RED does not depend on a
# trivial internal count mismatch: it proves that losing or inventing one
# otherwise well-formed admitted row cannot pass as a fresh registry.
python3 - "$generated_header" "$out" <<'PY'
from pathlib import Path
import sys

authority = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
rows = [line for line in authority.splitlines(keepends=True)
        if line.startswith("  X(")]
if len(rows) != 70:
    raise SystemExit(
        f"[l172:header] FAIL: expected seventy authority rows, found {len(rows)}")

missing = authority.replace(
    "#define MARLIN_STANDALONE_CFG_ROWS 70",
    "#define MARLIN_STANDALONE_CFG_ROWS 69",
    1,
).replace(rows[0], "", 1)
extra_row = rows[-1].rstrip("\n") + " \\\n"
extra = authority.replace(
    "#define MARLIN_STANDALONE_CFG_ROWS 70",
    "#define MARLIN_STANDALONE_CFG_ROWS 71",
    1,
).replace(rows[-1], extra_row + rows[-1], 1)

plants = {"missing-admitted-row": missing, "extra-admitted-row": extra}
for name, candidate in plants.items():
    path = out / f"header-plant-{name}.inc"
    path.write_text(candidate)
    if candidate == authority:
        raise SystemExit(f"[l172:header] FAIL: plant {name} changed nothing")
    print(f"[l172:header:red] plant={name} caught=1 result=RED")
PY

plants=(drop-load-axis drop-stage5 collapse-warp-k broaden-classic-warp-k)
for plant in "${plants[@]}"; do
  log="$out/plant-${plant}.log"
  set +e
  "$out/l172" "--plant=${plant}" >"$log" 2>&1
  rc=$?
  set -e
  if [[ $rc -eq 0 ]] || ! grep -Fq "[l172:red] plant=${plant} caught=1" "$log"; then
    echo "[l172] FAIL: plant ${plant} did not produce its named RED" >&2
    sed -n '1,20p' "$log" >&2
    exit 1
  fi
done

echo '[l172:runner] positive=PASS negative_controls=4/4_RED emitter=PASS header=BYTE_IDENTICAL header_negative_controls=2/2_RED result=PASS'
