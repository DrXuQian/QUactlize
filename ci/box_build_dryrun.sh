#!/usr/bin/env bash
# THE BOX'S BUILD, LOCALLY, UP TO THE COMPILE ITSELF -- with a stub PPU SDK.
#
# WHY THIS EXISTS, AND WHY THE OTHER CHECKS COULD NOT REPLACE IT. build.sh does five things before hgcc runs:
# overlays the sources, REGISTERS the example in actlize's foreach list, configures cmake, lets cmake
# add_subdirectory OUR CMakeLists, and creates the targets. Every local check written so far verified the first,
# third and fourth in isolation and none of them verified the second:
#
#   * dev/fold_derivation/overlay_targets_check.py runs cmake on the overlay DIRECTLY. It proves our CMakeLists
#     configures, and is structurally blind to whether anything REACHES it -- there is no examples/CMakeLists.txt
#     in that scratch tree at all.
#   * a build.sh run against an incomplete stub SDK died inside PPUToolchain's find_library, which happens BEFORE
#     the example is processed. Everything past that point was untested by construction.
#
# So when the registration step was deleted -- by a refactor, three commits before anyone noticed -- both checks
# stayed green while the box reported "No rule to make target test_moe_splitk_bench" alongside "cmake did not
# report PPU_EXTRA_DEFS". Two symptoms of one missing sed, and a diagnosis that reads like macro plumbing.
#
# THE STUB SDK is enough for cmake to configure and for make to reach the REAL HOST LINK: fake hgcc emits a valid
# x86 object for every device TU, generated-unit objects carry an unresolved reference, and the main TU defines it.
# That cross-TU edge is deliberate.  The old fake hgcc exited zero without producing an object, so make failed before
# linking and this gate could never catch the class that produced c96fe8d: a generated per-config TU called a template
# whose definition existed only in the main TU.  We cannot compile PPU instructions locally, but we can and must prove
# that CMake presents every generated object to a linker and that an unresolved cross-TU edge makes that link red.
#
#   ./ci/box_build_dryrun.sh [TARGET] [PPU_DEFS] [NAME=VALUE ...]
#
# Exit 0 = the real build graph reached a host link (or the planted undefined reference was rejected there).
# Exit 1 = one of those claims did not hold. Exit 2 = the stub could not be built here, which is a skip, not a pass.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-test_moe_splitk_bench}"
if [ "$#" -gt 0 ]; then shift; fi
DEFS="${1:-SK_QUANT=2}"
if [ "$#" -gt 0 ]; then shift; fi
BUILD_ENV=("$@")
EXPECT_LINK_FAILURE=0
for _entry in "${BUILD_ENV[@]}"; do
  [[ "$_entry" =~ ^[A-Z][A-Z0-9_]*= ]] || {
    echo "  [FAIL] box_build_dryrun: extra build input is not NAME=VALUE: $_entry"; exit 1; }
  case "$_entry" in
    BOX_DRYRUN_EXPECT_LINK_FAILURE=0) EXPECT_LINK_FAILURE=0 ;;
    BOX_DRYRUN_EXPECT_LINK_FAILURE=1) EXPECT_LINK_FAILURE=1 ;;
    BOX_DRYRUN_EXPECT_LINK_FAILURE=*)
      echo "  [FAIL] box_build_dryrun: BOX_DRYRUN_EXPECT_LINK_FAILURE must be 0 or 1"; exit 1 ;;
  esac
done
# ALWAYS A PRIVATE TEMPORARY DIRECTORY. This used to accept BOX_DRYRUN_SDK from the caller and then rm -rf it --
# pointed at a real SDK or at the repository, that deletes it. A fixed path also races between concurrent runs.
SDK="$(mktemp -d)"
BUILDDIR="$(mktemp -d)"
SENTINEL="$(mktemp -u)"
LOG="$(mktemp)"
cleanup() { rm -rf "$SDK" "$BUILDDIR"; rm -f "$SENTINEL" "$LOG"; }
trap cleanup EXIT

command -v gcc >/dev/null 2>&1 || { echo "  [SKIP] box_build_dryrun: no gcc to build the stub SDK"; exit 2; }
command -v cmake >/dev/null 2>&1 || { echo "  [SKIP] box_build_dryrun: no cmake"; exit 2; }

# --- the stub SDK ------------------------------------------------------------------------------------------------
mkdir -p "$SDK/targets/x86_64-linux/include" "$SDK/include" "$SDK/lib" "$SDK/bin"
# THE STUB hgcc RECORDS THAT IT RAN, and on which file, then emits a REAL host object.  A generated unit owns an
# undefined reference to qz_boxdry_generated_unit_anchor; a non-unit/main TU defines that anchor and main().  Thus the
# final executable can exist only if the generated units and a main object reach one host link.  The negative-control
# mode deliberately omits the anchor and must fail naming that exact symbol.
{
  printf '%s\n' '#!/bin/sh' 'set -eu'
  printf 'sentinel=%s\n' "$SENTINEL"
  printf '%s\n' \
    'printf "%s\n" "$*" >> "$sentinel"' \
    'out=""; src=""; previous=""' \
    'for argument in "$@"; do' \
    '  [ "$previous" = "-o" ] && out="$argument"' \
    '  [ "$previous" = "-c" ] && src="$argument"' \
    '  previous="$argument"' \
    'done' \
    '[ -n "$out" ] && [ -n "$src" ] || { echo "boxdry hgcc: missing -c/-o" >&2; exit 2; }' \
    'body="${out}.boxdry.c"' \
    'base=${src##*/}' \
    'case "$base" in' \
    '  *unit*.cu)' \
    '    printf "%s\n" "extern void qz_boxdry_generated_unit_anchor(void);" "static void (*volatile qz_boxdry_keep_anchor)(void) = qz_boxdry_generated_unit_anchor;" > "$body"' \
    '    ;;' \
    '  *)' \
    '    printf "%s\n" "int __attribute__((weak)) main(void) { return 0; }" > "$body"' \
    '    if [ "${BOX_DRYRUN_EXPECT_LINK_FAILURE:-0}" != 1 ]; then' \
    '      printf "%s\n" "void __attribute__((weak)) qz_boxdry_generated_unit_anchor(void) {}" >> "$body"' \
    '    fi' \
    '    ;;' \
    'esac' \
    'gcc -x c -c "$body" -o "$out"'
} > "$SDK/bin/hgcc"
chmod +x "$SDK/bin/hgcc"
_c="$SDK/stub.c"
for _l in hg_wrapper hggc_wrapper hggcrt1 hggc hgrtc; do
  printf 'void _quactlize_stub_%s(void){}\n' "$_l" > "$_c"
  gcc -shared -fPIC -o "$SDK/lib/lib$_l.so" "$_c" 2>/dev/null || {
    echo "  [SKIP] box_build_dryrun: cannot build stub lib$_l.so"; exit 2; }
done
# Keep this tiny source inside the private SDK until the EXIT cleanup.  Besides
# avoiding a second cleanup path, preserving it makes a failed dry-run fully
# inspectable while the script is still running.

# --- the real build.sh, exactly as the box runs it, but writing NOWHERE the box would ------------------------------
# PPU_BUILD_DIR keeps this out of build_ppu. Without it, "checking the build" DELETED
# the real build tree on every run -- on the box, someone's working build. JOBS=1 so the sentinel's contents are a
# sequence rather than an interleaving.
( cd "$ROOT" && env "${BUILD_ENV[@]}" PPU_SDK="$SDK" PPU_BUILD_DIR="$BUILDDIR" JOBS=1 PPU_DEFS="$DEFS" TARGET="$TARGET" ./build.sh ) >"$LOG" 2>&1
rc=$?

fail() { echo "  [FAIL] box_build_dryrun: $1"; echo "         last lines of the build log:"; tail -12 "$LOG" | sed 's/^/           /'; exit 1; }

# 1. cmake must have PROCESSED OUR CMakeLists. This message comes from it and from nowhere else, so its absence
#    means the example was never add_subdirectory'd -- the exact failure this check was written for.
grep -q "PPU_EXTRA_DEFS ->" "$LOG" || fail "cmake never reported PPU_EXTRA_DEFS -- our CMakeLists was not reached, so the example is not registered in actlize's foreach list"

# 2. the target must EXIST in the real build system, not in a scratch tree with stubbed helpers.
BM="$(find "$BUILDDIR" -path "*${TARGET}.dir/build.make" 2>/dev/null | head -1)"
[ -n "$BM" ] || fail "no build.make for $TARGET -- cmake configured but the target was not created"

# 3. the defines must reach the DEVICE compile command. build.sh checks this too, but only if the build gets far
#    enough to; here it is checked directly off the generated makefile.
# A WHOLE ARGUMENT, not a substring: -DSK_QUANT=2 must not be satisfied by -DSK_QUANT=20.
for _d in $DEFS; do
  _esc="$(printf '%s' "$_d" | sed 's/[][\.^$*+?(){}|]/\\&/g')"
  grep -qE -- "(^|[[:space:]])-D$_esc([[:space:]]|\$)" "$BM" || fail "-D$_d is not a whole argument on $TARGET's compile command"
done

# 3b. Forwarded MoE cache inputs must be acknowledged by build.sh, and MOE_STAGES must become exact device
#     preprocessor arguments after surviving env -> shell array -> CMake list expansion.
for _entry in "${BUILD_ENV[@]}"; do
  _name="${_entry%%=*}"
  _value="${_entry#*=}"
  case "$_name" in
    MOE_*) grep -qF "[build.sh] $_name=$_value" "$LOG" || fail "$_name was not forwarded by build.sh" ;;
  esac
  if [ "$_name" = MOE_STAGES ]; then
    _old_ifs="$IFS"; IFS=';'
    for _stage in $_value; do
      _esc="MOE_STAGES_${_stage}"
      grep -qE -- "(^|[[:space:]])-D${_esc}([[:space:]]|\$)" "$BM" || fail "-D$_esc is not a whole argument on $TARGET's compile command"
    done
    IFS="$_old_ifs"
  fi
done

# 4. THE STUB hgcc MUST HAVE RUN. This is the first half of the evidence; the linked binary below is the second.
[ -s "$SENTINEL" ] || fail "the stub hgcc never ran -- the build stopped before compiling anything, so registration and target creation are the only things this proves"
_n="$(wc -l < "$SENTINEL")"
_unit_n="$(grep -c 'unit[^ ]*\.cu' "$SENTINEL" || true)"

# 5. LINK IS A VERDICT, not an expected casualty of the stub.  The planted mode is the negative arm: it is valid only
#    for a generated-unit target and only when the real linker names the deliberately absent cross-TU anchor.
if [ "$EXPECT_LINK_FAILURE" = 1 ]; then
  [ "$_unit_n" -gt 0 ] || fail "link-failure control selected a target with no generated unit"
  [ "$rc" -ne 0 ] || fail "link-failure control unexpectedly produced a binary"
  grep -q 'qz_boxdry_generated_unit_anchor' "$LOG" || \
    fail "planted generated-unit link failed for the wrong reason (missing anchor was not diagnosed)"
  echo "  [ok]   box_build_dryrun: planted generated-unit undefined reference reached the real host linker and was rejected"
  exit 0
fi

[ "$rc" -eq 0 ] || fail "compile objects were emitted but the real host link failed"
BIN="$(find "$BUILDDIR" -type f -name "$TARGET" -perm -u+x -print -quit 2>/dev/null)"
[ -n "$BIN" ] || fail "build returned success but no linked $TARGET executable exists"
"$BIN" >/dev/null 2>&1 || fail "the linked $TARGET executable does not start successfully against the stub SDK"
if [ "$_unit_n" -gt 0 ]; then
  command -v nm >/dev/null 2>&1 || { echo "  [SKIP] box_build_dryrun: nm is required to inspect the generated-unit link seam"; exit 2; }
  nm "$BIN" > "$BUILDDIR/boxdry.nm" || fail "nm could not inspect the linked generated-unit target"
  grep -q 'qz_boxdry_generated_unit_anchor' "$BUILDDIR/boxdry.nm" || \
    fail "generated-unit target linked without retaining its cross-TU anchor"
fi
echo "  [ok]   box_build_dryrun: $TARGET configured, registered, compiled and genuinely linked ($_n hgcc invocations, $_unit_n generated units)"
