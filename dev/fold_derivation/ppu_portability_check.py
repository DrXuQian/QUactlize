#!/usr/bin/env python3
"""Box-built sources must not name NVIDIA-only identifiers in code the box actually compiles.

WHY THIS EXISTS, AND WHY IT IS NOT A PLAIN GREP. syntax_check.sh reported 0 noise lines for these very files
and the box then failed on `fatal error: cuda_fp16.h: No such file or directory`. It could not have caught it:
local nvcc HAS cuda_fp16.h whether or not -D__HGGCCC__ is passed, so a wrong-platform INCLUDE parses fine and
only the box disagrees. The same blind spot covers cudaStream_t in a signature and __halves2half2, an
intrinsic that exists in CUDA and appears NOWHERE in actlize.

A plain grep is not enough either, because the portable spelling is itself written as
    #if defined(__HGGCCC__) ... #else <the CUDA one> #endif
so the CUDA name is PRESENT in every correctly-ported file. The check therefore evaluates the same conditions
the box does -- __HGGCCC__ defined, ENABLE_BF16 not -- and only reports hits in the live branches. That makes
it exact rather than heuristic, and it is the difference between a check that is trusted and one that gets a
growing exception list until it hides the next real hit.

Identifiers that exist in CUDA *and* in actlize are deliberately NOT on the list: half2, __hfma2, __hsub2,
__hadd2, __hmul2, __half2half2, __shfl_xor_sync, __syncthreads, uint4, uint2, dim3, float2, __float2half and
__half2float were all confirmed present in the actlize include tree, so they are portable.
"""
import os, re, sys

DENY = re.compile(r'cuda_fp16\.h|cuda_bf16\.h|cuda_runtime\.h|cudaStream_t|cudaMalloc|cudaFree|cudaMemcpy'
                  r'|cudaDeviceSynchronize|cudaGetLastError|cudaError_t|cudaSuccess|cudaEvent'
                  r'|__halves2half2|__halves2bfloat162|__nv_bfloat')
# gemv_rt.hpp's whole job is to straddle the two runtimes, so it names both by design.
ALLOW = {'gemv_rt.hpp'}
# THIS IS AN APPLICABILITY BOUNDARY, NOT A PORTABILITY EXCEPTION.  These files form one local RTX5090/sm_120
# experiment.  They are built directly by benchmarks/q4k_pdf_5090_ab.py with nvcc and NVML; they are not sources
# of any PPU CMake target.  Treating their CUDA API as a PPU failure stopped every boxdry check before CMake.
#
# A bare allow-list would be dangerous: if one of these files later became PPU-reachable, the exception would hide
# exactly the regression this check exists to catch.  _nvidia_island_errors() therefore proves both sides on every
# run: the island still has its NVIDIA-only build contract, and no PPU CMake/source edge reaches it.
NVIDIA_ONLY_ISLAND = {
    'benchmarks/q4k_pdf_5090_ab.cu',
    'benchmarks/q4k_pdf_ab_fixture.hpp',
    'benchmarks/q4k_pdf_reconstruction.cuh',
    'dev/fold_derivation/l146_q4k_pdf_ab_fixture.cu',
}
# *_cuda_probe.* is a LOCAL CUDA harness by convention and is excluded from the overlay by build.sh, so it never
# reaches hgcc. Skipping it here is not a weakening: the two rules have to agree, and if a probe loses the suffix it
# starts being overlaid AND starts being reported here at the same moment.
def _is_local_cuda_probe(path):
    import re as _re
    return _re.search(r'_cuda_probe\.[^.]+$', os.path.basename(path)) is not None
# What the box defines. Anything guarded on something else is assumed live (conservative: reports more).
DEFINED = {'__HGGCCC__'}
UNDEFINED = {'ENABLE_BF16'}


def _rel(root, path):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(os.sep, '/')


def _deny_hits(path):
    return [(ln, text) for ln, text in live_lines(path) if DENY.search(text.split('//')[0])]


def _nvidia_island_errors(root, ppu_candidates, cmake_text, runner_text=None):
    """Prove the exact NVIDIA-only island remains N/A to PPU rather than silently exempting it."""
    errors = []
    for rel in sorted(NVIDIA_ONLY_ISLAND):
        if not os.path.isfile(os.path.join(root, rel)):
            errors.append(f"declared NVIDIA-only source vanished: {rel}")

    runner = os.path.join(root, 'benchmarks/q4k_pdf_5090_ab.py')
    if runner_text is None and os.path.isfile(runner):
        runner_text = open(runner, errors='replace').read()
    runner_text = runner_text or ''
    for token in ('nvcc', '-arch=sm_120', '-lnvidia-ml', 'compute capability {cap}'):
        if token not in runner_text:
            errors.append(f"NVIDIA-only Q4_K contract lost {token!r}; its PPU N/A classification is stale")

    # A source named by the PPU CMake authority is PPU-reachable even if the wide source scan still labels it an
    # island.  Basenames are unique in this tree (overlay_targets_check proves that), so a basename is sufficient
    # and also catches both relative and generated absolute spellings.
    for rel in sorted(NVIDIA_ONLY_ISLAND):
        base = os.path.basename(rel)
        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(base) + r'(?![A-Za-z0-9_])', cmake_text):
            errors.append(f"declared NVIDIA-only source became PPU-CMake-reachable: {rel}")

    island_bases = {os.path.basename(p) for p in NVIDIA_ONLY_ISLAND}
    include = re.compile(r'^\s*#\s*include\s*[\"<]([^\">]+)[\">]', re.M)
    for path in ppu_candidates:
        rel = _rel(root, path)
        if rel in NVIDIA_ONLY_ISLAND or not os.path.isfile(path):
            continue
        text = open(path, errors='replace').read()
        reached = sorted({os.path.basename(m.group(1)) for m in include.finditer(text)} & island_bases)
        if reached:
            errors.append(f"PPU candidate {rel} includes NVIDIA-only island member(s): {', '.join(reached)}")
    return errors

def live_lines(path):
    """Yield (lineno, text) for lines the box would compile, honouring #if/#else/#endif on the macros above."""
    stack = []           # each entry: True (live) / False (dead) / None (unknown -> treat as live)
    out = []
    cond = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')
    def truth(expr):
        e = expr.strip()
        for m in DEFINED:
            if f'defined({m})' in e or e == m: return True
        for m in UNDEFINED:
            if f'defined({m})' in e or e == m: return False
        return None
    for i, line in enumerate(open(path, errors='replace'), 1):
        m = cond.match(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ('if', 'ifdef'):
                stack.append(truth(rest.replace('defined', 'defined')) if kind == 'if' else truth(f'defined({rest.strip()})'))
            elif kind == 'ifndef':
                t = truth(f'defined({rest.strip()})')
                stack.append(None if t is None else (not t))
            elif kind == 'elif':
                if stack: stack[-1] = truth(rest)
            elif kind == 'else':
                if stack: stack[-1] = None if stack[-1] is None else (not stack[-1])
            elif kind == 'endif':
                if stack: stack.pop()
            continue
        if all(s is not False for s in stack):
            out.append((i, line.rstrip('\n')))
    return out

def main():
    # WHERE THE SOURCES ARE, told by build.sh so this cannot drift from what the overlay actually ships. The fallback
    # is the repo layout, for running this by hand. It used to be "the directory above this one", which held every
    # source before the tree split into quactlize/include, tests/ and benchmarks/ -- after the split this scanned an
    # empty set, and only the vacuity self-check below made that visible instead of a silent pass.
    root = os.environ.get('QUACTLIZE_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src_dirs = (os.environ.get('QUACTLIZE_SRC_DIRS') or 'quactlize/include tests benchmarks').split()
    files = []
    for d in src_dirs:
        full = os.path.join(root, d)
        if os.path.isdir(full):
            files += [os.path.join(full, f) for f in sorted(os.listdir(full))
                      if f.endswith(('.cu', '.cuh', '.hpp', '.h'))]
    sub = os.environ.get('QUACTLIZE_GEMV_DIR') or os.path.join(root, 'quactlize/include/gemv_lowbit')
    if os.path.isdir(sub):
        files += [os.path.join(sub, f) for f in sorted(os.listdir(sub))
                  if f.endswith(('.cu', '.cuh', '.hpp', '.h'))]
    if not files:
        print("  [FAIL] ppu_portability: scanned no files at all -- the source directories moved")
        return 1

    cmake = os.environ.get('QUACTLIZE_CMAKE') or os.path.join(root, 'quactlize/csrc/CMakeLists.txt.in')
    if not os.path.isfile(cmake):
        print(f"  [FAIL] ppu_portability: PPU CMake source authority is missing: {cmake}")
        return 1
    island_errors = _nvidia_island_errors(root, files, open(cmake, errors='replace').read())
    if island_errors:
        for error in island_errors:
            print(f"  [FAIL] ppu_portability: {error}")
        return 1

    bad = 0
    for f in files:
        if os.path.basename(f) in ALLOW: continue
        if _is_local_cuda_probe(f): continue
        if _rel(root, f) in NVIDIA_ONLY_ISLAND: continue
        for ln, text in _deny_hits(f):
            # `root`, not `here` -- there is no `here`. This line is in the FAILURE path, which had never run,
            # so a NameError sat in the one branch whose whole job is to report a problem: the check could only
            # ever pass or crash. Verified below by making it fire on purpose.
            print(f"  [FAIL] ppu_portability: {os.path.relpath(f, root)}:{ln} is NVIDIA-only in a branch "
                  f"the box compiles:\n           {text.strip()}")
            bad = 1

    # TWO SELF-CHECKS, because a portability check that silently stops matching is worse than none.
    #
    # (1) The deny list must still match REAL code somewhere. gemv_rt.hpp is the file that names both runtimes,
    #     so it is the anchor -- but its cuda* names sit in the #else of #if defined(__HGGCCC__), i.e. in a DEAD
    #     branch, so the anchor has to be checked against the raw text and not against the live subset. Getting
    #     that backwards made this self-check fail on a correctly-ported tree the first time it ran.
    rt = os.path.join(sub, 'gemv_rt.hpp')
    if not (os.path.exists(rt) and DENY.search(open(rt, errors='replace').read())):
        print("  [FAIL] ppu_portability: the deny list matches nothing even in gemv_rt.hpp -- vacuous check")
        return 1
    # (2) An UNCONDITIONAL cuda include in a PPU candidate must be reported.  This is a planted REAL source defect,
    #     not a synthetic status string; it proves the island cannot turn a new PPU regression into N/A/SKIP.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        probe = os.path.join(td, 'probe.cuh')
        open(probe, 'w').write("#include <cuda_fp16.h>\nint x;\n")
        if not _deny_hits(probe):
            print("  [FAIL] ppu_portability: an unconditional cuda_fp16.h include is NOT reported -- the "
                  "liveness filter is swallowing everything")
            return 1
        # (3) The N/A boundary itself must fail closed in both directions.  Registering an island TU with PPU CMake,
        #     or including an island header from a PPU source, is a FAIL -- never an inherited exemption.
        cmake_text = open(cmake, errors='replace').read()
        planted_cmake = cmake_text + "\nq4k_pdf_5090_ab.cu\n"
        if not any('became PPU-CMake-reachable' in e
                   for e in _nvidia_island_errors(root, files, planted_cmake)):
            print("  [FAIL] ppu_portability: registering the NVIDIA-only harness as a PPU source escaped")
            return 1
        edge = os.path.join(td, 'ppu_candidate.cuh')
        open(edge, 'w').write('#include "q4k_pdf_reconstruction.cuh"\n')
        if not any('includes NVIDIA-only island' in e
                   for e in _nvidia_island_errors(root, files + [edge], cmake_text)):
            print("  [FAIL] ppu_portability: a PPU include edge into the NVIDIA-only island escaped")
            return 1
    if not bad:
        applicable = sum(_rel(root, f) not in NVIDIA_ONLY_ISLAND and not _is_local_cuda_probe(f)
                         for f in files)
        print(f"  [ok]   ppu_portability: {applicable} PPU candidates are portable; exact RTX5090 Q4_K "
              f"island is N/A to PPU and proved unreachable; 3 planted boundary defects fail")
    return bad

sys.exit(main())
