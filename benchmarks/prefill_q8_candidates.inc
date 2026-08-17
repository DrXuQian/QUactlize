// One authority for the finite Q8_0 ScaleFirst prefill reader family.
//
// The including translation unit must define
//   PREFILL_Q8_CANDIDATE(TM, TN, TK, WM, WN, STAGES)
// before including this file.  test_scalefirst_bench.cu instantiates the
// shipping collective; L208 instantiates the host layout oracle over the
// exact same tuples.  Keeping the rows here prevents either side from going
// green after silently adding, dropping, or changing a candidate on only one
// side.
#ifndef PREFILL_Q8_CANDIDATE
#error "define PREFILL_Q8_CANDIDATE before including prefill_q8_candidates.inc"
#endif

PREFILL_Q8_CANDIDATE(32,  64,  32, 32, 32, 2)
PREFILL_Q8_CANDIDATE(32,  64,  32, 32, 32, 3)
PREFILL_Q8_CANDIDATE(64,  64,  32, 32, 32, 2)
PREFILL_Q8_CANDIDATE(64,  64,  32, 32, 32, 3)
PREFILL_Q8_CANDIDATE(64,  64,  32, 64, 32, 2)
PREFILL_Q8_CANDIDATE(64,  64,  32, 64, 32, 3)
PREFILL_Q8_CANDIDATE(64, 128,  32, 32, 64, 2)
PREFILL_Q8_CANDIDATE(64, 128,  32, 32, 64, 3)
PREFILL_Q8_CANDIDATE(64, 128,  32, 64, 64, 2)
PREFILL_Q8_CANDIDATE(64, 128,  32, 64, 64, 3)
PREFILL_Q8_CANDIDATE(32,  64,  64, 32, 32, 2)
PREFILL_Q8_CANDIDATE(32,  64,  64, 32, 32, 3)
PREFILL_Q8_CANDIDATE(64,  64,  64, 64, 32, 2)
PREFILL_Q8_CANDIDATE(64, 128,  64, 32, 64, 2)
PREFILL_Q8_CANDIDATE(64, 128,  64, 64, 64, 3)
PREFILL_Q8_CANDIDATE(32,  64, 128, 32, 32, 2)
PREFILL_Q8_CANDIDATE(64,  64, 128, 64, 32, 2)
PREFILL_Q8_CANDIDATE(64, 128, 128, 64, 64, 3)

#undef PREFILL_Q8_CANDIDATE
