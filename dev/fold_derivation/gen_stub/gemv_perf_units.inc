// GENERATED-INTERFACE STUB for the local syntax tier.  The real file is emitted by CMake with one declaration and
// call per compiled unit; this zero-unit form still has to spell the same main-TU ABI so interface drift is caught.
#define GEMV_UNIT_COUNT 0
#define GEMV_GROUP_COUNT 1
static const char* const gemv_group_names[] = {"syntax-stub"};
static const GemvCompiledGroup gemv_compiled_groups[] = {
    {"i4-native", ppu_gemv::tactic_space::Format::Int4,
     ppu_gemv::tactic_space::Layout::Native, 0},
};
static inline void gemv_run_all(const Shape& sh, Best* best, SweepRuntime& sweep) {
  (void)sh;
  (void)best;
  (void)sweep;
}
