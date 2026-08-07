// STUB of the CMake-generated moe_bench_units.inc, for the LOCAL syntax gate only.
//
// WHY IT EXISTS. test_lowbit_moe_bench.cu is the file the whole sweep runs through and it had no local gate,
// because a bare nvcc front end cannot find this include -- CMake writes it into the binary tree. So the one
// bench with the most host-side logic was the one nothing checked, and a selection rewrite (median over
// interleaved repeats, band, tie reporting) was committed without ever being compiled.
//
// ZERO UNITS IS THE POINT. The generated version declares one function per swept shape; none of that is host
// logic under test here. What this stub has to reproduce exactly is the INTERFACE main() consumes --
// MOE_UNIT_COUNT, MOE_TU_COUNT, MOE_FMT_COUNT, moe_fmt_names[], moe_run_all(const Band&, Best*) -- so a change to any of
// those signatures fails here rather than on the box. Keep the shape of the real generator (CMakeLists.txt.in
// around line 806) if that template changes.
#define MOE_UNIT_COUNT 0
#define MOE_TU_COUNT 0
#define MOE_FMT_COUNT 1
static const char* const moe_fmt_names[] = {"stub"};
static inline void moe_run_all(const Band& bd, Best* b) { (void)bd; (void)b; }
