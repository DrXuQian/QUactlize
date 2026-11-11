#pragma once
#include <cstring>

namespace quactlize::device {
inline char const* policy_devices() {
#if defined(QUACTLIZE_PPU_ARCH) && QUACTLIZE_PPU_ARCH == 150
    return "ZW-M890P/ZW890/PPU-ZW890 with positive CUs (native PPU15 package)";
#else
    return "PPU-ZW810/72,PPU-ZW610/64,PPU-ZW810E/positive-CUs";
#endif
}
// ZW610, ZW810E and ZW810 share the PPU1.0 kernel architecture. ZW610/ZW810E
// deliberately use the SAME ZW810 exact tables, latency values and bucket
// rules. These are selection estimates, NOT measurements on ZW610/ZW810E.
// Keep the evidence and table keys unchanged.
// This is policy compatibility only: launch grids, occupancy and resource
// checks must still use the actual device. ZW810E's CU count is queried at
// runtime, not assumed to equal ZW810's 72 CUs; a missing count is rejected.
inline bool uses_zw810_policy(char const* name, int compute_units) {
#if defined(QUACTLIZE_PPU_ARCH) && QUACTLIZE_PPU_ARCH == 150
    // PPU1.5 reuses ZW810 selection tables only as estimates. It requires a
    // native PPU15 package/JIT and uses the real CU count (not ZW810's 72).
    return name && compute_units > 0 && (!std::strcmp(name,"ZW-M890P") ||
        !std::strcmp(name,"PPU-ZW890") || !std::strcmp(name,"ZW890"));
#else
    return name && ((!std::strcmp(name, "PPU-ZW810") && compute_units == 72) ||
                    (!std::strcmp(name, "PPU-ZW610") && compute_units == 64) ||
                    (!std::strcmp(name, "PPU-ZW810E") && compute_units > 0));
#endif
}
} // namespace quactlize::device
