#pragma once
#include <cstring>

namespace quactlize::device {
// ZW610, ZW810E and ZW810 share the PPU1.0 kernel architecture. ZW610/ZW810E
// deliberately use the SAME ZW810 exact tables, latency values and bucket
// rules. These are selection estimates, NOT measurements on ZW610/ZW810E.
// Keep the evidence and table keys unchanged.
// This is policy compatibility only: launch grids, occupancy and resource
// checks must still use the actual device. ZW810E's CU count is queried at
// runtime, not assumed to equal ZW810's 72 CUs; a missing count is rejected.
inline bool uses_zw810_policy(char const* name, int compute_units) {
    return name && ((!std::strcmp(name, "PPU-ZW810") && compute_units == 72) ||
                    (!std::strcmp(name, "PPU-ZW610") && compute_units == 64) ||
                    (!std::strcmp(name, "PPU-ZW810E") && compute_units > 0));
}
} // namespace quactlize::device
