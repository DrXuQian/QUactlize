#pragma once
#include <cstring>

namespace quactlize::device {
// ZW610 and ZW810 share the PPU1.0 kernel architecture. ZW610 deliberately
// uses the SAME ZW810 exact tables, latency values and bucket rules. Those
// times are ZW810 measurements used as ZW610 selection estimates, NOT new
// ZW610 measurements. Keep the evidence and table keys unchanged.
// This is policy compatibility only: launch grids, occupancy and resource
// checks must still use the actual device (64 CUs on ZW610, 72 on ZW810).
inline bool uses_zw810_policy(char const* name, int compute_units) {
    return name && ((!std::strcmp(name, "PPU-ZW810") && compute_units == 72) ||
                    (!std::strcmp(name, "PPU-ZW610") && compute_units == 64));
}
} // namespace quactlize::device
