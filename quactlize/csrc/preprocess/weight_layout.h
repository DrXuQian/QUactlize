// THE NAME OF A STORED WEIGHT'S ARRANGEMENT. C++ side of quactlize/layouts.py; the two are checked against each
// other by tests/test_layouts.py, because a vocabulary that exists twice will otherwise drift.
//
// A layout's name is the ordered join of its step tokens -- mmarow_tr_cl4_aiu256_cvtword_bias -- so a layout that
// gains, loses or reorders a step gets a different name with no version counter to remember. See layouts.py for why
// each of those properties corresponds to a bug this tree actually had.
//
// This header replaces the pair of booleans (is_int8_mma, use_aiu_interleaved) at the entry point. Those booleans
// are still what the chain switches on internally; what changes is that a CALLER now names an arrangement that
// exists, instead of setting two flags whose legal combinations were written down nowhere and whose illegal ones
// silently produced a different arrangement than the one asked for.
#pragma once

#include <string>
#include <vector>

namespace quactlize {

// What the chain needs in order to run, recovered from a name. Everything here was previously a caller's guess.
struct LayoutPlan {
    bool        is_int8_mma;         // W4A8: the 32-row row permutation, and no code bias
    bool        use_aiu_interleave;  // the AIU's 256-row column interleave
    int         bits;                // element width the name is for; a mismatch with the tensor is a caller error
    int         requires_multiple;   // k and n must both be a multiple of this (0 = no constraint)
    std::string name;
};

namespace detail {

// The registered layouts. Kept as a literal table rather than composed from step objects: on this side the table is
// READ, never built, and a table that can be diffed line-by-line against layouts.py is worth more than symmetry.
// Order matches layouts.py so the two can be compared by eye as well as by test.
struct Registered {
    char const* name;
    char const* alias;
    bool        is_int8_mma;
    bool        use_aiu;
    int         bits;
    int         requires_multiple;
};

inline std::vector<Registered> const& registry()
{
    static std::vector<Registered> const kTable = {
        {"logical",                           "logical",          false, false,  0,   0},
        {"mmarow_tr_cl4_cvtword_bias",        "mixed_gemm",       false, false,  4,   0},
        {"mmarow_tr_cl4_aiu256_cvtword_bias", "mixed_gemm_aiu",   false, true,   4, 256},
        {"mmarow_tr_cl4",                     "w4a8",             true,  false,  4,   0},
        {"mmarow16_tr_cl2_cvtword_bias",      "mixed_gemm_int8",  false, false,  8,   0},
        {"mmarow_tr_cl8_cvtword_bias",        "mixed_gemm_int2",  false, false,  2,   0},
    };
    return kTable;
}

inline std::string registered_names()
{
    std::string s;
    for (auto const& r : registry()) {
        if (!s.empty()) s += ", ";
        s += r.name;
        s += " (";
        s += r.alias;
        s += ")";
    }
    return s;
}

}  // namespace detail

// Resolve a canonical name or an alias. Returns false and fills `error` for anything else -- in particular for a
// name that parses as a token join but is not registered, which is what a weight reordered by a different version
// of this code carries. That case is the reason the name is built from steps: before, such a weight was
// byte-count-, dtype- and shape-identical to a current one.
inline bool resolve_layout(std::string const& name_or_alias, LayoutPlan* out, std::string* error)
{
    for (auto const& r : detail::registry()) {
        if (name_or_alias == r.name || name_or_alias == r.alias) {
            *out = LayoutPlan{r.is_int8_mma, r.use_aiu, r.bits, r.requires_multiple, r.name};
            return true;
        }
    }
    if (error) {
        *error = "'" + name_or_alias + "' is not a registered weight layout. Registered: " + detail::registered_names()
               + ". A name that looks like a layout but is not registered usually means the weight was reordered by "
                 "a different version of the preprocessing: the steps changed, so the name changed.";
    }
    return false;
}

}  // namespace quactlize
