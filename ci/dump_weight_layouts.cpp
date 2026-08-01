// THE C++ REGISTRY, AS THE COMPILER SEES IT. Printed so python can compare against it instead of parsing the header.
//
// WHY THIS EXISTS. The cross-check used to regex the header for `kTable = {` and read rows until the first `};`.
// That parse was defeated by putting a commented-out table with the expected rows ahead of the real one and then
// changing a real row: every regex comparison read the comment and passed, while the compiled resolver returned the
// changed value -- selecting the IMMA row permutation for 8-bit weights, which the producer then refuses because its
// formula needs a 32-row tile. A behaviour-changing edit, invisible to the gate.
//
// A better regex would not fix that. Comments, #if, macros and continuation lines are all things a C++ compiler
// resolves and a pattern cannot, so the only sound comparison is against what the compiler produced. This program
// links nothing and needs no CUDA: weight_layout.h is plain host C++.
//
//   g++ -std=c++17 -I quactlize/csrc/preprocess -o dump ci/dump_weight_layouts.cpp && ./dump
//
// One row per line: name<TAB>alias<TAB>is_int8_mma<TAB>use_aiu<TAB>bits<TAB>requires_multiple
// Every row is emitted through resolve_layout(), not read out of the table, so a resolver that disagrees with its
// own table is caught as well.
#include <cstdio>
#include <string>

#include "weight_layout.h"

int main()
{
    for (auto const& r : quactlize::detail::registry()) {
        quactlize::LayoutPlan p;
        std::string err;
        if (!quactlize::resolve_layout(r.name, &p, &err)) {
            std::printf("UNRESOLVABLE\t%s\t%s\n", r.name, err.c_str());
            return 1;
        }
        // Printed from the RESOLVED plan, so a lookup that returns a different row than the one it matched shows up
        // as a name mismatch rather than passing silently.
        std::printf("%s\t%s\t%d\t%d\t%d\t%d\n", p.name.c_str(), r.alias, int(p.is_int8_mma),
                    int(p.use_aiu_interleave), p.bits, p.requires_multiple);
    }
    return 0;
}
