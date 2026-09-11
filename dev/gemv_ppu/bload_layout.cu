// Host-only composition with real CuTe fragment types. No PPU emulation.
#include <cstdio>
#include <array>
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/half.h"
#include "bload_contract.hpp"
#include "q4_kpack4_offline.hpp"

int main() {
    using namespace cute;
    using Atom = PPU0010_8x16x16_F32F16F16F32_TN;
    using Mma = TiledMMA<MMA_Atom<Atom>, Layout<Shape<_1,_1,_1>>, Tile<_8,_16,_16>>;
    auto tensor = make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                              make_layout(Shape<_16,_16>{}, Stride<_1,_16>{}));
    auto identity = make_identity_tensor(Shape<_16,_16>{});
    int bad = 0, wrong_reg = 0, wrong_nibble = 0;
    std::array<int, 256> words{};
    std::array<int, 1024> codes{};
    for (int lane = 0; lane < 32; ++lane) {
        auto thr = Mma{}.get_thread_slice(lane);
        auto frag = thr.partition_fragment_B(tensor);
        auto part = thr.partition_B(identity);
        auto pi = right_inverse(frag.layout());
        for (int reg = 0; reg < 4; ++reg) {
            for (int half = 0; half < 2; ++half) {
                auto coord = part(pi(reg * 2 + half));
                unsigned nn = q4_bload::word_n(lane, reg);
                unsigned kg = q4_bload::word_kg(lane, reg, half);
                bad += nn != unsigned(get<0>(coord)) || kg != unsigned(get<1>(coord));
                wrong_reg += q4_bload::word_n(lane, reg ^ 2) != unsigned(get<0>(coord));
                ++words[kg * 16 + nn];
                for (int s = 0; s < 4; ++s) {
                    auto kk = q4_bload::code_k(kg, s);
                    bad += kk != unsigned(q4_kpack4::logical_k(kg, s));
                    ++codes[kk * 16 + nn];
                    wrong_nibble += q4_bload::code_k(kg, s ^ 1) != kk;
                }
            }
        }
    }
    for (auto visits : words) bad += visits != 1;
    for (auto visits : codes) bad += visits != 1;
    bool ok = !bad && wrong_reg == 256 && wrong_nibble == 1024;
    std::printf("Q4_BLOAD_LAYOUT %s words=256 codes=1024 bad=%d wrong_reg=%d wrong_nibble=%d\n",
                ok ? "PASS" : "FAIL", bad, wrong_reg, wrong_nibble);
    return ok ? 0 : 1;
}
