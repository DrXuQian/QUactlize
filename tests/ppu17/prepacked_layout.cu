#include <cstdio>
#include <cstdlib>
#include <vector>
#include "quactlize/backends/ppu17/gptq_prepacked.cuh"
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "packing check failed line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
  using Map = quactlize::ppu17::PrepackedGptq<128,16>::Packing;
  auto inner = Map::ppblock_ilvd_NK_to_offset();
  std::vector<int> owners(128*64, 0);
  unsigned stale = 0;
  for (int n=0; n<128; ++n) for (int k=0; k<64; ++k) {
    // Independent scalar derivation of the FP16 WGMMA RS A fragment:
    // warp selects sixteen rows; lane/4 selects the low three row bits;
    // lane%4 selects an adjacent-K pair. Fragment bits add K1,N8,K8;
    // four K16 atoms provide the remaining K extent.
    int thread = (n/64)*128 + ((n%64)/16)*32 + (n%8)*4 + (k%8)/2;
    int value = (k%2) + ((n%16)/8)*2 + ((k%16)/8)*4 + (k/16)*8;
    int interleaved = (value/8)*8 + (value%2)*4 + (value%8)/2;
    int want = thread*32 + interleaved;
    int got = inner(cute::make_coord(n,k));
    CHECK(got == want && got >= 0 && got < 128*64);
    ++owners[got];
    stale += got != n*64+k;
    CHECK(got/32 == thread); // all32 nibbles/thread occupy one aligned16B vector
  }
  for (int count : owners) CHECK(count == 1);
  CHECK(stale > 8000); // substituting the old K-major layout must fail
  unsigned blocks = 0;
  for (int n : {128,512,4096}) for (int k : {128,512,4096}) {
    auto outer = Map::bNbKL_to_offset(cute::make_shape(n,k,1));
    for (int bn=0; bn<n/128; ++bn) for (int bk=0; bk<k/64; ++bk) {
      CHECK(outer(cute::make_coord(bn,bk,0)) == 8192*(bn+bk*(n/128)));
      ++blocks;
    }
  }
  std::printf("SM90_PREPACKED_HOST PASS coordinates=8192 vectors=256 blocks=%u stale_layout_red=%u\n", blocks, stale);
}
