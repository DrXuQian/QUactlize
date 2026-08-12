#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "ppu_format_config.hpp"
#include "thop/ppu_backend.h"

namespace {

struct Mapping {
  int qtype;
  int packed_format;
};

// Independent wire-format anchors.  The implementation under test obtains the
// mapping from ppu_formats::for_qtype(); using the same registry for `want`
// would let a transposed registry select the wrong library and still pass.
constexpr std::array<Mapping, 5> kWireMappings{{
    {10, 2},  // Q2_K
    {11, 3},  // Q3_K
    {12, 0},  // Q4_K
    {13, 1},  // Q5_K
    {14, 4},  // Q6_K
}};

int marker(torch_ext::ppu_backend::Api const* api) {
  return api->vecdot(nullptr, 0, nullptr, nullptr, 0, 0, 0);
}

int check_formats(int tag_base, bool plant_wrong_map) {
  int failures = 0;
  for (auto const& wire : kWireMappings) {
    auto const& config = ppu_formats::for_qtype(wire.qtype);
    if (config.qtype != wire.qtype || config.packed_format != wire.packed_format) {
      std::cerr << "L140 REGISTRY_MAP_MISMATCH qtype=" << wire.qtype
                << " registry=" << config.packed_format
                << " wire=" << wire.packed_format << '\n';
      failures++;
      continue;
    }

    int selected_format = config.packed_format;
    if (plant_wrong_map && wire.qtype == 11)
      selected_format = 4;  // Q3_K deliberately routed to Q6_K's binary.

    std::string why;
    auto const* api = torch_ext::ppu_backend::load_format(selected_format, &why);
    int const got = api ? marker(api) : -1;
    int const want = tag_base + wire.packed_format;
    std::cout << "L140 qtype=" << wire.qtype
              << " format=" << selected_format
              << " marker=" << got
              << " want=" << want
              << " path='" << why << "'\n";
    if (!api || got != want) failures++;
  }
  if (failures) {
    std::cerr << "L140 EXPECTED_CONTRACT_RED failures=" << failures << '\n';
    return 1;
  }
  std::cout << "L140 FORMAT_MAP_AND_PATHS PASS\n";
  return 0;
}

int check_default(int expected_tag) {
  std::string why;
  auto const* api = torch_ext::ppu_backend::load(&why);
  int const got = api ? marker(api) : -1;
  std::cout << "L140 default marker=" << got << " want=" << expected_tag
            << " path='" << why << "'\n";
  if (!api || got != expected_tag) {
    std::cerr << "L140 DEFAULT_PATH_RED\n";
    return 1;
  }
  std::cout << "L140 DEFAULT_PATH PASS\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--default")
    return check_default(std::stoi(argv[2]));
  if (argc != 2) {
    std::cerr << "usage: l140 TAG_BASE | l140 --default TAG\n";
    return 2;
  }
  bool const plant_wrong_map = std::getenv("L140_PLANT_WRONG_MAP") != nullptr;
  return check_formats(std::stoi(argv[1]), plant_wrong_map);
}
