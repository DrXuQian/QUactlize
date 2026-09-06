// Included after an existing benchmark .cu with main renamed to
// kpack_benchmark_main. A request is ID<TAB>modules-file<TAB>argv ... .
// No shell evaluation. A failed request terminates this device context;
// the coordinator preserves completed requests and restarts only remaining
// work. The original exact fixture/correctness/timing implementation is used.
#include <chrono>
#include <iostream>
#include <sstream>
#include "kpack_tuner_registry.hpp"

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);
  int count = 0, device = 0;
  hggcDeviceProp prop{};
  if (hggcGetDeviceCount(&count) != hggcSuccess || count != 1 ||
      hggcGetDevice(&device) != hggcSuccess ||
      hggcGetDeviceProperties(&prop, device) != hggcSuccess) {
    std::fprintf(stderr, "KPACK_TUNER_DEVICE_FAIL visible=%d expected=1\n", count);
    return 2;
  }
  int const cu = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  std::printf("KPACK_TUNER_DEVICE visible=%d device=%d name=%s pci=%04x:%02x:%02x cu=%d\n",
      count, device, prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, cu);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, '\t')) fields.push_back(field);
    if (fields.size() < 3 || fields[0].empty() || fields[1].empty()) return 2;
    for (auto const& item : fields)
      if (item.empty() || item.find_first_of("\r\n") != std::string::npos) return 2;
    std::string const id = fields[0];
    if (id.find_first_not_of("0123456789abcdef") != std::string::npos) return 2;
    if (setenv("KPACK_TUNER_MODULES", fields[1].c_str(), 1)) return 2;
    std::vector<char*> argv{const_cast<char*>("kpack-tuner")};
    for (std::size_t i = 2; i < fields.size(); ++i) argv.push_back(fields[i].data());
    argv.push_back(nullptr);
    std::printf("KPACK_TUNER_BEGIN id=%s\n", id.c_str());
    auto const start = std::chrono::steady_clock::now();
    int rc = 2;
    try { rc = kpack_benchmark_main(int(argv.size()) - 1, argv.data()); }
    catch (std::exception const& error) {
      std::fprintf(stderr, "KPACK_TUNER_ERROR id=%s reason=%s\n", id.c_str(), error.what());
    }
    double const seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::printf("KPACK_TUNER_END id=%s rc=%d wall_seconds=%.6f\n", id.c_str(), rc, seconds);
    if (rc) return rc;
  }
  return 0;
}
