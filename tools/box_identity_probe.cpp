// One atomic view of the PPU devices visible to the hggc runtime.
//
// This deliberately prints a tiny, versioned wire format instead of JSON.  The
// Python owner (probe_box_identity.py) validates it and owns the one canonical
// JSON schema consumed by the box runners.  Keeping all ordinals in this one
// process matters: four independent shell probes can each select a different
// "first" device and still produce a plausible-looking identity bundle.

#include <hggc_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <iostream>
#include <string>

namespace {

std::string hex_encode(std::string const& text) {
  static char const digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(text.size() * 2);
  for (unsigned char byte : text) {
    encoded.push_back(digits[byte >> 4]);
    encoded.push_back(digits[byte & 0xf]);
  }
  return encoded;
}

struct PciQuery {
  using Function = int (*)(char*, int, int);
  Function function = nullptr;
  char const* symbol = "";
};

PciQuery pci_query() {
  for (char const* symbol : {"hggcDeviceGetPCIBusId",
                             "cudaDeviceGetPCIBusId"}) {
    if (void* address = dlsym(RTLD_DEFAULT, symbol)) {
      return {reinterpret_cast<PciQuery::Function>(address), symbol};
    }
  }
  return {};
}

std::string pci_identity(PciQuery const& query, int ordinal) {
  if (query.function == nullptr) {
    return {};
  }
  char buffer[64]{};
  if (query.function(buffer, sizeof(buffer), ordinal) != 0 || buffer[0] == 0) {
    return {};
  }
  return buffer;
}

struct DriverVersion {
  std::string value;
  std::string symbol;
};

DriverVersion runtime_driver_version() {
  // Some SDK releases export the CUDA-compatible runtime query without
  // declaring it in every public header.  Resolve only that documented-shape
  // symbol at runtime; absence is evidence of unavailability, not "unknown".
  using DriverVersionFn = int (*)(int*);
  for (char const* symbol : {"hggcDriverGetVersion", "cudaDriverGetVersion"}) {
    void* address = dlsym(RTLD_DEFAULT, symbol);
    if (address == nullptr) {
      continue;
    }
    auto query = reinterpret_cast<DriverVersionFn>(address);
    int version = 0;
    if (query(&version) == 0 && version > 0) {
      return {std::to_string(version), symbol};
    }
  }
  return {};
}

}  // namespace

int main() {
  std::cout << "QZ_HGGC_DEVICE_PROBE_V1\n";

  int count = 0;
  hggcError_t count_status = hggcGetDeviceCount(&count);
  if (count_status != hggcSuccess) {
    std::cerr << "hggcGetDeviceCount failed: "
              << hggcGetErrorName(count_status) << ": "
              << hggcGetErrorString(count_status) << '\n';
    return 3;
  }
  if (count < 0) {
    std::cerr << "hggcGetDeviceCount returned a negative count\n";
    return 4;
  }

  std::cout << "count\t" << count << '\n';
  PciQuery const pci = pci_query();
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    hggcDeviceProp prop{};
    hggcError_t status = hggcGetDeviceProperties(&prop, ordinal);
    if (status != hggcSuccess) {
      std::cout << "device_error\t" << ordinal << '\t'
                << static_cast<int>(status) << '\n';
      continue;
    }
    std::string const bdf = pci_identity(pci, ordinal);
    std::cout << "device\t" << ordinal << '\t'
              << hex_encode(std::string(prop.name)) << '\t' << prop.major
              << '\t' << prop.minor << '\t' << prop.multiProcessorCount
              << '\t' << (bdf.empty() ? "-" : bdf) << '\n';
  }

  std::cout << "pci_method\t"
            << (pci.function == nullptr ? "-" : pci.symbol) << '\n';
  DriverVersion const driver = runtime_driver_version();
  std::cout << "driver\t" << (driver.value.empty() ? "-" : driver.value)
            << '\t' << (driver.symbol.empty() ? "-" : driver.symbol) << '\n';
  return 0;
}
