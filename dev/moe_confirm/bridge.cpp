// Test-only SDK graph bridge, independent of llama's checkout/build identity.
#include <exception>
#include <string>
namespace nc {
inline thread_local std::string error;
template<class F> int guard(F fn) {
    try { fn();error.clear();return 0; }
    catch(std::exception const& e) { error=e.what();return 1; }
}
}
#include "dev/native_components/kernel_bridge.hpp"
extern "C" char const* nc_error() { return nc::error.c_str(); }
