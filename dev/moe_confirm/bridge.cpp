// Test-only SDK graph bridge, independent of llama's checkout/build identity.
#include <exception>
#include <string>
#include "quactlize/prefill/layout.hpp"
namespace nc {
inline thread_local std::string error;
template<class F> int guard(F fn) {
    try { fn();error.clear();return 0; }
    catch(std::exception const& e) { error=e.what();return 1; }
}
}
#include "dev/native_components/kernel_bridge.hpp"
extern "C" char const* nc_error() { return nc::error.c_str(); }
extern "C" int qzm_test_full_workspace(qkp_call_v1 const* call,
        quactlize_ppu_placed_arrangement_v2 const* arr,uint64_t* bytes) {
    if(!call || !bytes) return QKG_INVALID;
    *bytes=0;
    try {
        quactlize::prefill::Layout l{};
        int rc=quactlize::prefill::layout(*call,arr,l);
        if(!rc)*bytes=l.bytes;
        return rc;
    } catch(std::exception const&) {return QKG_OVERFLOW;}
}
