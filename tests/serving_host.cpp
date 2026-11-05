#include "quactlize/compat/serving_contract.hpp"
using namespace quactlize::compat::serving;
template <class T> int check(T const *p) {
    try {
        validate(*p);
        return 0;
    } catch (std::exception const &) {
        return 20;
    }
}
extern "C" int serving_router_check(qpm_router_v2 const *p) { return check(p); }
extern "C" int serving_swiglu_check(qpm_swiglu_v2 const *p) { return check(p); }
extern "C" int serving_rows_check(qpm_rows_v1 const *p) { return check(p); }
extern "C" int serving_size(int which) {
    return which == 0 ? sizeof(qpm_router_v2) : which == 1 ? sizeof(qpm_swiglu_v2) : sizeof(qpm_rows_v1);
}
