// Diagnostic copy of the actual host helper lookup, never another policy table.
// The coordinator checks device eligibility before using this as a control.
#include "quactlize/prefill/dense_helpers.hpp"
#include <hggc_runtime_api.h>
#include <cstring>
extern "C" int dense_helper_device_name(char* name,int capacity) {
    hggcDeviceProp prop{};
    int rc=int(hggcGetDeviceProperties(&prop,0));
    if(rc || !name || capacity<256)return rc?rc:1;
    std::memcpy(name,prop.name,256);name[255]=0;return 0;
}
extern "C" void dense_helper_control(int q,int m,int n,int k,int storage,int* dq,int* io) {
    auto c=quactlize::prefill::dense_helpers(q,m,n,k,storage);
    *dq=c.dequant;*io=c.io;
}
