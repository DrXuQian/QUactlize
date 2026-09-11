// A real PPU launch and device identity, before any performance measurement.
#include <hggc_runtime.h>
#include <cstdint>

__global__ void q4_ppu_marker(uint32_t* out) { if(threadIdx.x==0) *out=0x514b504b; }
extern "C" int q4_ppu_probe(int* l2, int* sm, int* warp, char* name) {
    hggcDeviceProp p{};
    auto rc=hggcGetDeviceProperties(&p,0);
    if(rc!=hggcSuccess) return int(rc);
    *l2=p.l2CacheSize; *sm=p.multiProcessorCount; *warp=p.warpSize;
    for(int i=0;i<256;++i) name[i]=p.name[i];
    uint32_t *device=nullptr, host=0;
    rc=hggcMalloc(&device,sizeof(host));
    if(rc!=hggcSuccess) return int(rc);
    rc=hggcMemset(device,0,sizeof(host));
    if(rc==hggcSuccess) {
        q4_ppu_marker<<<1,32>>>(device);
        rc=hggcGetLastError();
    }
    if(rc==hggcSuccess) rc=hggcDeviceSynchronize();
    if(rc==hggcSuccess) rc=hggcMemcpy(&host,device,sizeof(host),hggcMemcpyDeviceToHost);
    auto released=hggcFree(device);
    if(rc!=hggcSuccess) return int(rc);
    if(released!=hggcSuccess) return int(released);
    return host==0x514b504b ? 0 : -1;
}
