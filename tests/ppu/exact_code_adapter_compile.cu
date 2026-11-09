// Same fixture can compile against the frozen prototype and the new adapters.
// It is not a correctness/timing runner and does not launch on a local device.
#include "mma_staged.cuh"
namespace mr=quactlize::execution::mma_reader;

extern "C" int exact_code_adapter_probe(qkg_call_v1 const* c,int which,int split) {
    switch(which) {
#if defined(EXACT_CODE_STAGED)
        case 0: return mr::launch_staged<12,0,0,4,4,1,1,2>(*c,split);
        case 1: return mr::launch_staged<12,0,0,4,4,1,2,2>(*c,split);
        case 2: return mr::launch_staged<13,0,0,4,4,1,1,2>(*c,split);
        case 3: return mr::launch_staged<14,0,0,4,4,1,1,2>(*c,split);
        case 4: return mr::launch_staged<8,0,0,4,4,1,1,2>(*c,split);
        case 5: return mr::launch_staged<12,2,1,4,4,1,1,2>(*c,split);
        case 6: return mr::launch_staged<13,2,1,4,4,1,1,2>(*c,split);
        case 7: return mr::launch_staged<12,0,0,2,8,2,1,3>(*c,split);
        case 8: return mr::launch_staged<14,0,0,8,2,2,2,4>(*c,split);
#else
        case 0: return mr::launch_typed<12,0,0,4,4,1,1,2>(*c,split);
        case 1: return mr::launch_typed<12,0,0,4,4,1,2,2>(*c,split);
        case 2: return mr::launch_typed<12,0,0,4,4,1,4,2>(*c,split);
        case 3: return mr::launch_typed<12,0,0,4,4,1,8,2>(*c,split);
        case 4: return mr::launch_typed<13,0,0,4,4,1,1,2>(*c,split);
        case 5: return mr::launch_typed<14,0,0,4,4,1,1,2>(*c,split);
        case 6: return mr::launch_typed<8,0,0,4,4,1,1,2>(*c,split);
        case 7: return mr::launch_typed<12,2,1,4,4,1,1,2>(*c,split);
        case 8: return mr::launch_typed<13,2,1,4,4,1,1,2>(*c,split);
        case 9: return mr::launch_typed<12,0,0,2,2,4,4,4>(*c,split);
        case 10: return mr::launch_typed<14,0,0,8,2,2,1,2>(*c,split);
#endif
        default: return QKG_INVALID;
    }
}
