#include "quactlize/prefill/dense_api.h"
#include <hggc_runtime_api.h>

using Run=int(*)(void*,qkp_call_v1 const*,int);
// Ordinary stream events, one component OR one complete call. No graph,
// Python between events, or provider JIT inside measured intervals.
extern "C" int dense_opt_samples(Run run,void* handle,qkp_call_v1 const* calls,int copies,
        int stage,int samples,double* output) {
    if(!run || !handle || !calls || copies<1 || samples<1 || !output)return 1;
    auto stream=static_cast<hggcStream_t>(calls[0].weight.stream);
    hggcEvent_t begin=nullptr,end=nullptr;
    int rc=hggcEventCreate(&begin);
    if(!rc)rc=hggcEventCreate(&end);
    for(int i=0;!rc && i<copies;++i)rc=run(handle,&calls[i],-1);
    if(!rc)rc=hggcStreamSynchronize(stream);
    int count=copies*((8+copies-1)/copies);
    for(int sample=0;!rc && sample<samples;++sample) {
        rc=hggcEventRecord(begin,stream);
        for(int i=0;!rc && i<count;++i)rc=run(handle,&calls[i%copies],stage);
        if(!rc)rc=hggcEventRecord(end,stream);
        if(!rc)rc=hggcEventSynchronize(end);
        float ms=0;if(!rc)rc=hggcEventElapsedTime(&ms,begin,end);
        if(!rc)output[sample]=double(ms)*1000/count;
    }
    int sync=hggcStreamSynchronize(stream);if(!rc)rc=sync;
    if(begin)hggcEventDestroy(begin);
    if(end)hggcEventDestroy(end);
    return rc;
}
