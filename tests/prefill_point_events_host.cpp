// Fake runtime: verifies event/call ordering, not GPU performance.
#include <cassert>
#include <utility>
#include <vector>
struct FakeEvent { float clock = 0; };
using fakeStream_t = void *;
using fakeEvent_t = FakeEvent *;
static float clock_value = 0;
static int creates = 0, destroys = 0, records = 0, syncs = 0, calls_at_sync = 0;
static std::vector<std::pair<int, int>> calls;
int fakeEventCreate(fakeEvent_t * p) { *p=new FakeEvent; ++creates; return 0; }
int fakeEventDestroy(fakeEvent_t p) { delete p; ++destroys; return 0; }
int fakeEventRecord(fakeEvent_t p, fakeStream_t) { p->clock=clock_value; ++records; return 0; }
int fakeEventSynchronize(fakeEvent_t) { ++syncs; calls_at_sync=int(calls.size()); return 0; }
int fakeStreamSynchronize(fakeStream_t) { ++syncs; return 0; }
int fakeEventElapsedTime(float * t, fakeEvent_t a, fakeEvent_t b) { *t=b->clock-a->clock; return 0; }
#define POINT_RT(name) fake##name
#include "../dev/prefill_point/events.hpp"
#undef POINT_RT
int main() {
    for (int mode=0; mode<3; ++mode) {
        clock_value=0; creates=destroys=records=syncs=0; calls.clear();
        const int width=mode==1?4:1;
        std::vector<double> result(5*width);
        prefill_point::sample(nullptr,4,5,mode,2,3,[&](int stage,int copy) {
            calls.emplace_back(stage,copy); clock_value+=float(stage+1); return 0;
        },result.data());
        assert(creates==10*width && destroys==creates && records==creates && syncs==1);
        // Two complete ring passes and sixteen warmups precede the first sample.
        const size_t prefix=40+16*(mode==0?1:4);
        assert(calls.size()==prefix+5*(mode==0?1:4) && calls_at_sync==int(calls.size()));
        for (int i=0; i<5; ++i) {
            if (mode==0) assert((calls[prefix+i]==std::make_pair(2,i)));
            else for (int s=0; s<4; ++s) assert((calls[prefix+4*i+s]==std::make_pair(s,i)));
            for (int j=0; j<width; ++j) {
                assert(result[i*width+j]==1000*(mode==0?3:mode==1?j+1:10));
            }
        }
    }
    // Errors must drain outstanding work, release all events and propagate.
    creates=destroys=syncs=0;
    double result[3]; bool failed=false;
    try { prefill_point::sample(nullptr,1,1,0,0,3,[](int,int){return 9;},result); }
    catch (const std::runtime_error &) { failed=true; }
    assert(failed && creates==6 && destroys==6 && syncs==1);
    failed=false;
    try { prefill_point::sample(nullptr,1,0,0,0,3,[](int,int){return 0;},result); }
    catch (const std::runtime_error &) { failed=true; }
    assert(failed);
}
