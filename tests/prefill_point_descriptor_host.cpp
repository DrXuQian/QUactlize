// Execute the actual new C++ descriptor dispatch. No SDK/GPU timing claim.
#include "../dev/prefill_point/timer.cpp"
#include <cassert>
extern "C" hggcError_t hggcMemsetAsync(void*,int,size_t,hggcStream_t){return hggcSuccess;}
int full(void* h,void* c,int stage){assert(h==reinterpret_cast<void*>(1) && c==reinterpret_cast<void*>(2) && stage==3);return 17;}
int io(int storage,int input,int vector,const void* a,void* out,int rows,int width,int64_t stride,const int* error,void* stream){
    assert(storage==0 && input==1 && vector==3 && a==reinterpret_cast<void*>(1) && out==reinterpret_cast<void*>(2));
    assert(rows==2048 && width==5120 && stride==5124 && error==nullptr && stream==reinterpret_cast<void*>(4));return 19;
}
int main(){
    PointCall c{};assert(sizeof(c)==136);
    c.kind=4;c.function=reinterpret_cast<void*>(&full);c.p[0]=reinterpret_cast<void*>(1);c.p[1]=reinterpret_cast<void*>(2);c.v[0]=3;
    assert(invoke(c)==17);
    c.kind=5;c.function=reinterpret_cast<void*>(&io);c.p[3]=reinterpret_cast<void*>(4);
    c.v[0]=0;c.v[1]=1;c.v[2]=3;c.v[3]=2048;c.v[4]=5120;c.stride=5124;
    assert(invoke(c)==19);
}
