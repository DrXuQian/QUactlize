// Independent replay of the pre-refactor host branch predicates.
#include "quactlize/execution/simt_strategy.hpp"
#include <cassert>
#include <iostream>

using namespace quactlize::execution::simt;

static Q8Strategy old_q8(qkg_simt_call_v2 const& d,int v,int c,int w,int p,int s) {
    auto const& a=d.call;
    bool dense=a.input_type==QKG_F32 && a.mode==QKG_DENSE && a.rows==1 &&
        a.experts==1 && a.topk==1 && a.channels==1 && a.n>0 && a.k>0;
    if(d.compute_type==QKG_COMPUTE_F16) {
        if(v==5 && c==8 && w==4 && p==4) {
            if(s==8 && dense && a.n==2048 && a.k==4096)return Q8Strategy::FixedCold;
            if(s==1 && dense && a.n==8192 && a.k==2048)return Q8Strategy::FixedHoisted;
            if(s==1 && dense)return Q8Strategy::Hoisted;
        } else if(v==5 && c==4 && w==2 && p==4) {
            if(s==1 && dense)return Q8Strategy::S1Hoisted;
        } else if(v==5 && p==4 && c==4 && w==8) {
            if(s==1 && dense && a.n==4096 && a.k==2048)return Q8Strategy::S1Narrow;
            if(s==1 && dense)return Q8Strategy::Hoisted;
        }
    }
    return Q8Strategy::Generic;
}

int main() {
    size_t cases=0;
    for(int n:{768,1024,1536,2048,4096,8192})for(int k:{512,2048,3072,4096})
    for(int q:{8,12,13})for(int compute:{0,1})for(int input:{0,1,2})
    for(int mode:{QKG_DENSE,QKG_INDEXED})for(int tokens=1;tokens<=8;++tokens)
    for(int channels:{1,8})for(int v:{3,4,5})for(int c:{4,8})for(int w:{2,4,8})
    for(int p:{4,8})for(int s:{1,2,4,8}) {
        qkg_simt_call_v2 d{};d.compute_type=compute;auto& a=d.call;
        a.qtype=q;a.mode=mode;a.n=n;a.k=k;a.input_type=input;a.channels=channels;
        a.experts=mode==QKG_DENSE?1:256;a.topk=mode==QKG_DENSE?1:8;a.rows=tokens*a.topk;
        assert(q8_strategy(d,v,c,w,p,s)==old_q8(d,v,c,w,p,s));
        bool measured=compute==1 && input==QKG_F32 && mode==QKG_INDEXED && tokens==1 &&
            v==3 && c==4 && s==1 &&
            ((q==12 && w==4 && p==4 && n==1024 && k==2048 && channels==1) ||
             (q==13 && w==2 && p==8 && n==2048 && k==512 && channels==8));
        assert(measured_reuse(d,v,c,w,p,s)==measured);
        ++cases;
    }
    static_assert(kBf16F32Changes<13,3,4,2,8> == 3);
    static_assert(kBf16F32Changes<12,3,4,2,8> == 0);
    std::cout<<"SIMT_STRATEGY_SHADOW PASS cases="<<cases<<'\n';
}
