#include "quactlize/dispatch/tc_target.hpp"
#include "quactlize/include/ppu_device.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    using namespace quactlize::dispatch;
    bool native15=false;
#if defined(QUACTLIZE_PPU_ARCH) && QUACTLIZE_PPU_ARCH == 150
    native15=true;
#endif
    assert(quactlize::device::uses_zw810_policy("ZW-M890P",39)==native15);
    assert(quactlize::device::uses_zw810_policy("PPU-ZW890",39)==native15);
    assert(quactlize::device::uses_zw810_policy("PPU-ZW810",72)==!native15);
    assert(quactlize::device::uses_zw810_policy("PPU-ZW610",64)==!native15);
    assert(!quactlize::device::uses_zw810_policy("ZW-M890P",0));
    assert(!quactlize::device::uses_zw810_policy("ZW8900",39));
    for(int q:{8,10,11,12,13,14}) for(int route:{0,1,2,3}) {
        if(q==8 && route%2==0) continue;
        for(int tm:{8,16,32,64}) for(int tn:{16,32,64,128}) for(int dn:{0,16,32,64}) {
            if(dn>tn || (!dn && route!=0)) continue;
            Config old{};
            old.symbol="original_rc4";old.qtype=q;old.route=route;
            old.tm=tm;old.tn=tn;old.tk=256;old.wm=tm;old.wn=16;old.stages=2;
            old.ap=q==12 && route==0 && tm==8;old.dn=dn;
            old.parent_persistent=route==2 ? 1 : -1;old.split=2;
            old.grid_mode=route==2 ? 2 : 0;old.grid_b=1;old.mapping_id=123;
            std::string name;
            auto c=target_config(old,name);
            assert(c.qtype==old.qtype && c.route==old.route && c.tk==old.tk && c.wn==old.wn);
            assert(c.stages==old.stages && c.split==old.split && c.mapping_id==old.mapping_id);
            assert(c.parent_persistent==old.parent_persistent && c.grid_mode==old.grid_mode && c.grid_b==old.grid_b);
            if(native15) {
                assert(c.tm>=16 && c.wm>=16 && c.tn>=32 && c.ap==0 && c.dn>=32);
                assert(c.tm%c.wm==0 && c.tn%c.wn==0 && c.tn%c.dn==0);
                assert(name.rfind("p15_",0)==0 && name.substr(name.size()-4)=="_rc4");
                std::string repeat;
                auto again=target_config(c,repeat);
                assert(name==repeat && again.tm==c.tm && again.dn==c.dn);
            } else {
                assert(name.empty() && c.symbol==old.symbol);
                assert(c.tm==old.tm && c.tn==old.tn && c.wm==old.wm && c.ap==old.ap && c.dn==old.dn);
            }
        }
    }
    std::puts("TARGET_CONTRACT_PASS");
}
