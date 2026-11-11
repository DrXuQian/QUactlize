#pragma once
#include "policy.hpp"
#include <string>

namespace quactlize::dispatch {
// Architecture legality, not a new performance selector. Keep the chosen
// FQ/SF route, K tile, stages, split and quantized bytes; expose the effective
// geometry as its own parent. PPU15 has native M16, not PPU10's M8/packed-A.
inline Config target_config(Config c, std::string& name) {
#if defined(QUACTLIZE_PPU_ARCH) && QUACTLIZE_PPU_ARCH == 150
    std::string original=c.symbol;
    c.tm=std::max(c.tm,16); c.wm=std::max(c.wm,16);
    c.tn=std::max(c.tn,32); c.ap=0;
    c.dn=c.dn ? std::max(c.dn,32) : std::min(c.tn,64);
    name="p15_q"+std::to_string(c.qtype)+"_r"+std::to_string(c.route)+
        "_tm"+std::to_string(c.tm)+"_tn"+std::to_string(c.tn)+"_tk"+std::to_string(c.tk)+
        "_wm"+std::to_string(c.wm)+"_wn"+std::to_string(c.wn)+"_s"+std::to_string(c.stages)+
        "_ap0_dn"+std::to_string(c.dn)+"_pp"+std::to_string(c.parent_persistent+1);
    // Dense reducer remains part of the parent identity and generated body.
    if (original.size()>=4) {
        auto suffix=original.substr(original.size()-4);
        if(suffix=="_rc2" || suffix=="_rc4" || suffix=="_rc8") name+=suffix;
    }
    c.symbol=name.c_str();
#else
    (void)name;
#endif
    return c;
}
} // namespace quactlize::dispatch
