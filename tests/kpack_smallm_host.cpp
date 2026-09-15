#include "quactlize/dispatch/smallm.hpp"
#include <iostream>

int main() {
    int q,mode,n,k,m,channels;
    while (std::cin>>q>>mode>>n>>k>>m>>channels) {
        qkg_call_v1 c{};
        c.version=1;c.size=sizeof(c);c.qtype=q;c.mode=mode;c.n=n;c.k=k;c.input_type=QKG_F32;
        c.rows=mode==QKG_INDEXED ? m*8 : m;c.experts=mode==QKG_INDEXED ? 256 : 1;
        c.channels=channels;c.topk=mode==QKG_INDEXED ? 8 : 1;
        auto selected=quactlize::dispatch::smallm::select(c);
        if (!selected.row) { std::cout<<"MISS\n";continue; }
        auto const& r=*selected.row;
        auto const& f=quactlize::smallm_data::kChoices[r.choice];
        std::cout<<selected.policy<<' '<<r.n<<' '<<r.k<<' '<<r.tokens<<' ';
        if (f.simt) std::cout<<"simt "<<f.reader.variant<<' '<<f.reader.columns<<' '<<f.reader.warps<<' '<<f.reader.values<<' '<<f.reader.split;
        else std::cout<<"tc "<<f.tc.symbol<<' '<<f.tc.split;
        std::cout<<'\n';
    }
    return std::cin.eof() ? 0 : 1;
}
