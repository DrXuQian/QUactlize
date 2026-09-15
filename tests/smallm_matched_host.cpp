#include "quactlize/dispatch/smallm_matched.hpp"
#include <iostream>
int main() {
  using namespace quactlize::dispatch;
  int q,mode,n,k,e,top,ch,m,compute;
  while(std::cin>>q>>mode>>n>>k>>e>>top>>ch>>m>>compute) {
    qkg_call_v1 c{};c.version=1;c.size=sizeof(c);c.qtype=q;c.mode=mode;c.n=n;c.k=k;
    c.experts=e;c.topk=top;c.channels=ch;c.rows=mode==2?m*top:m;c.input_type=QKG_F32;
    qkg_simt_call_v2 d{2,sizeof(d),c,compute};auto s=matched::select(d);
    if(!s.row) {std::cout<<"MISS\n";continue;}
    auto r=*s.row;auto f=matched::data::kChoices[r.choice];
    std::cout<<s.policy<<" "<<r.n<<" "<<r.k<<" "<<r.tokens<<" "<<f.kind;
    if(f.kind==0) std::cout<<" "<<f.tc.symbol<<" "<<f.tc.split;
    else std::cout<<" "<<f.reader.reader<<" "<<f.reader.variant<<" "<<f.reader.columns<<" "<<f.reader.warps<<" "<<f.reader.values<<" "<<f.reader.split;
    std::cout<<"\n";
  }
}
