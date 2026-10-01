#include "resource_policy.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

using pdl_experiment::choose_resources;
void require(bool ok) { if (!ok) throw std::runtime_error("resource-policy assertion"); }

int main() {
  int current=-1;
  std::vector<int> writes;
  auto set=[&](int n){current=n;writes.push_back(n);};
  auto query=[&]{return current==-1 || current>=37 ? 3 : 2;};
  auto ordinary=choose_resources(-1,current,set,query);
  require(writes.empty() && ordinary.selected==-1 && ordinary.selected_blocks==3);
  auto automatic=choose_resources(-2,current,set,query);
  require(automatic.selected==37 && automatic.default_blocks==3 && automatic.selected_blocks==3);
  require(writes.size()==39 && writes.front()==-1 && writes.back()==37);
  // A fixed diagnostic is permitted to lose occupancy, but records that loss.
  auto fixed=choose_resources(25,current,set,query);
  require(fixed.selected==25 && fixed.default_blocks==3 && fixed.selected_blocks==2);
  writes.clear();current=25;
  ordinary=choose_resources(-1,current,set,query);
  require(writes.empty() && ordinary.selected==25 && ordinary.selected_blocks==2);
  for(int kind=0;kind<5;++kind) {
    current=25;writes.clear();bool rejected=false;
    auto bad=[&] {
      if(kind==0)return 0;
      if(kind==1)return current==-1?3:1;
      if(kind==2 && current==7)throw std::runtime_error("query failure");
      return current==-1?3:2;
    };
    try {choose_resources(kind==3?101:kind==4?-3:-2,current,set,bad);}
    catch(std::exception const &) {rejected=true;}
    require(rejected && current==25);
    require(kind>=3 ? writes.empty() : writes.back()==25);
  }
  current=-1;writes.clear();
  auto all_equal=choose_resources(-2,current,set,[]{return 8;});
  require(all_equal.selected==0 && all_equal.selected_blocks==8);
  require(writes.size()==2 && writes[0]==-1 && writes[1]==0);
  std::puts("PDL_RESOURCE_POLICY PASS default_inert=1 first_preserving=1 fixed_control=1 restore_on_error=1 negatives=5");
}
