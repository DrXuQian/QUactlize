"""Fixture publication order; these host tests do not admit a device kernel."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_actual_stream_upload_body_orders_pageable_map_and_lifetime(tmp_path):
    text = (ROOT/'dev/moe_prepare/bench.cu').read_text()
    begin = text.index('  void put_on_stream(')
    end = text.index('  std::vector<T> get()', begin)
    method = text[begin:end]
    # Model the legal behavior where pageable H2D returns before DMA finishes.
    # The unchanged-stream negative must be observable, never a lucky copy.
    cpp = tmp_path/'stream.cpp'
    cpp.write_text(r'''
#include <cassert>
#include <cstring>
#include <vector>
using cudaStream_t=int;
constexpr int cudaMemcpyHostToDevice=1;
struct Pending { void* dst; const void* src; size_t size; int stream; };
std::vector<Pending> pending;
int cudaMemcpyAsync(void* dst,const void* src,size_t size,int,int stream) {
  pending.push_back({dst,src,size,stream});return 0;
}
int cudaStreamSynchronize(int stream) {
  for(auto it=pending.begin();it!=pending.end();) {
    if(it->stream==stream) {std::memcpy(it->dst,it->src,it->size);it=pending.erase(it);}
    else ++it;
  }
  return 0;
}
void ck(int code){assert(!code);}
void require(bool yes,const char*){assert(yes);}
template<class T> struct Buffer {T* ptr;size_t count;
''' + method + r'''
};
int main() {
  int device[2]={0,1};Buffer<int> b{device,2};
  std::vector<int> bad={1,0},original={0,1};
  cudaMemcpyAsync(device,bad.data(),sizeof(device),cudaMemcpyHostToDevice,0);
  cudaStreamSynchronize(7);
  assert(device[0]==0 && device[1]==1); // Legacy negative can remain invisible.
  cudaStreamSynchronize(0);b.put_on_stream(original,7);
  b.put_on_stream(bad,7);
  assert(device[0]==1 && device[1]==0 && pending.empty());
  b.put_on_stream(original,7);
  assert(device[0]==0 && device[1]==1 && pending.empty());
}
''')
    exe=tmp_path/'stream'
    subprocess.run(['g++','-std=c++17','-O2',cpp,'-o',exe],check=True)
    subprocess.run([exe],check=True)
    proof=text[text.index('  void activation_proof('):text.index('  void correctness(')]
    assert 'down.map.put_on_stream(bad,stream)' in proof
    assert 'down.map.get()==bad' in proof
    assert 'expected_negative!=positive' in proof and 'out==expected_negative' in proof
    assert 'down.map.put_on_stream(map,stream)' in proof
    assert 'ids.put_on_stream(bad,stream)' in text
    assert 'ids.put_on_stream(in,stream)' in text
    timing=text[text.index('  void benchmark('):]
    assert 'put_on_stream' not in timing
