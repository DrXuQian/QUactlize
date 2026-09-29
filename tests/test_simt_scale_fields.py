"""Compile the actual selected scale reader against exhaustive packed fields."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(path, marker):
    text = path.read_text()
    if text.count(marker) != 1:
        raise ValueError("scale reader seam changed")
    start = text.index(marker)
    begin = text.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class SimtScaleFields(unittest.TestCase):
    def test_isa_inventory_rejects_missing_or_duplicate_instructions(self):
        from dev.gemv_simt.tp2_scale_isa import instructions
        text = ' 0: 00 00 00 00 00 00 00 00\ts.nop\n 8: 00 00 00 00 00 00 00 00\ts.nop\n'
        self.assertEqual(len(instructions(text)[0]),2)
        for broken in ('',text+text,text.replace(' 8:', ' 10:')):
            with self.assertRaises(ValueError):
                instructions(broken)

    def test_mask_model_explains_both_k_groups_and_column_residues(self):
        from dev.gemv_simt.tp2_scale_isa import mask_model
        rows = mask_model()
        self.assertEqual([r['groups'] for r in rows],[[6],[14]])
        for r in rows:
            self.assertEqual(r['missing_lanes'],[24,25,26,27])
            self.assertEqual(r['mask'],0x0f000000)
            self.assertEqual(r['n_residues'],[0,4,8,12])

    def test_actual_reader_all_fields_formats_and_recipe_modes(self):
        # Pure integer/affine code only. This is not a CUDA/CuTe host stub and
        # makes no claim about device lowering; the PPU field A/B covers that.
        helper = function(ROOT/'quactlize/execution/q4_s1_helpers.cuh',
                          '__device__ __forceinline__ float2 q4_affine_header32(')
        selected = function(ROOT/'quactlize/execution/simt_kernel.cuh',
                            'template<int Q,int Changes>\n__device__ __forceinline__ float2 affine_selected(')
        source = r'''
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#define __device__
#define __forceinline__ inline
struct uint4 { uint32_t x,y,z,w; };
struct float2 { float x,y; };
uint4 make_uint4(uint32_t x,uint32_t y,uint32_t z,uint32_t w) { return {x,y,z,w}; }
float2 make_float2(float x,float y) { return {x,y}; }
// Superscales are fixed at the exactly representable values 1 and 2.
uint16_t __ushort_as_half(uint16_t x) { return x; }
float __half2float(uint16_t h) {
    if(h!=0x3c00 && h!=0x4000) std::abort();
    return h==0x3c00 ? 1.f : 2.f;
}
struct Unit { static constexpr int kGroups=8; };
template<int Q> struct Format { static constexpr int words=4; using Unit=::Unit; };
template<int Count> struct Meta { uint32_t word[Count]; };
template<int Q> float2 affine(Meta<4> const&,int) { return {-float(Q),float(Q)}; }
namespace q4_s1 {
''' + helper + '\n}\n' + selected + r'''
void require(bool b) { if(!b) std::abort(); }
void bits(Meta<4>& m,unsigned bit,unsigned value) {
    for(unsigned b=0;b<6;++b) {
        unsigned pos=bit+b;
        m.word[pos/32]=(m.word[pos/32]&~(1u<<(pos%32)))|(((value>>b)&1u)<<(pos%32));
    }
}
template<int Q,int Changes>
void check(Meta<4> const& m,int g,float sc,float mn) {
    float2 v=affine_selected<Q,Changes>(m,g);
    require(v.x==sc && v.y==-2.f*mn);
}
template<int Q>
void modes(Meta<4> const& m,int g,float sc,float mn) {
    check<Q,0>(m,g,sc,mn); check<Q,1>(m,g,sc,mn);
    check<Q,2>(m,g,sc,mn); check<Q,3>(m,g,sc,mn);
}
int main() {
    unsigned checked=0;
    for(unsigned g=0;g<8;++g) for(unsigned sc=0;sc<64;++sc) for(unsigned mn=0;mn<64;++mn) {
        // Independent bit-by-bit pack of two 4-group scale/min records.
        Meta<4> m{{0x40003c00u,0xaaaaaaaa,0x55555555,0xcccccccc}};
        unsigned base=32+(g/4)*48;
        bits(m,base+6*(g%4),sc); bits(m,base+24+6*(g%4),mn);
        modes<12>(m,g,sc,mn); modes<13>(m,g+8,sc,mn);
        ++checked;
    }
    Meta<4> m{{0x40003c00u,0,0,0}};
    bits(m,92,19); bits(m,116,7);
    modes<12>(m,6,19,7); modes<13>(m,14,19,7);
    m.word[2]&=0x0fffffffu;
    modes<12>(m,6,16,7); modes<13>(m,14,16,7);
    require(affine_selected<12,0>(m,6).x!=19.f);
    require(affine_selected<8,0>(m,0).x==-8.f);
    require(affine_selected<10,0>(m,0).x==-10.f);
    require(affine_selected<11,0>(m,0).x==-11.f);
    require(affine_selected<14,0>(m,0).x==-14.f);
    std::printf("SIMT_SCALE_FIELDS PASS fields=%u formats=Q4,Q5 modes=4 field_loss=RED other_formats=UNCHANGED\n",checked);
}
'''
        with tempfile.TemporaryDirectory(prefix='simt-scale-fields-') as d:
            path = Path(d)
            (path/'test.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-O2',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
            result = subprocess.run([str(path/'test')],capture_output=True,text=True,check=True)
        self.assertIn('PASS fields=32768 formats=Q4,Q5 modes=4 field_loss=RED', result.stdout)


if __name__ == '__main__':
    unittest.main()
