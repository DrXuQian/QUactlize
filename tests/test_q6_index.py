from pathlib import Path
import subprocess

import pytest

from dev.gemv_simt.q6_index import ROOT, index_body, codes_body, source
from dev.gemv_simt.q6_sparse import sparse_body, source as parent_source


def test_actual_unsigned_helpers_match_canonical_at_boundaries(tmp_path):
    exe=tmp_path/'map'
    subprocess.run(['g++','-std=c++17','-O2',f'-I{ROOT}',str(ROOT/'tests/q6_unsigned_host.cpp'),
                    '-o',str(exe)],check=True)
    result=subprocess.check_output([str(exe)],text=True)
    assert 'Q6_UNSIGNED_HOST PASS' in result
    assert 'negative=DETECTED wide_address=PASS' in result


def test_only_coordinate_and_map_seams_change():
    text=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    body=index_body(text).replace('q6_unsigned_fields_body(', 'q6_sparse_fields_body(').replace(
        'using F=Q6UnsignedFormat<Q>;', 'using F=Format<Q>;').replace(
        'using Index=unsigned;', 'using Index=std::conditional_t<(Changes&2)!=0,unsigned,int>;').replace(
        'unsigned k=g*F::group+', 'int k=g*F::group+').replace(
        'float2 code=q6_unsigned_codes<Q>(', 'float2 code=codes<Q>(')
    assert body==sparse_body(text)
    with pytest.raises(ValueError,match='coordinate seam'):
        index_body(text.replace('int k=g*F::group+', 'int renamed=g*F::group+',1))


def test_dequant_math_is_byte_for_byte_unchanged():
    text=(ROOT/'quactlize/execution/simt_format.cuh').read_text()
    start=text.index('template<int Q>\n__device__ __forceinline__ float2 codes(')
    end=text.index('\n} // namespace',start)
    body=codes_body(text).replace('float2 q6_unsigned_codes(', 'float2 codes(').replace(
        'unsigned col,unsigned k)', 'int col,int k)').replace(
        'static_assert(Q==14);\n    using F=Q6UnsignedFormat<Q>;', 'using F=Format<Q>;').replace(
        'unsigned slot;', 'int slot;')
    assert body==text[start:end]


def test_control_prefix_and_validators_unchanged():
    for compute in ('f16','bf16'):
        old=parent_source(compute);new=source(compute)
        assert new.startswith(old.split('extern "C"')[0])
        export=new[new.index('extern "C"'):]
        export=export.replace('if(arm==0) simt::q6_sparse_fields', 'if(arm==0) simt::q6_register_fields').replace(
            'else simt::q6_unsigned_fields', 'else simt::q6_sparse_fields')
        assert export==old[old.index('extern "C"'):]
    with pytest.raises(ValueError):source('int8')
