"""Keep the product build/test closure independent from experiment runners."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_no_experiment_imports_in_product_or_ci():
    for directory in ('quactlize', 'tools', 'ci', 'tests'):
        for path in (ROOT / directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                assert not any(n.split('.')[0] in ('dev', 'benchmarks') for n in names), path


def test_no_experiment_payloads_in_product_tree():
    for directory in ('dev', 'benchmarks', 'prebuilt'):
        files = [p for p in (ROOT / directory).rglob('*') if p.is_file() and
                 '__pycache__' not in p.parts and p.suffix != '.pyc']
        assert not files, files


def test_admitted_q8_body_and_generated_catalog_are_current():
    from tools import generate_q8_refresh as gen
    assert gen.header() == (ROOT / 'quactlize/execution/simt_q8_refresh.cuh').read_text()
    assert gen.catalog() == (ROOT / 'quactlize/execution/q8_refresh.inc').read_text()


def test_q8_source_change_is_not_silently_re_admitted(monkeypatch):
    from tools import generate_q8_refresh as gen
    body = gen.body
    monkeypatch.setattr(gen, 'body', lambda i: body(i) + '\n// changed')
    with pytest.raises(ValueError, match='new admission required'):
        gen.header()


def test_typed_q4_codegen_retains_every_selected_reader():
    from quactlize.execution import q4_decode_codegen as gen
    sources = gen.sources()
    for (n, k), recipes in gen.recipes().items():
        text = sources[f'q4decode_{n}_{k}.cu']
        for r, v, w, p, c in recipes:
            assert f'launch<{r},{v},{w},{p},{c},{n},{k}>(c)' in text
            assert f'launch<{r},{v},{w},{p},{c},{n},{k},1>(c)' in text
