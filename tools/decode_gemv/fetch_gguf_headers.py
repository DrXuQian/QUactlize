#!/usr/bin/env python3
"""Fetch only the GGUF header prefix of each model shard (HTTP Range), parse it, record identity.

No weights are downloaded. Output: one JSON per model with metadata, tensor list, file sizes,
LFS sha256 (from the hub's linked etag) and the sha256 of the parsed header byte range.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SRC = Path(os.environ.get('QGEMV_SRC', '/workspace/quactlize-decode-gemv-20261108/source'))
sys.path.insert(0, str(SRC))
from tools.gguf_header import read_gguf_header  # noqa: E402

ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')
MODELS = {
    'qwen3-32b': ('Qwen/Qwen3-32B-GGUF', ['Qwen3-32B-Q4_K_M.gguf'], 1),
    'qwen3-32b-unsloth': ('unsloth/Qwen3-32B-GGUF', ['Qwen3-32B-Q4_K_M.gguf'], 1),
    'qwen38-27b': ('unsloth/Qwen3.8-27B-GGUF', ['Qwen3.8-27B-UD-Q4_K_M.gguf'], 1),
    'qwen35-35b-a3b': ('unsloth/Qwen3.5-35B-A3B-GGUF', ['Qwen3.5-35B-A3B-Q4_K_M.gguf'], 1),
    'qwen35-122b-a10b': ('unsloth/Qwen3.5-122B-A10B-GGUF',
                         [f'Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-{i:05d}-of-00003.gguf' for i in (1, 2, 3)], 2),
    'deepseek-v4-flash': ('unsloth/DeepSeek-V4-Flash-0731-GGUF',
                          [f'UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-{i:05d}-of-00005.gguf' for i in range(1, 6)], 4),
}
ARCH_KEYS = ('block_count', 'embedding_length', 'feed_forward_length', 'expert_feed_forward_length',
             'expert_shared_feed_forward_length', 'expert_count', 'expert_used_count', 'expert_shared_count',
             'nextn_predict_layers', 'full_attention_interval', 'attention.head_count', 'attention.head_count_kv',
             'attention.key_length', 'attention.value_length', 'attention.key_length_swa',
             'attention.value_length_swa', 'attention.recurrent_layers', 'attention.output_group_count',
             'attention.output_lora_rank', 'attention.q_lora_rank', 'attention.kv_lora_rank',
             'hyper_connection.count', 'ssm.conv_kernel', 'ssm.inner_size', 'ssm.state_size',
             'ssm.time_step_rank', 'ssm.group_count', 'leading_dense_block_count', 'vocab_size',
             'attention.sliding_window')
ARCHES = ('qwen3', 'qwen35', 'qwen35moe', 'qwen3moe', 'deepseek4', 'qwen3next')


def get(url, rng=None, method='GET'):
    headers = {'User-Agent': 'quactlize-gemv-inventory'}
    if rng:
        headers['Range'] = 'bytes=%d-%d' % rng
    req = urllib.request.Request(url, headers=headers, method=method)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, dict(r.headers), (r.read() if method == 'GET' else b'')
        except Exception as exc:  # retry transient mirror errors
            err = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f'{url}: {err}')


def repo_sha(repo):
    _, _, body = get(f'{ENDPOINT}/api/models/{repo}')
    return json.loads(body)['sha']


def lfs_info(repo, rev, path):
    """Tree listing gives size and LFS oid without following the redirect."""
    folder = os.path.dirname(path)
    _, _, body = get(f'{ENDPOINT}/api/models/{repo}/tree/{rev}/{folder}' if folder else
                     f'{ENDPOINT}/api/models/{repo}/tree/{rev}')
    for entry in json.loads(body):
        if entry.get('path') == path:
            lfs = entry.get('lfs') or {}
            return entry.get('size'), lfs.get('oid')
    raise RuntimeError('file not in tree: ' + path)


def fetch_header(url, size, cache):
    want = 4 << 20
    data = b''
    if cache.exists():
        data = cache.read_bytes()
    while True:
        if len(data) < min(want, size):
            status, headers, body = get(url, (len(data), min(want, size) - 1))
            if status != 206:
                raise RuntimeError(f'{url}: range request returned {status}')
            data += body
            cache.write_bytes(data)
        stream = io.BytesIO(data)
        try:
            capture = {f'{a}.{k}' for a in ARCHES for k in ARCH_KEYS}
            header = read_gguf_header(stream, url, capture_keys=capture)
            return header, data[:stream.tell()]
        except Exception as exc:
            if 'truncated' not in str(exc) or want >= size:
                raise
            want *= 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=list(MODELS))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/dev/shm/qgemv/headers'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        repo, files, tp = MODELS[name]
        rev = repo_sha(repo)
        shards = []
        for index, path in enumerate(files):
            size, oid = lfs_info(repo, rev, path)
            url = f'{ENDPOINT}/{repo}/resolve/{rev}/{path}'
            cache = args.cache / (name + '-' + os.path.basename(path) + '.prefix')
            header, raw = fetch_header(url, size, cache)
            meta = header['metadata']
            if meta.get('split.count', 1) != len(files) or meta.get('split.no', 0) != index:
                raise RuntimeError('split metadata disagrees with filename set: ' + path)
            shards.append(dict(path=path, size=size, lfs_sha256=oid, header_bytes=len(raw),
                               header_sha256=hashlib.sha256(raw).hexdigest(), metadata=meta,
                               tensors=header['tensors']))
            print(name, path, 'tensors', len(header['tensors']), 'header', len(raw), flush=True)
        result = dict(schema='qgemv.gguf-header.v1', model=name, repo=repo, revision=rev, tp=tp,
                      endpoint=ENDPOINT, weights_read=False, shards=shards)
        (args.out / f'{name}.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
