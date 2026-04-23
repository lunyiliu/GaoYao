#!/usr/bin/env python3
"""Patch vllm cuda.py to use TRITON_ATTN on Blackwell (sm_120) GPUs.

Idempotent: exits 0 if already patched. Exits 2 if the expected source
block is not found (vllm version drift) so the caller can decide.
"""
import glob
import os
import sys

home = os.path.expanduser('~')
candidates = (
    glob.glob('/usr/local/lib/python3.*/dist-packages/vllm/platforms/cuda.py')
    + glob.glob('/usr/lib/python3.*/dist-packages/vllm/platforms/cuda.py')
    + glob.glob('/opt/conda/lib/python3.*/site-packages/vllm/platforms/cuda.py')
    + glob.glob(f'{home}/**/site-packages/vllm/platforms/cuda.py', recursive=True)
)
path = next((p for p in candidates if p), None)
if not path:
    print('ERROR: vllm/platforms/cuda.py not found in standard locations', file=sys.stderr)
    sys.exit(1)

with open(path) as f:
    src = f.read()

if 'device_capability.major == 12' in src:
    print('ALREADY PATCHED:', path)
    sys.exit(0)

old = '''        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]'''

new = '''        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.FLASHINFER,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]'''

if old not in src:
    print('NOT FOUND: expected block missing in', path, file=sys.stderr)
    print('vllm version may differ — open the file and adapt manually.', file=sys.stderr)
    sys.exit(2)

with open(path, 'w') as f:
    f.write(src.replace(old, new, 1))
print('PATCHED:', path)
