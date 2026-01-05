#!/usr/bin/env python3
"""Patch vLLM nvfp4.py to fix experts_cls returning None bug"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"

with open(file_path, 'r') as f:
    content = f.read()

# The bug: when looping through flashinfer backends, it returns None instead of k_cls
old_code = '''                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, None'''

new_code = '''                if supported:
                    logger.info_once(_make_log_backend(backend), scope="local")
                    return backend, k_cls'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print("PATCHED: Fixed experts_cls bug in nvfp4.py")
elif new_code in content:
    print("Already patched")
else:
    print("WARNING: Pattern not found - vLLM version may be different")
