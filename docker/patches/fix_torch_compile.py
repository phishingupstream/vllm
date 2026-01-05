#!/usr/bin/env python3
"""Patch vLLM to handle NVIDIA PyTorch builds that lack assume_32bit_indexing config.

The issue: vLLM checks `is_torch_equal_or_newer("2.10.0.dev")` to decide whether
to use `assume_32bit_indexing`, but NVIDIA's PyTorch 2.10.0a0 build passes this
check while lacking the actual config option.

Fix: Use try/except to check if the config exists instead of version checking.
"""

import sys

FILE_PATH = '/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py'

OLD_CODE = '''        # Prepare inductor config patches
        # assume_32bit_indexing is only available in torch 2.10.0.dev+
        inductor_config_patches = {}
        if is_torch_equal_or_newer("2.10.0.dev"):
            inductor_config_patches["assume_32bit_indexing"] = True'''

NEW_CODE = '''        # Prepare inductor config patches
        # assume_32bit_indexing is only available in official torch 2.10.0.dev+
        # NVIDIA builds may not have this config even if version matches
        inductor_config_patches = {}
        try:
            _ = torch._inductor.config.assume_32bit_indexing
            inductor_config_patches["assume_32bit_indexing"] = True
        except AttributeError:
            logger.debug("assume_32bit_indexing config not available")'''

def main():
    try:
        with open(FILE_PATH, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {FILE_PATH}")
        sys.exit(1)

    if OLD_CODE in content:
        content = content.replace(OLD_CODE, NEW_CODE)
        with open(FILE_PATH, 'w') as f:
            f.write(content)
        print("✓ Patched vLLM torch.compile for NVIDIA PyTorch builds")
    elif NEW_CODE in content:
        print("✓ Already patched")
    else:
        print("✗ Could not find code to patch - vLLM version may have changed")
        sys.exit(1)

if __name__ == "__main__":
    main()
