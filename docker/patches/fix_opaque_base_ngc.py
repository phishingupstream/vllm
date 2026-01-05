#!/usr/bin/env python3
"""Patch vLLM torch_utils.py for NGC PyTorch compatibility.

NGC PyTorch 2.11.0a0 passes vLLM's is_torch_equal_or_newer("2.11.0.dev")
check but doesn't ship the private torch._opaque_base module that upstream
PyTorch 2.11 nightlies include. This causes an ImportError at startup.

Fix: Wrap the _opaque_base import in a try/except so it gracefully falls
back to OpaqueBase = object when the module is missing.
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/utils/torch_utils.py"

with open(file_path, 'r') as f:
    content = f.read()

old_code = '''if HAS_OPAQUE_TYPE:
    from torch._opaque_base import OpaqueBase
else:
    OpaqueBase = object  # type: ignore[misc, assignment]'''

new_code = '''if HAS_OPAQUE_TYPE:
    try:
        from torch._opaque_base import OpaqueBase
    except ImportError:
        HAS_OPAQUE_TYPE = False
        OpaqueBase = object  # type: ignore[misc, assignment]
else:
    OpaqueBase = object  # type: ignore[misc, assignment]'''

if old_code in content:
    content = content.replace(old_code, new_code)
    # Also fix the second HAS_OPAQUE_TYPE block (register_opaque_type)
    old_register = '''if HAS_OPAQUE_TYPE:
    from torch._library.opaque_object import register_opaque_type

    register_opaque_type(ModuleName, typ="value", hoist=True)'''

    new_register = '''if HAS_OPAQUE_TYPE:
    try:
        from torch._library.opaque_object import register_opaque_type
        register_opaque_type(ModuleName, typ="value", hoist=True)
    except ImportError:
        pass'''

    if old_register in content:
        content = content.replace(old_register, new_register)

    with open(file_path, 'w') as f:
        f.write(content)
    print("PATCHED: Wrapped torch._opaque_base import in try/except for NGC PyTorch compatibility")
elif "except ImportError" in content and "_opaque_base" in content:
    print("Already patched")
else:
    print("WARNING: Pattern not found - vLLM version may be different")
