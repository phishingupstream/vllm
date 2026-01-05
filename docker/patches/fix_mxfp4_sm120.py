#!/usr/bin/env python3
"""Patch vLLM mxfp4.py to add SM120 (RTX 5090) support.

SM120 is not recognized by get_mxfp4_backend() — it falls through all
FlashInfer checks (SM90/SM100 only) and Triton checks (capped at SM110),
landing on Marlin kernels whose PTX doesn't support SM120.

Fix: Route SM120 to SM100_FI_MXFP4_BF16 (FlashInfer TRT-LLM with BF16
activations). SM90 CUTLASS path lacks SM120 tactics, but SM100 TRT-LLM
kernels work on Blackwell consumer GPUs.
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py"

with open(file_path, 'r') as f:
    content = f.read()

# Add SM120 to the SM100 family FlashInfer BF16 check
old_code = '''        elif current_platform.is_device_capability_family(100) and has_flashinfer():
            logger.info_once(
                "Using FlashInfer MXFP4 BF16 backend for SM100, "'''

new_code = '''        elif (current_platform.is_device_capability_family(100) or current_platform.is_device_capability(120)) and has_flashinfer():
            logger.info_once(
                "Using FlashInfer MXFP4 BF16 backend, "'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print("PATCHED: Added SM120 support to mxfp4.py get_mxfp4_backend() [SM100 BF16 path]")
elif new_code in content:
    print("Already patched")
else:
    print("WARNING: Pattern not found - vLLM version may be different")
