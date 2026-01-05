#!/usr/bin/env python3
"""Patch vLLM FA2 attention backend selection for SM120 (RTX 5090 / Blackwell).

The FA2 varlen_fwd kernel's PTX is compiled for an unsupported toolchain
on SM120. FlashInfer also has GQA shape mismatches with some VLM models
(e.g. Qwen3-VL: expected [8,8,128] got [8,32,128] during warmup).

Fixes:
- VIT backends: Reorder so TRITON_ATTN is tried first on SM120
- Main model backends: On SM120 prefer FLASHINFER but put TRITON_ATTN
  before FLASH_ATTN as fallback (FA2 PTX is broken)
"""

file_path = "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py"
patched = []

with open(file_path, 'r') as f:
    content = f.read()

# Patch 1: VIT attention backends — prefer TRITON_ATTN on SM120
old_vit = """    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]"""

new_vit = """    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        # SM120 (Blackwell consumer): FA2 varlen PTX crashes, prefer Triton/SDPA
        cc = cls.get_device_capability()
        if cc is not None and cc[0] >= 12:
            return [
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.FLASH_ATTN,
            ]
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]"""

if old_vit in content:
    content = content.replace(old_vit, new_vit)
    patched.append("VIT attention")

# Patch 2: Main model backends — SM120 needs FA2 deprioritized
# FlashInfer first (works for most models), then Triton, FA2 last
old_main = """    else:
        if device_capability.major == 10:
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]"""

new_main = """    else:
        if device_capability.major >= 10:
            # SM100 (Hopper) and SM120 (Blackwell): FlashInfer first
            # FA2 varlen PTX is broken on SM120, Triton as safe fallback
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]"""

if old_main in content:
    content = content.replace(old_main, new_main)
    patched.append("main model attention")

if patched:
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"PATCHED: SM120 attention backends fixed for: {', '.join(patched)}")
elif "SM120 (Blackwell consumer)" in content:
    print("Already patched")
else:
    print("WARNING: One or more patterns not found - vLLM version may be different")
