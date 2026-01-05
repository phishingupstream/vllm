# google/gemma-4-26B-A4B-it

Source: https://huggingface.co/google/gemma-4-26B-A4B-it
Downloaded: 2026-04-03

## Architecture

- **Type**: Mixture-of-Experts (MoE), multimodal (text + image)
- **Total params**: 25.2B
- **Active params**: 3.8B (8 active / 128 total experts + 1 shared)
- **Layers**: 30
- **Hidden size**: 2816
- **Intermediate size**: 2112 (dense), 704 (MoE)
- **Attention heads**: 16
- **KV heads**: 8 (standard), 2 (global)
- **Head dim**: 256 (sliding), 512 (global)
- **Vocab size**: 262,144
- **Context**: 256K tokens
- **Sliding window**: 1024 tokens
- **License**: Apache 2.0

### Layer Types (30 layers)

```
0-4:   sliding_attention
5:     full_attention
6-10:  sliding_attention
11:    full_attention
12-16: sliding_attention
17:    full_attention
18-22: sliding_attention
23:    full_attention
24-28: sliding_attention
29:    full_attention
```

5 sliding + 1 full, repeated 5 times = 25 sliding + 5 full attention layers.

### RoPE

- **Sliding attention**: default, theta=10,000
- **Full attention**: proportional, theta=1,000,000, partial_rotary_factor=0.25

### Vision Encoder

- **Params**: ~550M
- **Layers**: 27
- **Hidden size**: 1152
- **Attention heads**: 16 (all MHA, no GQA)
- **Head dim**: 72
- **Patch size**: 16
- **Soft tokens per image**: 280 (configurable: 70, 140, 280, 560, 1120)
- **Max position embeddings**: 131,072

### Special Tokens

| Token | ID |
|-------|-----|
| image_token | 258880 |
| audio_token | 258881 |
| eoi_token | 258882 |
| eoa_token | 258883 |
| video_token | 258884 |
| EOS | 1, 106 |
| BOS | 2 |
| PAD | 0 |

## Recommended Sampling

- temperature: 1.0
- top_p: 0.95
- top_k: 64

## Thinking Mode

- **Enable**: Include `<|think|>` at start of system prompt
- **Disable**: Remove the token
- **Output format**: `<|channel>thought\n[reasoning]<channel|>[answer]`
- **Multi-turn**: Strip thinking from history, only keep final response

## Best Practices

1. Place image/audio content **before** text in prompts
2. Use variable image token budgets (70-1120) based on task complexity
3. Lower budgets for classification/captioning, higher for OCR/document parsing
4. Video: max 60 seconds at 1 fps
5. No thinking content in multi-turn history

## Benchmarks (instruction-tuned)

| Benchmark | Score |
|-----------|-------|
| MMLU Pro | 82.6% |
| AIME 2026 (no tools) | 88.3% |
| LiveCodeBench v6 | 77.1% |
| Codeforces ELO | 1718 |
| GPQA Diamond | 82.3% |
| MMMU Pro (vision) | 73.8% |
| MATH-Vision | 82.4% |
| MRCR v2 8-needle 128K | 44.1% |
| BigBench Extra Hard | 64.8% |
| MMMLU | 86.3% |

## config.json

```json
{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "model_type": "gemma4",
  "dtype": "bfloat16",
  "text_config": {
    "num_hidden_layers": 30,
    "hidden_size": 2816,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 2,
    "head_dim": 256,
    "global_head_dim": 512,
    "intermediate_size": 2112,
    "moe_intermediate_size": 704,
    "num_experts": 128,
    "top_k_experts": 8,
    "enable_moe_block": true,
    "sliding_window": 1024,
    "max_position_embeddings": 262144,
    "hidden_activation": "gelu_pytorch_tanh",
    "final_logit_softcapping": 30.0,
    "attention_k_eq_v": true,
    "vocab_size": 262144,
    "tie_word_embeddings": true,
    "layer_types": [
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention"
    ],
    "rope_parameters": {
      "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
      },
      "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
      }
    }
  },
  "vision_config": {
    "model_type": "gemma4_vision",
    "num_hidden_layers": 27,
    "hidden_size": 1152,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 72,
    "intermediate_size": 4304,
    "patch_size": 16,
    "max_position_embeddings": 131072,
    "default_output_length": 280
  },
  "vision_soft_tokens_per_image": 280,
  "image_token_id": 258880,
  "video_token_id": 258884,
  "eos_token_id": [1, 106]
}
```
