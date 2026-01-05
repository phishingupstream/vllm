#!/usr/bin/env python3
"""Generate pre-built JSONL benchmark datasets.

Uses random words to avoid expensive tokenizer overhead in GuideLLM.
Calibrated for Qwen3.5 tokenizer (~1.3 chat tokens per English word).

Usage:
    python3 tests/bench_data/generate.py                    # generate all sizes
    python3 tests/bench_data/generate.py --validate http://vllm.localhost  # generate + validate token counts
"""

import argparse
import json
import random
from pathlib import Path

# Common English words — each ~1.3 tokens with Qwen3.5 chat template overhead
WORDS = (
    "the of and to a in is it you that he was for on are with as his they be "
    "at one have this from or had by not but what all were when we there can an "
    "your which their said if do will each about how up out them then she many "
    "some so these would other into has her two like him see time could no make "
    "than first been its who now people my made over did down only way find use "
    "may water long little very after words called just where most know get "
    "through back much before also around another came come work three word must "
    "because does part even place well such here take why things help put years "
    "different away again off went old number great tell men say small every "
    "found next still between name should home big give air line set own under "
    "read last never us left end along while might close something seem thought "
    "both few those always looked show large often together asked house don "
    "point world going want school important until form food keep children feet "
    "land side without boy once animal life enough took four head above kind "
    "began almost live page got earth need far hand high year mother light "
    "country father let night picture being study second soon story since white "
    "ever paper hard near sentence better best across during today however sure "
    "knew before face anything"
).split()

# Calibrated for Qwen3.5: ~1.3 chat tokens per word (including template overhead)
TOKENS_PER_WORD = 1.3

# Benchmark sizes: target prompt tokens -> (num_prompts, output_tokens)
SIZES = {
    128: (10, 128),
    1000: (10, 50),
    4000: (10, 50),
    8000: (10, 50),
    16000: (10, 50),
    32000: (5, 50),
    64000: (3, 50),
    96000: (2, 50),
    130000: (1, 20),
}


def gen_prompt(target_tokens: int) -> str:
    n_words = int(target_tokens / TOKENS_PER_WORD)
    text = " ".join(random.choices(WORDS, k=n_words))
    return f"Summarize the following text in one sentence:\n\n{text}"


def generate_all(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)  # Reproducible

    for target_tokens, (num_prompts, output_tokens) in sorted(SIZES.items()):
        path = output_dir / f"prompts_{target_tokens}.jsonl"
        with open(path, "w") as f:
            for _ in range(num_prompts):
                json.dump({"prompt": gen_prompt(target_tokens), "output_tokens": output_tokens}, f)
                f.write("\n")
        fsize = path.stat().st_size
        print(f"  {target_tokens:>7} tokens -> {path.name} ({num_prompts} prompts, {fsize:,} bytes)")


def validate(output_dir: Path, url: str):
    import httpx

    print(f"\nValidating token counts against {url}...")
    for target_tokens in sorted(SIZES.keys()):
        path = output_dir / f"prompts_{target_tokens}.jsonl"
        with open(path) as f:
            line = json.loads(f.readline())

        r = httpx.post(f"{url}/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": line["prompt"]}],
            "max_tokens": 1,
        }, timeout=120)
        actual = r.json().get("usage", {}).get("prompt_tokens", "?")
        pct = (actual / target_tokens * 100) if isinstance(actual, int) else 0
        status = "OK" if 70 < pct < 140 else "WARN"
        print(f"  [{status}] target={target_tokens:>7}  actual={actual:>7}  ({pct:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", metavar="URL", help="Validate token counts against vLLM URL")
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    generate_all(output_dir)

    if args.validate:
        validate(output_dir, args.validate)
