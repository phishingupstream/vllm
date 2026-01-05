#!/usr/bin/env python3
"""Test: Gemma 4 reasoning/thinking separation across all endpoints.

Checks three things:
1. Thinking doesn't leak into content (non-streaming bug: skip_special_tokens
   strips delimiters before parser sees them)
2. No raw channel tokens in reasoning or content deltas
3. No "thought\n" prefix leaked into either field

Usage:
    python3 tests/test_gemma4_reasoning.py [--url URL]
"""

from vllm_client import VllmClient, argparser, main_exit, run_matrix

PROMPT = "What is 2+2? Answer in one word."
RAW_TOKENS = ["<|channel>", "<channel|>"]


def test_reasoning(client: VllmClient, endpoint: str, stream: bool) -> tuple[bool, str]:
    if endpoint == "openai":
        if stream:
            reasoning, content = "", ""
            for chunk in client.openai.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=512, stream=True,
            ):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning += getattr(delta, "reasoning", "") or ""
                content += delta.content or ""
        else:
            r = client.openai.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=512,
            )
            msg = r.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", "") or ""
    else:
        if stream:
            reasoning, content = "", ""
            with client.anthropic.messages.stream(
                model="default",
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=512,
            ) as s:
                for event in s:
                    if event.type != "content_block_delta":
                        continue
                    if event.delta.type == "thinking_delta":
                        reasoning += event.delta.thinking
                    elif event.delta.type == "text_delta":
                        content += event.delta.text
        else:
            r = client.anthropic.messages.create(
                model="default",
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=512,
            )
            reasoning, content = "", ""
            for block in r.content:
                if block.type == "thinking":
                    reasoning += block.thinking
                elif block.type == "text":
                    content += block.text

    issues = []

    # Thinking leaked into content
    if content.startswith("thought\n"):
        issues.append("content starts with 'thought\\n' (thinking leaked)")
    if not reasoning and len(content) > 100:
        issues.append(f"no reasoning but content={len(content)} chars (likely leaked)")

    # Raw delimiter tokens
    for tok in RAW_TOKENS:
        if tok in content:
            issues.append(f"raw {tok} in content")
        if tok in reasoning:
            issues.append(f"raw {tok} in reasoning")

    # thought\n prefix not stripped
    if reasoning.startswith("thought\n"):
        issues.append("reasoning starts with 'thought\\n' (prefix not stripped)")

    if issues:
        return False, "; ".join(issues) + f"\n  content={content[:120]!r}\n  reasoning={reasoning[:120]!r}"
    return True, f"content={content[:40]!r}, reasoning={len(reasoning)}ch"


if __name__ == "__main__":
    args = argparser("Test Gemma 4 reasoning/thinking separation").parse_args()
    print(f"vLLM: {args.url}\n")
    main_exit(run_matrix(VllmClient(args.url), test_reasoning))
