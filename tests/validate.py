#!/usr/bin/env python3
"""
vLLM model capability regression validator.
Tests chat, streaming, tool calling, thinking, vision, and long context.

Usage:
    python3 tests/validate.py --model gemma-4-26b-a4b
    python3 tests/validate.py --model all
    python3 tests/validate.py --model gemma-4-26b-a4b --test thinking_on streaming
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from openai import OpenAI

DEFAULT_URL = "http://vllm.localhost"
TIMEOUT = 120
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

client: OpenAI = None
results = []


def result(name, passed, detail="", duration=None):
    tag = PASS if passed else FAIL
    dur = f" ({duration:.1f}s)" if duration else ""
    print(f"  [{tag}] {name}{dur}" + (f" — {detail}" if detail else ""))
    results.append((name, passed, detail))
    return passed


# ── Individual test functions ──────────────────────────────────────────────────

def test_basic_chat(model):
    t = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: hello"}],
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = r.choices[0].message.content or ""
        return result("basic_chat", len(content) > 0, repr(content[:60]), time.perf_counter() - t)
    except Exception as e:
        return result("basic_chat", False, str(e))


def test_streaming(model):
    t = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count to 5"}],
            max_tokens=32,
            stream=True,
        )
        chunks, tokens = 0, 0
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content or getattr(delta, "reasoning", None):
                chunks += 1
                tokens += 1
        ok = chunks > 2
        return result("streaming", ok, f"{tokens} tokens in {chunks} chunks", time.perf_counter() - t)
    except Exception as e:
        return result("streaming", False, str(e))


def test_thinking_on(model):
    t = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Is 17 prime? Think step by step."}],
            max_tokens=2048,
            stream=True,
            temperature=1.0, top_p=0.95,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": 20,
            },
            presence_penalty=1.5,
        )
        reasoning, content = "", ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning += getattr(delta, "reasoning", "") or ""
            content += delta.content or ""
        ok = len(reasoning) > 20 and len(content) > 0
        return result("thinking_on", ok,
                      f"reasoning={len(reasoning)}ch content={len(content)}ch", time.perf_counter() - t)
    except Exception as e:
        return result("thinking_on", False, str(e))


def test_thinking_off(model):
    t = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=32,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        reasoning, content = "", ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning += getattr(delta, "reasoning", "") or ""
            content += delta.content or ""
        ok = len(content) > 0 and len(reasoning) == 0
        return result("thinking_off", ok,
                      f"content={len(content)}ch reasoning={len(reasoning)}ch", time.perf_counter() - t)
    except Exception as e:
        return result("thinking_off", False, str(e))


def test_tool_calling(model):
    t = time.perf_counter()
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "description": "City name"}
            }, "required": ["city"]}
        }
    }]
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            max_tokens=128,
            tools=tools,
        )
        msg = r.choices[0].message
        if msg.tool_calls:
            fn = msg.tool_calls[0].function
            args = json.loads(fn.arguments or "{}")
            ok = fn.name == "get_weather" and "city" in args
            detail = f"name={fn.name} args={args}"
        else:
            ok = False
            detail = f"no tool_calls in response: {(msg.content or '')[:80]}"
        return result("tool_calling", ok, detail, time.perf_counter() - t)
    except Exception as e:
        return result("tool_calling", False, str(e))


def test_multi_turn(model):
    t = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "My name is Alice. Remember it."},
                {"role": "assistant", "content": "Got it, I'll remember your name is Alice."},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = (r.choices[0].message.content or "").lower()
        ok = "alice" in content
        return result("multi_turn", ok, repr(content[:80]), time.perf_counter() - t)
    except Exception as e:
        return result("multi_turn", False, str(e))


def test_long_context(model, length=16000):
    t = time.perf_counter()
    try:
        filler = ("The quick brown fox jumps over the lazy dog. " * 400)[:length]
        needle = "SECRETWORD42"
        prompt = f"{filler}\n\nRemember this: {needle}\n\n{filler}\n\nWhat was the special word?"
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = r.choices[0].message.content or ""
        ok = needle.lower() in content.lower() or "42" in content
        return result(f"long_context_{length//1000}k", ok, repr(content[:60]), time.perf_counter() - t)
    except Exception as e:
        return result(f"long_context_{length//1000}k", False, str(e))


def test_vision(model):
    t = time.perf_counter()
    # 1x1 red PNG
    RED_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED_PNG}"}},
                {"type": "text", "text": "What colour is this image? One word answer."}
            ]}],
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = (r.choices[0].message.content or "").lower()
        colors = ["red", "yellow", "orange", "pink", "crimson", "scarlet"]
        ok = any(c in content for c in colors)
        return result("vision", ok, repr(content[:60]), time.perf_counter() - t)
    except Exception as e:
        return result("vision", False, str(e))


# ── Test dispatch ──────────────────────────────────────────────────────────────

TEST_FNS = {
    "basic_chat": test_basic_chat,
    "streaming": test_streaming,
    "thinking_on": test_thinking_on,
    "thinking_off": test_thinking_off,
    "tool_calling": test_tool_calling,
    "multi_turn": test_multi_turn,
    "long_context_16k": lambda m: test_long_context(m, 16000),
    "long_context_64k": lambda m: test_long_context(m, 64000),
    "vision": test_vision,
}


def run_model(model_name, capabilities, requested_tests=None, label=None):
    global results
    results = []
    print(f"\n── {label or model_name} ──────────────────────────────────────────")
    for test_name, fn in TEST_FNS.items():
        if requested_tests and test_name not in requested_tests:
            continue
        if test_name not in capabilities:
            if not requested_tests:
                print(f"  [{SKIP}] {test_name}")
            continue
        fn(model_name)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  Result: {passed}/{total} passed")
    return passed == total


def run_all(url=None, model="nemotron-30b", tests=None, config=None):
    """Run validation and return structured results. For programmatic use."""
    global client, results
    base_url = (url or DEFAULT_URL).rstrip("/")
    client = OpenAI(base_url=f"{base_url}/v1", api_key="x", timeout=TIMEOUT)

    cfg_path = Path(config) if config else Path(__file__).parent / "models.yaml"
    if not cfg_path.exists():
        return {"error": f"Config not found: {cfg_path}"}

    with open(cfg_path) as f:
        model_configs = yaml.safe_load(f)

    try:
        available = [m.id for m in client.models.list().data]
        print(f"Available models: {available}")
    except Exception as e:
        return {"error": f"Cannot reach {base_url}: {e}"}

    models_to_test = list(model_configs.keys()) if model == "all" else [model]
    all_results = {}
    all_passed = True
    for model_name in models_to_test:
        if model_name not in model_configs:
            print(f"No config for '{model_name}' in models.yaml — skipping")
            continue
        cfg = model_configs[model_name]
        if cfg.get("status") in ("BROKEN", "IMPRACTICAL"):
            print(f"  [{SKIP}] {model_name} — status: {cfg['status']}")
            continue
        caps = cfg.get("capabilities", [])
        served = cfg.get("served_name", model_name)
        passed = run_model(served, caps, tests, label=model_name)
        all_results[model_name] = [
            {"name": n, "passed": p, "detail": d} for n, p, d in results
        ]
        all_passed = all_passed and passed

    return {
        "passed": all_passed,
        "available_models": available,
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM regression validator")
    parser.add_argument("--model", default="nemotron-30b", help="Served model name or 'all'")
    parser.add_argument("--url", default=DEFAULT_URL, help="vLLM base URL")
    parser.add_argument("--test", nargs="*", help="Run specific tests only")
    parser.add_argument("--config", default=Path(__file__).parent / "models.yaml")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    result = run_all(url=args.url, model=args.model, tests=args.test,
                     config=str(args.config))

    if "error" in result:
        print(result["error"])
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
