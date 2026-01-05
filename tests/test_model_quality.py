#!/usr/bin/env python3
"""Test LLM quality: off-by-one errors, fence-post, day-of-week, scheduling.

Runs each test multiple times (default 3) to catch non-determinism.
Saves results to JSON for tracking quality across model changes.

Usage:
    python3 tests/test_model_quality.py [--url URL] [--runs N] [--output PATH]
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_URL = "http://vllm.localhost"


@dataclass
class TestCase:
    name: str
    messages: list
    check: callable  # (content) -> (passed, detail)
    category: str = "general"


def query(client: OpenAI, messages: list, max_tokens: int = 8192) -> tuple[str, int, int]:
    """Returns (content, completion_tokens, duration_ms)."""
    start = time.perf_counter()
    r = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=max_tokens,
        temperature=1.0, top_p=0.95,
        presence_penalty=1.5,
        extra_body={"top_k": 20},
    )
    duration = int((time.perf_counter() - start) * 1000)
    content = (r.choices[0].message.content or "").strip()
    tokens = r.usage.completion_tokens if r.usage else 0
    return content, tokens, duration


# ── Check functions ──────────────────────────────────────────────

def check_day(c):
    c = c.lower()
    if "thursday" in c:
        return True, "Correctly identified Thursday"
    for d in ["monday", "tuesday", "wednesday", "friday", "saturday", "sunday"]:
        if d in c:
            return False, f"Said {d} instead of Thursday"
    return False, "Did not identify the day"


def check_next_wednesday(c):
    c = c.lower()
    if "march 11" in c or "11th" in c:
        return True, "Correctly said March 11"
    if "march 12" in c or "12th" in c:
        return False, "Off-by-one: said March 12 instead of March 11"
    return False, "Unclear answer"


def check_fence_post(c):
    if "5" in c:
        return True, "Correctly counted 5 days"
    if "4" in c:
        return False, "Fence-post error: said 4 instead of 5"
    return False, f"Unclear: {c[:100]}"


def check_items_between(c):
    if "5" in c:
        return True, "Correctly said 5 items"
    if "4" in c or "3" in c:
        return False, "Fence-post error"
    return False, f"Unclear: {c[:100]}"


def check_time_between(c):
    if "8" in c:
        return True, "Correctly said 8 hours"
    if "7" in c:
        return False, "Fence-post: said 7 hours"
    return False, f"Unclear: {c[:100]}"


def check_gym_schedule(c):
    c = c.lower()
    errors = []
    if "march 12" in c and "wednesday" in c:
        errors.append("March 12 is not Wednesday (March 11 is)")
    if "march 14" in c and "friday" in c:
        errors.append("March 14 is not Friday (March 13 is)")
    if not errors:
        return True, "Schedule looks correct"
    return False, "; ".join(errors)


def check_next_session(c):
    c = c.lower()
    if "friday" in c and ("march 13" in c or "13th" in c):
        return True, "Correctly said Friday March 13"
    if "thursday" in c:
        return False, "Thursday is not a gym day"
    if "march 14" in c:
        return False, "Off-by-one: March 14 is Saturday"
    return False, f"Unclear: {c[:100]}"


def check_uc_sc(c):
    if "HELLO WORLD" in c and "Hello world" in c:
        return True, "Correct UC and SC"
    errors = []
    if "HELLO WORLD" not in c:
        errors.append("UC wrong")
    if "Hello world" not in c:
        errors.append("SC wrong" + (" (title-cased)" if "Hello World" in c else ""))
    return False, "; ".join(errors)


def check_leap_year(c):
    c = c.lower().strip()
    if "not a leap year" in c or "is not" in c or "isn't" in c or c.startswith("no"):
        return True, "Correctly said not a leap year"
    if "is a leap year" in c or c.startswith("yes"):
        return False, "Wrong: said 2100 IS a leap year"
    return False, f"Unclear: {c[:100]}"


def check_days_in_feb(c):
    if "28" in c:
        return True, "Correctly said 28 days"
    if "29" in c:
        return False, "Wrong: said 29 days (not a leap year)"
    return False, f"Unclear: {c[:100]}"


def check_array_slice(c):
    if "30" in c and "40" in c and "50" in c:
        return True, "Correct slice [30,40,50]"
    if "20" in c and "30" in c and "40" in c:
        return False, "Off-by-one: included index 1"
    return False, f"Unclear: {c[:100]}"


# ── Test definitions ─────────────────────────────────────────────

TESTS = [
    TestCase("day_of_week_march_12",
             [{"role": "system", "content": "You are a helpful assistant. Be concise."},
              {"role": "user", "content": "What day of the week is March 12, 2026?"}],
             check_day, "day-of-week"),
    TestCase("next_wednesday_after_march_9",
             [{"role": "system", "content": "You are a scheduling assistant. Be concise."},
              {"role": "user", "content": "Monday is March 9, 2026. What date is the next Wednesday?"}],
             check_next_wednesday, "day-of-week"),
    TestCase("inclusive_day_count",
             [{"role": "system", "content": "Answer with just the number and brief explanation."},
              {"role": "user", "content": "How many days are there from March 1 to March 5, counting both the start and end days?"}],
             check_fence_post, "fence-post"),
    TestCase("items_between_inclusive",
             [{"role": "system", "content": "Answer concisely."},
              {"role": "user", "content": "In a 1-indexed list of 10 items, how many items are there from position 3 to position 7, inclusive?"}],
             check_items_between, "fence-post"),
    TestCase("time_between",
             [{"role": "system", "content": "Answer concisely."},
              {"role": "user", "content": "How many hours are there between 9:00 AM and 5:00 PM?"}],
             check_time_between, "fence-post"),
    TestCase("gym_week_schedule",
             [{"role": "system", "content": "You are a gym scheduling assistant. Today is Monday, March 9, 2026. The gym schedule is Monday, Wednesday, Friday."},
              {"role": "user", "content": "List the next 6 gym session dates with their days of the week, starting from today."}],
             check_gym_schedule, "scheduling"),
    TestCase("next_gym_session",
             [{"role": "system", "content": "You are a gym scheduling assistant. The gym schedule is Monday, Wednesday, Friday."},
              {"role": "user", "content": "Today is Wednesday, March 11, 2026 and I just finished my workout. When is my next gym session?"}],
             check_next_session, "scheduling"),
    TestCase("case_conversion",
             [{"role": "system", "content": "Follow instructions exactly. Show only the results."},
              {"role": "user", "content": 'Convert "hello world" to: 1) UPPER CASE (UC) 2) Sentence case (SC). Show each result on its own line.'}],
             check_uc_sc, "string-ops"),
    TestCase("leap_year_2100",
             [{"role": "system", "content": "Answer concisely."},
              {"role": "user", "content": "Is the year 2100 a leap year?"}],
             check_leap_year, "calendar"),
    TestCase("feb_2026_days",
             [{"role": "system", "content": "Answer with just the number."},
              {"role": "user", "content": "How many days are in February 2026?"}],
             check_days_in_feb, "calendar"),
    TestCase("python_slice",
             [{"role": "system", "content": "Answer concisely."},
              {"role": "user", "content": "Given arr = [10, 20, 30, 40, 50, 60], what does arr[2:5] return in Python?"}],
             check_array_slice, "indexing"),
]


# ── Runner ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM quality tests")
    parser.add_argument("--url", default=DEFAULT_URL, help="vLLM base URL")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.url}/v1", api_key="x", timeout=120)

    # Get model info
    try:
        models = client.models.list()
        model_id = models.data[0].id
    except Exception as e:
        print(f"Cannot reach {args.url}: {e}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Model Quality Test — {model_id}")
    print(f"{'='*70}")

    all_results = []
    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n── Run {run+1}/{args.runs} ──")
        run_results = []
        for test in TESTS:
            try:
                content, tokens, duration = query(client, test.messages)
                passed, detail = test.check(content)
            except Exception as e:
                content, tokens, duration = "", 0, 0
                passed, detail = False, f"ERROR: {e}"
            status = "\u2705" if passed else "\u274c"
            print(f"  {status} {test.name:30s} {tokens:>4d} tok {duration:>5d}ms  {detail}")
            run_results.append({
                "name": test.name, "category": test.category,
                "passed": passed, "detail": detail,
                "content": content[:300], "tokens": tokens, "duration_ms": duration,
            })
        all_results.append(run_results)

    # Summary
    total = len(TESTS) * args.runs
    passed = sum(1 for run in all_results for r in run if r["passed"])
    by_cat = {}
    for run in all_results:
        for r in run:
            by_cat.setdefault(r["category"], [0, 0])
            by_cat[r["category"]][0 if r["passed"] else 1] += 1

    print(f"\n{'─'*70}")
    print(f"TOTAL: {passed}/{total} passed ({100*passed//total}%)")
    for cat, (p, f) in sorted(by_cat.items()):
        print(f"  {cat:15s}: {p}/{p+f}")

    # Save
    outfile = args.output or f"test-results-{model_id.replace('/', '_')}.json"
    output = {
        "model": model_id, "url": args.url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": args.runs,
        "summary": {"total": total, "passed": passed, "failed": total - passed, "by_category": by_cat},
        "results": all_results,
    }
    with open(outfile, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nResults saved to: {outfile}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
