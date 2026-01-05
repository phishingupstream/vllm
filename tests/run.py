#!/usr/bin/env python3
"""Unified vLLM test runner — correctness + performance benchmarks.

Runs validate.py for content/correctness testing and GuideLLM for
performance benchmarking. Results are saved for long-term tracking.

Usage:
    python3 tests/run.py --suite quick               # validate + smoke bench (~30s)
    python3 tests/run.py --suite standard             # + stability + quality + throughput (~5min)
    python3 tests/run.py --suite full                 # + long context (~10min)
    python3 tests/run.py --suite quick --skip-validate  # benchmarks only
    python3 tests/run.py --suite standard --skip-quality  # skip quality tests
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_URL = "http://vllm.localhost"

# ── GuideLLM benchmark scenarios ────────────────────────────────────────────
# Pre-built JSONL datasets in tests/bench_data/ avoid expensive tokenizer
# overhead. Regenerate with: python3 tests/bench_data/generate.py

BENCH_DATA = Path(__file__).parent / "bench_data"

SCENARIOS = [
    {
        "name": "smoke",
        "description": "Smoke test (short prompts, synchronous)",
        "data": str(BENCH_DATA / "prompts_128.jsonl"),
        "profile": "synchronous",
        "max_requests": 5,
        "suites": ["quick", "standard", "full"],
    },
    {
        "name": "stability_1k",
        "description": "Stability: 1K token input",
        "data": str(BENCH_DATA / "prompts_1000.jsonl"),
        "profile": "synchronous",
        "max_requests": 3,
        "suites": ["standard", "full"],
    },
    {
        "name": "stability_4k",
        "description": "Stability: 4K token input",
        "data": str(BENCH_DATA / "prompts_4000.jsonl"),
        "profile": "synchronous",
        "max_requests": 3,
        "suites": ["standard", "full"],
    },
    {
        "name": "stability_8k",
        "description": "Stability: 8K token input",
        "data": str(BENCH_DATA / "prompts_8000.jsonl"),
        "profile": "synchronous",
        "max_requests": 3,
        "suites": ["standard", "full"],
    },
    {
        "name": "stability_16k",
        "description": "Stability: 16K token input",
        "data": str(BENCH_DATA / "prompts_16000.jsonl"),
        "profile": "synchronous",
        "max_requests": 3,
        "suites": ["standard", "full"],
    },
    {
        "name": "stability_32k",
        "description": "Stability: 32K token input",
        "data": str(BENCH_DATA / "prompts_32000.jsonl"),
        "profile": "synchronous",
        "max_requests": 3,
        "suites": ["standard", "full"],
    },
    {
        "name": "stability_64k",
        "description": "Stability: 64K token input",
        "data": str(BENCH_DATA / "prompts_64000.jsonl"),
        "profile": "synchronous",
        "max_requests": 3,
        "suites": ["standard", "full"],
    },
    {
        "name": "decode_speed",
        "description": "Decode speed (short input, longer output)",
        "data": str(BENCH_DATA / "prompts_128.jsonl"),
        "profile": "synchronous",
        "max_requests": 5,
        "suites": ["standard", "full"],
    },
    {
        "name": "throughput_sweep",
        "description": "Throughput sweep (rate sweep across load levels)",
        "data": str(BENCH_DATA / "prompts_128.jsonl"),
        "profile": "sweep",
        "max_seconds": 30,
        "suites": ["standard", "full"],
    },
    {
        "name": "long_96k",
        "description": "Long context: 96K token input",
        "data": str(BENCH_DATA / "prompts_96000.jsonl"),
        "profile": "synchronous",
        "max_requests": 2,
        "suites": ["full"],
    },
    {
        "name": "long_130k",
        "description": "Long context: 130K token input",
        "data": str(BENCH_DATA / "prompts_130000.jsonl"),
        "profile": "synchronous",
        "max_requests": 1,
        "suites": ["full"],
    },
]


def slugify(model_id):
    """Convert model ID to filesystem-safe slug."""
    return re.sub(r'[^a-z0-9._-]', '-', model_id.lower()).strip('-')


def find_processor_path(model_id):
    """Resolve HuggingFace cache path for a model ID (needed for HF_HUB_OFFLINE)."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"

    # model_id like "txn545/Qwen3.5-35B-A3B-NVFP4" -> "models--txn545--Qwen3.5-35B-A3B-NVFP4"
    cache_name = f"models--{model_id.replace('/', '--')}"
    model_dir = hub_dir / cache_name / "snapshots"

    if not model_dir.exists():
        return None

    # Use the most recent snapshot
    snapshots = sorted(model_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if snapshots:
        return str(snapshots[0])
    return None


def get_model_info(url):
    """Get model ID, HF root, and available models from vLLM."""
    r = requests.get(f"{url}/v1/models", timeout=10)
    r.raise_for_status()
    models = r.json()["data"]
    # Find first non-"default" model
    model_id = None
    hf_root = None
    for m in models:
        if m["id"] != "default":
            model_id = m["id"]
            hf_root = m.get("root", model_id)
            break
    if not model_id:
        model_id = models[0]["id"]
        hf_root = models[0].get("root", model_id)
    return model_id, hf_root, [m["id"] for m in models]


def run_validate(url, models, output_dir):
    """Run validate.py for given models and return results."""
    print(f"\n{'='*60}")
    print(f"  PHASE: Validation (validate.py)")
    print(f"  Models: {models}")
    print(f"{'='*60}\n")

    tests_dir = Path(__file__).parent
    sys.path.insert(0, str(tests_dir))

    from validate import run_all

    # Run each model and merge results
    merged = {"passed": True, "results": {}}
    for model in models:
        result = run_all(url=url, model=model)
        if "error" in result:
            print(f"  Validate error for {model}: {result['error']}")
            merged["passed"] = False
            continue
        merged["passed"] = merged["passed"] and result.get("passed", False)
        merged["results"].update(result.get("results", {}))
        if "available_models" not in merged:
            merged["available_models"] = result.get("available_models", [])

    output_file = output_dir / "validate.json"
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    return merged["passed"]


def run_guidellm_scenario(scenario, url, model, processor_path, output_dir):
    """Run a single GuideLLM benchmark scenario."""
    name = scenario["name"]
    print(f"\n  [{name}] {scenario['description']}...")

    output_file = output_dir / f"bench_{name}.json"

    cmd = [
        sys.executable, "-m", "guidellm", "benchmark", "run",
        "--target", url,
        "--model", model,
        "--data", scenario["data"],
        "--rate-type", scenario["profile"],
        "--output-path", str(output_file),
        "--disable-progress",
    ]

    if processor_path:
        cmd.extend(["--processor", processor_path])

    if "max_requests" in scenario:
        cmd.extend(["--max-requests", str(scenario["max_requests"])])
    if "max_seconds" in scenario:
        cmd.extend(["--max-seconds", str(scenario["max_seconds"])])

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=600,  # 10 min max per scenario
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"    FAIL ({elapsed:.1f}s): {result.stderr[-200:] if result.stderr else 'unknown error'}")
            return False

        print(f"    OK ({elapsed:.1f}s)")
        return True

    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (600s)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def run_benchmarks(suite, url, model, processor_path, output_dir):
    """Run GuideLLM benchmark scenarios for the given suite."""
    print(f"\n{'='*60}")
    print(f"  PHASE: Benchmarks (GuideLLM, suite={suite})")
    print(f"{'='*60}")

    scenarios = [s for s in SCENARIOS if suite in s["suites"]]
    passed = 0
    failed = 0

    for scenario in scenarios:
        ok = run_guidellm_scenario(scenario, url, model, processor_path, output_dir)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n  Benchmarks: {passed}/{passed + failed} scenarios completed")
    return failed == 0


def run_quality(url, output_dir, runs=1):
    """Run test_model_quality.py."""
    print(f"\n{'='*60}")
    print(f"  PHASE: Quality Tests")
    print(f"{'='*60}\n")

    tests_dir = Path(__file__).parent
    output_file = output_dir / "quality.json"

    cmd = [
        sys.executable, str(tests_dir / "test_model_quality.py"),
        "--url", url,
        "--runs", str(runs),
        "--output", str(output_file),
    ]

    result = subprocess.run(cmd, timeout=300)
    return result.returncode == 0



def extract_scenario_metrics(bench_file):
    """Extract compact metrics from a GuideLLM benchmark JSON file."""
    with open(bench_file) as f:
        data = json.load(f)

    # v0.5.x wraps in {metadata, args, benchmarks}; v0.3.x has {benchmarks} directly
    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        return None

    # Use last benchmark (final rate in sweeps, or the only one for synchronous)
    b = benchmarks[-1]

    # Metrics path is the same across versions
    m = b.get("metrics", {})

    def stat(key):
        s = m.get(key, {}).get("successful", {})
        if not s or s.get("count", 0) == 0:
            return None
        # Skip metrics where all values are 0 (v0.5.4 TTFT/ITL bug)
        if s.get("mean", 0) == 0 and s.get("max", 0) == 0:
            return None
        return {
            "mean": round(s.get("mean", 0), 2),
            "median": round(s.get("median", 0), 2),
            "p99": round(s.get("percentiles", {}).get("p99", 0), 2),
        }

    # Request count: v0.5.x=scheduler_metrics, v0.3.x=run_stats
    rm = (b.get("scheduler_metrics") or b.get("run_stats", {})).get("requests_made", {})

    result = {
        "requests": rm.get("successful", 0),
        "output_tok_s": stat("output_tokens_per_second"),
        "total_tok_s": stat("tokens_per_second"),
        "ttft_ms": stat("time_to_first_token_ms"),
        "itl_ms": stat("inter_token_latency_ms"),
        "time_per_token_ms": stat("time_per_output_token_ms"),
        "latency_s": stat("request_latency"),
    }

    # For sweeps, include all rates
    if len(benchmarks) > 1:
        result["sweep_points"] = len(benchmarks)

    # Remove None entries for cleaner output
    return {k: v for k, v in result.items() if v is not None}


def save_results(output_dir, model_id, hf_root, suite, phases):
    """Save compact results summary to results/ for git tracking."""
    results_dir = Path(__file__).parent.parent / "results"
    model_slug = slugify(model_id)
    date = datetime.now().strftime("%Y-%m-%d")
    dest = results_dir / model_slug / date
    dest.mkdir(parents=True, exist_ok=True)

    # Extract benchmark metrics from each bench_*.json
    scenarios = {}
    for bench_file in sorted(output_dir.glob("bench_*.json")):
        name = bench_file.stem.replace("bench_", "")
        metrics = extract_scenario_metrics(bench_file)
        if metrics:
            scenarios[name] = metrics

    # Build summary
    summary = {
        "model": model_id,
        "hf_root": hf_root,
        "date": date,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "suite": suite,
        "phases": {k: v["status"] for k, v in phases.items()},
        "scenarios": scenarios,
    }

    # Add quality results if present
    quality_file = output_dir / "quality.json"
    if quality_file.exists():
        with open(quality_file) as f:
            qdata = json.load(f)
        qs = qdata.get("summary", {})
        summary["quality"] = {
            "total": qs.get("total", 0),
            "passed": qs.get("passed", 0),
            "pass_rate": round(qs["passed"] / qs["total"], 3) if qs.get("total") else 0,
            "by_category": qs.get("by_category", {}),
        }

    # Add validate results if present
    validate_file = output_dir / "validate.json"
    if validate_file.exists():
        with open(validate_file) as f:
            vdata = json.load(f)
        test_count = sum(len(tests) for tests in vdata.get("results", {}).values())
        summary["validate"] = {
            "passed": vdata.get("passed", False),
            "test_count": test_count,
        }

    # Write summary (sort keys for clean git diffs)
    summary_file = dest / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"  Results saved: {summary_file}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Unified vLLM test runner")
    parser.add_argument("--suite", choices=["quick", "standard", "full"],
                        default="quick", help="Benchmark suite level (default: quick)")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"vLLM base URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=None,
                        help="Model name for validate.py (default: auto-detect)")
    parser.add_argument("--skip-quality", action="store_true",
                        help="Skip model quality tests (included in standard/full)")
    parser.add_argument("--quality-runs", type=int, default=1,
                        help="Number of quality test runs (default: 1)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip validation tests")
    parser.add_argument("--output-dir",
                        help="Output directory (default: benchmarks/<timestamp>)")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    # Health check
    print(f"vLLM Test Runner — suite={args.suite}")
    print(f"URL: {url}")
    try:
        r = requests.get(f"{url}/health", timeout=10)
        if r.status_code != 200:
            print(f"Server not healthy (HTTP {r.status_code})")
            sys.exit(1)
    except Exception as e:
        print(f"Server unreachable: {e}")
        sys.exit(1)

    # Get model info
    model_id, hf_root, available = get_model_info(url)
    print(f"Model: {model_id} ({hf_root})")
    print(f"Available: {available}")

    # Find processor path for GuideLLM (use HF root, not served name)
    processor_path = find_processor_path(hf_root)
    if processor_path:
        print(f"Processor: {processor_path}")
    else:
        print(f"Warning: No local processor found for {model_id}, GuideLLM will try to download")

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(__file__).parent.parent / "benchmarks"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Track results
    phases = {}
    exit_code = 0

    # Auto-detect validate models: find models.yaml entries whose served_name is available
    validate_models = []
    if args.model:
        validate_models = [args.model]
    else:
        import yaml
        models_yaml = Path(__file__).parent / "models.yaml"
        if models_yaml.exists():
            with open(models_yaml) as f:
                model_configs = yaml.safe_load(f)
            for name, cfg in model_configs.items():
                served = cfg.get("served_name", name)
                if served in available and cfg.get("status") not in ("BROKEN", "IMPRACTICAL"):
                    validate_models.append(name)
        if not validate_models:
            validate_models = ["all"]

    # Phase 1: Validation
    if not args.skip_validate:
        ok = run_validate(url, validate_models, output_dir)
        phases["validate"] = {"status": "passed" if ok else "failed"}
        if not ok:
            exit_code |= 1

    # Phase 2: Benchmarks
    ok = run_benchmarks(args.suite, url, "default", processor_path, output_dir)
    phases["benchmarks"] = {"status": "passed" if ok else "failed"}
    if not ok:
        exit_code |= 2

    # Phase 3: Quality (standard/full by default)
    if args.suite in ("standard", "full") and not args.skip_quality:
        ok = run_quality(url, output_dir, runs=args.quality_runs)
        phases["quality"] = {"status": "passed" if ok else "failed"}
        if not ok:
            exit_code |= 4

    # Save raw summary
    summary = {
        "timestamp": timestamp,
        "suite": args.suite,
        "model": model_id,
        "url": url,
        "phases": phases,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save compact results for git tracking
    print(f"\n{'='*60}")
    print(f"  Saving Results")
    print(f"{'='*60}")
    save_results(output_dir, model_id, hf_root, args.suite, phases)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for phase, info in phases.items():
        status = info["status"]
        icon = "OK" if status == "passed" else "FAIL"
        print(f"  [{icon}] {phase}")
    print(f"\n  Results: {output_dir}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
