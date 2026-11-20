#!/usr/bin/env python3
"""Run this BEFORE run_phase3_ollama_experiment.py. Confirms Ollama is
reachable and the requested model is actually pulled, and does one tiny
real inference call (forced CPU) to measure a realistic per-call latency --
so you can sanity-check the --max-minutes budget before committing 60 real
tasks to it. Exits non-zero on any failure, with a plain-language reason.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="qwen2.5:7b")
    args = ap.parse_args()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    print(f"1) Checking {args.url}/api/tags ...")
    try:
        with opener(args.url.rstrip("/") + "/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"   FAILED: Ollama is not reachable at {args.url} ({type(e).__name__}: {e})")
        print("   -> Is Ollama running? Try `ollama serve` or check it's already running as a service.")
        sys.exit(1)
    names = [m.get("name", "") for m in tags.get("models", [])]
    print(f"   OK. Installed models: {names}")

    if not any(n == args.model or n.split(":")[0] == args.model.split(":")[0] for n in names):
        print(f"   FAILED: model '{args.model}' not found among installed models.")
        print(f"   -> Try: ollama pull {args.model}")
        sys.exit(1)
    print(f"2) Model '{args.model}' is installed.")

    print("3) Running one real, tiny, CPU-forced inference call to measure latency...")
    body = json.dumps({
        "model": args.model, "stream": False, "keep_alive": "5m",
        "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
        "options": {"temperature": 0.1, "num_predict": 10, "num_gpu": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        args.url.rstrip("/") + "/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with opener(req, timeout=120) as r:
            out = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"   FAILED: inference call errored ({type(e).__name__}: {e})")
        sys.exit(1)
    elapsed = time.monotonic() - t0
    reply = ((out.get("message") or {}).get("content") or "").strip()
    print(f"   OK. Reply: {reply!r}  (took {elapsed:.1f}s on CPU)")

    # Phase 3 makes ~1 call per quick decision and ~1-2 per deep decision,
    # roughly 1.5x per task on average across ~60 tasks => ~90 calls.
    est_total_minutes = elapsed * 90 / 60.0
    print(f"\nRough estimate for a 60-task run (~90 model calls total): "
          f"~{est_total_minutes:.0f} minutes at this latency.")
    print("If that's far outside your 30-120 minute budget, consider a smaller "
          "--n-per-bucket or raising --per-call-timeout before the real run.")


if __name__ == "__main__":
    main()
