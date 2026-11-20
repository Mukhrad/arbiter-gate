"""Minimal real HTTP client for a local Ollama server — same endpoint shape
AMG Studio's own `llm_bridge.py` already uses (`/api/chat`, stream=false),
so this experiment adds no new dependency or attack surface to the
founder's machine. Deliberately stdlib-only (urllib), no `requests`.

Safety: this client only ever returns the RAW TEXT the model produced. It
never executes anything. The caller (code_repair_llm_tasks.py /
run_phase3_ollama_experiment.py) is responsible for parsing that text as a
single letter choice among pre-written candidates and never as code.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


def chat(
    prompt: str,
    model: str = "qwen2.5:7b",
    system: str = "",
    url: str = DEFAULT_URL,
    timeout: float = 120.0,
    force_cpu: bool = True,
    temperature: float = 0.1,
    num_predict: int = 200,
) -> tuple[str, float]:
    """One real, synchronous call to the local Ollama server. Returns
    (reply_text, wall_clock_seconds). Raises OllamaError on any failure —
    callers decide how to handle that (e.g. fall back to a default
    candidate), never silently execute something unvetted instead."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options = {"temperature": temperature, "num_predict": num_predict}
    if force_cpu:
        options["num_gpu"] = 0  # avoid competing with CEL-0 for GPU memory, per founder's explicit requirement
    body = json.dumps({
        "model": model, "messages": messages, "stream": False,
        "keep_alive": "5m", "options": options,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open  # local only, never via a proxy
    t0 = time.monotonic()
    try:
        with opener(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise OllamaError(f"{type(e).__name__}: {e}") from e
    elapsed = time.monotonic() - t0
    reply = ((out.get("message") or {}).get("content") or "").strip()
    return reply, elapsed


def parse_letter_choice(reply_text: str, valid_labels: list) -> str | None:
    """Extracts a single valid letter choice from free model text. Returns
    None (never a guess) if no unambiguous valid letter is found — callers
    must have an explicit, logged fallback for that case, never silently
    executing anything other than one of the pre-written candidates."""
    import re
    # Prefer an explicit "Answer: X" / "Choice: X" style line if present.
    for pattern in (r"(?:answer|choice|final answer)\s*[:\-]?\s*([A-Z])\b",
                     r"\b([A-Z])\b(?:\s*$)"):
        m = re.search(pattern, reply_text, re.IGNORECASE | re.MULTILINE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_labels:
                return letter
    # Fallback: first standalone valid letter token anywhere in the text.
    found = [tok for tok in re.findall(r"\b([A-Z])\b", reply_text) if tok in valid_labels]
    if len(set(found)) == 1:
        return found[0]
    return None
