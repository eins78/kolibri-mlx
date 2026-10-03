"""Server smoke test: start serve.py, run German/English/tool-call requests, report.

Usage: uv run python scripts/smoke_server.py [--model models/Kolibri-1-4bit] [--no-assert]
Exits non-zero on HTTP errors, empty content or a missing/invalid tool call
(unless --no-assert, which only reports them).
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLING = {"temperature": 1.0, "top_p": 0.97, "top_k": 128}  # all three accepted by mlx-lm's server
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

CASES = [
    {
        "name": "de_reasoning_low",
        "prompt": "Erkläre in zwei Sätzen, warum der Himmel blau ist.",
        "lang": "de",
        "reasoning": "on",
        "extra": {"chat_template_kwargs": {"reasoning_effort": "low"}},
    },
    {
        "name": "en_no_reasoning",
        "prompt": "Give me three short tips for writing clear commit messages.",
        "lang": "en",
        "reasoning": "off",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    {
        "name": "de_tool_call",
        "prompt": "Wie ist das Wetter in Zürich?",
        "lang": "de",
        "reasoning": "off",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}, "tools": [WEATHER_TOOL]},
        "expect_tool": True,
    },
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rss_gb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return int(out.strip() or 0) / 1e6


def mem_free_pct() -> str:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return line.split(":")[-1].strip()
    return "n/a"


def http(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base, proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            http(f"{base}/v1/models", timeout=5)
            return time.time() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"server not ready after {timeout}s")


def run_case(base, model, case, max_tokens, pid):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": max_tokens,
        "stream": False,
        **SAMPLING,
        **case["extra"],
    }
    t0 = time.time()
    resp = http(f"{base}/v1/chat/completions", body)
    wall = time.time() - t0
    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp["usage"]
    res = {
        "name": case["name"],
        "prompt": case["prompt"],
        "reasoning_mode": case["reasoning"],
        "message": msg,
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "wall_s": round(wall, 2),
        "tokens_per_s": round(usage["completion_tokens"] / wall, 2) if wall > 0 else None,
        "server_rss_gb": round(rss_gb(pid), 2),
        "mem_free_pct": mem_free_pct(),
        "failures": [],
    }
    tool_calls = msg.get("tool_calls") or []
    if case.get("expect_tool"):
        ok = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                args = fn.get("arguments")
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = None
            # The model may write "Zurich" or "Zürich"; both name the city.
            city = json.dumps(args, ensure_ascii=False).lower() if isinstance(args, dict) else ""
            if fn.get("name") == "get_weather" and ("zürich" in city or "zurich" in city):
                ok = True
        res["tool_call_ok"] = ok
        if not ok:
            res["failures"].append("no valid get_weather tool call mentioning Zurich")
    else:
        res["tool_call_ok"] = None
        if not (msg.get("content") or "").strip():
            res["failures"].append("empty content")
    return res


def show(r):
    print(f"\n=== {r['name']} ({r['prompt']!r}, reasoning {r['reasoning_mode']}) ===")
    m = r["message"]
    if m.get("reasoning"):
        print(f"[reasoning]\n{m['reasoning']}")
    print(f"[content]\n{m.get('content')}")
    if m.get("tool_calls"):
        print(f"[tool_calls] {json.dumps(m['tool_calls'], ensure_ascii=False)}")
    print(
        f"finish_reason={r['finish_reason']} prompt_tokens={r['prompt_tokens']} "
        f"completion_tokens={r['completion_tokens']} wall={r['wall_s']}s "
        f"tok/s={r['tokens_per_s']} server_rss={r['server_rss_gb']} GB mem_free={r['mem_free_pct']}"
    )
    if r["failures"]:
        print(f"FAILED: {r['failures']}")
        print(f"[raw assistant message] {json.dumps(m, ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Kolibri-1-4bit")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--ready-timeout", type=int, default=600)
    ap.add_argument("--no-assert", action="store_true", help="report failures but exit 0")
    ap.add_argument("--out", default=str(REPO / "verify/out/server-smoke.json"))
    args = ap.parse_args()

    model_path = str(Path(args.model).resolve())
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = REPO / "run" / "server-smoke.out"
    log_path.parent.mkdir(exist_ok=True)
    log_fh = open(log_path, "w")
    cmd = ["uv", "run", "python", str(REPO / "serve.py"), "--model", model_path, "--port", str(port)]
    print("starting:", " ".join(cmd), f"(server log: {log_path})", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT)
    results, failed, load_s = [], False, None
    try:
        load_s = wait_ready(base, proc, args.ready_timeout)
        print(f"server ready after {load_s:.0f}s", flush=True)
        # `uv run` is the direct child; measure the python process under it
        pid = proc.pid
        kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
        if kids:
            pid = int(kids[0])
        for case in CASES:
            try:
                r = run_case(base, model_path, case, args.max_tokens, pid)
            except Exception as e:  # HTTP error etc.
                r = {"name": case["name"], "prompt": case["prompt"], "reasoning_mode": case["reasoning"],
                     "message": {}, "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
                     "wall_s": None, "tokens_per_s": None, "server_rss_gb": None, "mem_free_pct": None,
                     "tool_call_ok": None, "failures": [f"request error: {e!r}"]}
            show(r)
            results.append(r)
    except Exception as e:
        print(f"ERROR: {e!r}", file=sys.stderr)
        failed = True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        # make sure the python child is gone too
        subprocess.run(["pkill", "-f", f"serve.py --model {model_path} --port {port}"], capture_output=True)
        log_fh.close()

    print("\n| prompt | reasoning | tool call ok | prompt tok | completion tok | tok/s | server RSS GB |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        tc = "-" if r["tool_call_ok"] is None else ("yes" if r["tool_call_ok"] else "NO")
        print(f"| {r['prompt']} | {r['reasoning_mode']} | {tc} | {r['prompt_tokens']} | "
              f"{r['completion_tokens']} | {r['tokens_per_s']} | {r['server_rss_gb']} |")

    any_fail = failed or len(results) != len(CASES) or any(r["failures"] for r in results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_path, "sampling": SAMPLING, "max_tokens": args.max_tokens,
                               "load_s": load_s, "no_assert": args.no_assert, "results": results},
                              indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")
    if any_fail and not args.no_assert:
        sys.exit(1)


if __name__ == "__main__":
    main()
