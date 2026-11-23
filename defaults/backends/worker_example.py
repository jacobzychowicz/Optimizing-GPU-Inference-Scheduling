"""A worker process that speaks the gpu-bench worker protocol. Replace the body of handle().

Protocol, one JSON object per line:
  worker -> gpu-bench   {"ready": true}            once the model is loaded
  gpu-bench -> worker   {"job": {...}}              one request
  worker -> gpu-bench   {"ok": true} or {"error": "..."}

Any language works; this one is Python so it runs anywhere. It sleeps for the
same time as the example-llm formula so you can check a setup (or an SSH host)
end to end before pointing a model file at your own server process.
Print logs to stderr, never stdout.
"""

import json
import sys
import time


def load_model():
    # Load weights, move them to the GPU, compile, etc.
    time.sleep(0.2)
    return {"base_ms": 30, "per_token_ms": 25}


def handle(model, job):
    tokens = job["input"].get("max_tokens", 16)
    time.sleep((model["base_ms"] + model["per_token_ms"] * tokens) / 1000.0)


def main():
    model = load_model()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            handle(model, json.loads(line)["job"])
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
        else:
            print(json.dumps({"ok": True}), flush=True)


if __name__ == "__main__":
    main()
