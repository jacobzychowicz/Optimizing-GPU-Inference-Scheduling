"""Generate workloads instead of writing them by hand.

narration: the traffic of a voice-cloning narration app. Each user pastes a long
script, which is split into one request per sentence, all arriving at once.
Meanwhile other users retake single lines, which they are waiting on. With
first-come-first-served, a retake waits behind every segment of every script
ahead of it.

traffic: generic requests arriving steadily, in bursts, or ramping up, with
long-tailed lengths, for language models (max_tokens) or speech (text).
"""

import math
import random

from gpu_bench.compare import duration_ms

_WORDS = (
    "the kingdom believed their greatest warrior had disappeared but what they did not know was that he "
    "had been training in silence for years every night the system message returned with a new quest and "
    "a stronger enemy waiting at the gate while the guild leaders argued about the dungeon break that "
    "changed everything for the hunters of the eastern district who never expected an ordinary student "
    "to rise this far before the final battle began under a red moon over the city"
).split()

PATTERNS = ("steady", "bursty", "ramp")


def sentence(rng, chars):
    """Narration-like text of about this many characters."""
    words = []
    length = 0
    while length < chars:
        word = rng.choice(_WORDS)
        words.append(word)
        length += len(word) + 1
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


def _lognormal_int(rng, median, sigma, low, high):
    return int(min(high, max(low, round(rng.lognormvariate(math.log(median), sigma)))))


def _estimate(model, job, fallback_ms):
    if model is not None:
        try:
            return duration_ms(model, job)
        except (KeyError, ValueError):
            pass
    return fallback_ms(job)


def generate_narration(
    name="narration",
    users=4,
    script_users=None,
    segments=(100, 400),
    retakes=20,
    median_chars=110,
    model=None,
    model_name=None,
    seed=7,
):
    """Long scripts from some users, single-line retakes from the others.

    users: how many people share the GPU.
    script_users: how many of them submit a full script (default: half, at least 1).
    segments: (min, max) sentences per script.
    retakes: how many single-line retakes arrive while the scripts are being voiced.
    model: model dict used to fill predicted_duration_ms (default: 40 + 3 ms per character).
    """
    rng = random.Random(seed)
    users = max(1, int(users))
    script_users = max(1, min(users, int(script_users if script_users is not None else max(1, users // 2))))
    low, high = sorted((max(1, int(segments[0])), max(1, int(segments[1]))))

    def fallback(job):
        return 40 + 3 * len(job["input"]["text"])

    scripts = []
    for index in range(script_users):
        user = f"user-{index + 1}"
        count = rng.randint(low, high)
        lines = [sentence(rng, _lognormal_int(rng, median_chars, 0.45, 20, 400)) for _ in range(count)]
        scripts.append((user, lines))

    jobs = []
    # The first script starts the run; the others arrive while it is being voiced.
    total_ms = sum(fallback({"input": {"text": line}}) for _user, lines in scripts for line in lines)
    start = 0
    for index, (user, lines) in enumerate(scripts):
        if index:
            start = int(rng.uniform(0, total_ms * 0.25))
        for number, line in enumerate(lines, start=1):
            jobs.append(
                {
                    "id": f"{user}-seg{number:03d}",
                    "user": user,
                    "kind": "script_segment",
                    "arrival_ms": start,
                    "priority": 1,
                    "input": {"text": line},
                }
            )

    retakers = [f"user-{index + 1}" for index in range(script_users, users)] or [user for user, _lines in scripts]
    for number in range(1, int(retakes) + 1):
        jobs.append(
            {
                "id": f"retake-{number:02d}",
                "user": rng.choice(retakers),
                "kind": "retake",
                "arrival_ms": int(rng.uniform(0, total_ms * 0.8)),
                "priority": 2,
                "input": {"text": sentence(rng, _lognormal_int(rng, median_chars, 0.45, 20, 400))},
            }
        )

    for job in jobs:
        job["predicted_duration_ms"] = round(_estimate(model, job, fallback))
    jobs.sort(key=lambda job: (job["arrival_ms"], job["id"]))
    described = ", ".join(f"{user} ({len(lines)} lines)" for user, lines in scripts)
    return {
        "name": name,
        "description": (
            f"Narration traffic: full scripts from {described} arrive as bursts of one request per sentence, "
            f"while {int(retakes)} single-line retakes arrive from other users who are waiting on them. "
            f"Generated with seed {seed}."
        ),
        "model": model_name or (model or {}).get("name") or "example-tts",
        "jobs": jobs,
    }


def generate_traffic(
    name="traffic",
    count=200,
    rate_per_s=4.0,
    pattern="steady",
    field="max_tokens",
    median_size=64,
    priorities=(1,),
    model=None,
    model_name=None,
    seed=7,
):
    """Requests arriving over time. field is max_tokens (language model) or text (speech)."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {', '.join(PATTERNS)}")
    if rate_per_s <= 0:
        raise ValueError("rate must be above 0")
    rng = random.Random(seed)
    jobs = []
    now = 0.0
    for number in range(1, int(count) + 1):
        if pattern == "steady":
            rate = rate_per_s
        elif pattern == "ramp":
            rate = rate_per_s * (0.25 + 1.75 * number / count)
        else:  # bursty: quiet stretches, then bursts at 5x the rate
            rate = rate_per_s * (5 if (number // 20) % 2 else 0.5)
        now += rng.expovariate(rate) * 1000
        size = _lognormal_int(rng, median_size, 0.8, 1, median_size * 16)
        if field == "text":
            payload = {"text": sentence(rng, size)}
        else:
            payload = {"prompt": "Continue the story.", field: size}
        jobs.append(
            {
                "id": f"req-{number:04d}",
                "arrival_ms": int(now),
                "priority": rng.choice(list(priorities)),
                "input": payload,
            }
        )

    def fallback(job):
        payload = job["input"]
        return 40 + 3 * len(payload["text"]) if "text" in payload else 30 + 25 * payload[field]

    for job in jobs:
        job["predicted_duration_ms"] = round(_estimate(model, job, fallback))
    return {
        "name": name,
        "description": f"{count} requests, {pattern} arrivals around {rate_per_s:g}/s, long-tailed sizes. Seed {seed}.",
        "model": model_name or (model or {}).get("name") or ("example-tts" if field == "text" else "example-llm"),
        "jobs": jobs,
    }
