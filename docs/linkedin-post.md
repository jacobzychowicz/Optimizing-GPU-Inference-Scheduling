# LinkedIn post draft

Replace the [bracketed] numbers with the live GPU run (docs/results.md). Attach docs/images/timeline.png or charts.png, or a 10-second screen recording of the app running on the server.

---

A creator fixing one line of narration shouldn't have to wait behind someone else's 40-minute script.

While building RecapVox, a voice-cloning narration app, I hit a classic GPU problem. A long script turns into hundreds of TTS requests at once. With first-come-first-served, a one-line retake waited [13.8 s] behind them.

So I built a tool to measure scheduling policies on real GPUs:

🔹 Replays realistic traffic (long scripts + interactive retakes) on any NVIDIA server over SSH
🔹 Runs a real voice-cloning model (XTTS-v2) and samples GPU use
🔹 Calibrates a simulator to the hardware, so new policies can be tested in seconds

Result on a [GPU]: a fair-share scheduler cut retake latency from [13.8 s] to [0.65 s] ([21×]), with the same throughput. Scripts finished [15%] later.

Biggest lesson: the "best" policy depends on who's waiting. Averages hide it.

Desktop app, CLI, and code: [repo link]

#GPU #MachineLearning #Python #SystemsEngineering #MLOps

---

Notes:
- Lead with the human problem; the tech goes in the bullets.
- Keep the real numbers; don't round up. Recruiters and engineers both trust specifics.
- If RecapVox should stay unnamed, use "a voice-cloning app I'm building".
- First comment: link the repo and docs/results.md (LinkedIn shows posts without links in the body to more people).
