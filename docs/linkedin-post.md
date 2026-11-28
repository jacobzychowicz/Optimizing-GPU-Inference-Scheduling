# LinkedIn post draft

Ready to post as written: every number below is from the simulator and reproducible with the commands in docs/results.md. Attach docs/images/app.png (the app) or docs/images/timeline.png. If you later do the live GPU run, use the "after a live run" variant at the bottom instead.

---

A creator fixing one line of narration shouldn't have to wait behind someone else's entire script.

While building RecapVox, a voice-cloning narration app, I hit a classic GPU problem. A long script turns into hundreds of TTS requests at once. With first-come-first-served, a one-line retake from another user waits behind all of them.

So I built a tool to compare GPU scheduling policies on realistic traffic:

🔹 Simulates hundreds of requests in milliseconds, or replays them live on a real GPU (locally or over SSH)
🔹 Runs a real voice-cloning model (XTTS-v2) and samples GPU utilization and memory
🔹 Calibrates the simulator against live runs, so new policies can be tested in seconds
🔹 Desktop app (Windows and Linux) plus a full CLI

In simulation, with 2 long scripts (389 sentences) and 20 retakes sharing one GPU, a fair-share scheduler cut average retake latency from 102.6 s to 0.83 s (123×). Script sentences finished 7.6% later. Total throughput is identical, because the GPU does the same work in a different order.

Biggest lesson: the "best" policy depends on who's waiting. Shortest-job-first had the best overall average, and still left retakes waiting up to two minutes. Averages hide it.

Code, app, and full results in the first comment.

#GPU #MachineLearning #Python #SystemsEngineering #MLOps

---

First comment:

Code and downloads: https://github.com/jacobzychowicz/Optimizing-GPU-Inference-Scheduling
Full results and how to reproduce them: https://github.com/jacobzychowicz/Optimizing-GPU-Inference-Scheduling/blob/main/docs/results.md

---

Variant after a live run (replace the "In simulation" paragraph):

On a [GPU] running XTTS-v2, with [N] script sentences and [N] retakes, a fair-share scheduler cut average retake latency from [x] s to [y] s ([z]×). Script sentences finished [p]% later, with the same throughput. The calibrated simulator predicted the live mean latency within [e]%.

---

Notes:
- Keep the real numbers; don't round up. Say "in simulation" until there is a live run.
- If RecapVox should stay unnamed, use "a voice-cloning app I'm building".
- Links go in the first comment: LinkedIn shows posts without links in the body to more people.
- Only post the "Desktop app" bullet once a GitHub Release exists, so the download links work.
