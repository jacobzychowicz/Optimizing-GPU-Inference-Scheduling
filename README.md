# Independent Research Project: Optimizing GPU Inference Scheduling

**Status:** Exploring / Researching Feasibility  
**Research Area:** Computer Systems, AI Infrastructure, GPU Computing, Performance Optimization  
**Proposed Topic:** Workload-Aware Scheduling for Latency and Fairness in Single-GPU Text-to-Speech Inference

## 1. My Background

I am a second-year Computer Science (Co-op) student at McMaster University, interested in software engineering, cloud computing, AI infrastructure, and systems research.

My relevant experience includes:

- Working as a research assistant at McMaster University's NSDA Lab, developing Python data pipelines, RAG infrastructure, and optimizing retrieval performance.
- Building RecapVox, a full-stack AI narration SaaS using React, FastAPI, PyTorch/CUDA, and Qwen3-TTS running on a self-hosted GPU.
- Implementing priority-based GPU inference job scheduling for different subscription tiers in RecapVox.
- Experience with Python, JavaScript, Linux, backend development, and AI systems.

I am considering independently conducting research into GPU inference scheduling, potentially with feedback from a professor.

## 2. Why I Want to Do This

My main goals are:

1. Gain a deeper understanding of GPU computing, AI inference, operating systems, and scheduling algorithms.
2. Learn how to conduct legitimate computer science research, including literature reviews, experimental design, benchmarking, and technical writing.
3. Build a technically impressive, reproducible open-source project.
4. Potentially write a research paper if the findings are rigorous and sufficiently original.
5. Strengthen my background for software engineering, cloud infrastructure, AI engineering, and future research opportunities.

**My priority is learning and producing high-quality engineering work, not publishing a paper simply for my resume.**

I want to avoid producing a weak, misleading, or unoriginal publication.

## 3. Proposed Research Question

**Can a lightweight GPU inference scheduling algorithm that considers predicted job duration, request priority, and queue waiting time reduce latency while maintaining acceptable throughput and fairness?**

More specifically, I want to investigate how scheduling policies affect the performance of single-GPU text-to-speech inference services under different workloads.

The project would initially focus on Qwen3-TTS because I already have experience deploying and using this model.

## 4. The Problem

When multiple users request AI inference from the same GPU, the system needs to decide which requests to process first.

For example:

- A user requests a long narration.
- Another requests a short sentence regeneration.
- A high-priority subscriber submits a request.
- Several new requests arrive while the GPU is busy.

A simple first-come, first-served queue can cause short requests to wait behind long-running requests.

Strict priority scheduling may improve responsiveness for premium users but cause low-priority requests to experience excessive delays.

Shortest-job-first scheduling may reduce average latency but requires estimates of job execution time and can disadvantage long requests.

I want to investigate these trade-offs experimentally.

The initial scope is **non-preemptive, request-level scheduling on a single GPU**. This means the scheduler chooses which queued request executes next, rather than interrupting active model execution.

This is different from advanced GPU-internal scheduling or token-level scheduling.

## 5. Proposed Scheduling Algorithms

I would implement and compare:

**Baseline 1 — FIFO / First Come, First Served**

Process requests in arrival order.

**Baseline 2 — Shortest Predicted Job First**

Prioritize requests based on estimated GPU execution time.

**Baseline 3 — Priority Scheduling**

Prioritize requests according to assigned service tiers.

**Baseline 4 — Aging-Based Scheduling**

Gradually increase a request's priority as its waiting time increases to reduce starvation.

**Candidate Algorithm — Workload-Aware Hybrid Scheduling**

Experiment with combining:

- Predicted inference duration
- Request priority
- Queue waiting time
- Aging mechanisms

The candidate algorithm is not assumed to be novel or superior. Its usefulness must be established through comparisons and a literature review.

## 6. Proposed Research Methodology

### Phase 1: Literature Review

Study existing research on inference scheduling, including:

- Orca: A Distributed Serving System for Transformer-Based Generative Models
- Fairness in Serving Large Language Models
- Efficient LLM Scheduling by Learning to Rank
- Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions
- Qwen3-TTS serving and batching research
- vLLM and NVIDIA Triton inference scheduling

Identify which scheduling methods have already been studied, what limitations remain, and whether my proposed experiments offer a meaningful contribution.

### Phase 2: GPU Inference Profiling

Benchmark Qwen3-TTS on available hardware.

Measure:

- Inference execution time
- Input text length
- Generated audio duration
- GPU utilization
- GPU memory consumption
- Inference throughput
- Runtime variation across requests

Investigate whether job execution time can be estimated using information available before inference begins.

### Phase 3: Scheduling Simulation

Build a Python-based discrete-event scheduling simulator.

Use simulated job arrival times and execution durations to evaluate different scheduling policies before running expensive GPU experiments.

Verify scheduling correctness, queueing behavior, priority handling, and starvation scenarios.

Simulation results will not be presented as equivalent to real GPU benchmarks.

### Phase 4: Real GPU Benchmarking

Implement a real inference worker connected to configurable scheduling policies.

Generate repeatable request workloads containing combinations of short, medium, and long text-to-speech jobs.

Benchmark each scheduling algorithm under identical workload conditions.

### Phase 5: Experiments

Investigate:

1. **Variable request rates:** How does each scheduler perform under low, medium, high, and overloaded conditions?
2. **Mixed job lengths:** How do short and long requests affect queueing latency?
3. **Priority classes:** How effectively are different request priorities handled?
4. **Prediction accuracy:** What happens when estimated job runtimes are incorrect?
5. **Starvation and fairness:** Do certain policies systematically disadvantage long or low-priority requests?
6. **Optional batching experiments:** Can batching improve throughput without introducing unacceptable latency?

### Phase 6: Analysis

Compare results using:

- Average and median latency
- P95 and P99 latency
- Queue waiting time
- Requests completed per second
- Generated audio throughput
- GPU utilization
- Per-priority-class performance
- Slowdown ratios
- Failure rates and out-of-memory events

Use repeated trials, controlled configurations, and appropriate uncertainty estimates.

Perform ablation experiments to evaluate which features of the proposed scheduler actually improve performance.

## 7. Research Hypothesis

A scheduling policy combining predicted job duration, request priority, and waiting-time aging may improve latency for mixed-length text-to-speech workloads while reducing the starvation associated with strict shortest-job-first or priority scheduling.

This is a hypothesis to test, not a conclusion.

It is possible that existing schedulers perform equally well or better.

A negative result would still be useful if the experiments are rigorous and identify meaningful limitations or trade-offs.

## 8. Potential Research Contributions

Possible contributions include:

- A reproducible benchmark suite for single-GPU TTS inference scheduling.
- An empirical comparison of existing scheduling policies under different TTS workloads.
- Analysis of how runtime prediction errors affect scheduling decisions.
- Identification of trade-offs between latency, throughput, and fairness.
- A lightweight scheduling policy that outperforms meaningful baselines under specified conditions, if experiments support this.

Simply combining existing algorithms is not necessarily an original research contribution.

I need to determine the project's novelty by reviewing existing publications and evaluating results.

## 9. Proposed Technology Stack

**Language:** Python  
**Inference:** PyTorch, Qwen3-TTS  
**GPU:** NVIDIA CUDA-compatible GPU, subject to available hardware  
**Backend:** FastAPI, if useful  
**Scheduling:** Custom Python scheduling algorithms  
**Benchmarking:** Python, CUDA timing tools, NVIDIA GPU monitoring  
**Analysis:** Pandas, NumPy, Matplotlib  
**Testing:** pytest  
**Version Control:** Git and GitHub  
**Writing:** LaTeX / Overleaf

Optional advanced tools include vLLM-Omni, NVIDIA Triton, Docker, and GPU profiling utilities.

## 10. Expected Deliverables

The project should ideally produce:

1. A public GitHub repository containing scheduling implementations.
2. A reproducible benchmarking framework.
3. Experimental datasets or generated workload configurations.
4. Performance measurements and comparison graphs.
5. Documentation explaining methodology, setup, and limitations.
6. A written technical report describing the research and findings.

**Optional final deliverable:** A research paper submitted to a legitimate peer-reviewed workshop, conference, or journal if the work demonstrates sufficient novelty and quality.

Publication is not required for the project to be successful.

## 11. Potential Challenges and Limitations

Important concerns include:

- GPU hardware and VRAM limitations.
- Whether Qwen3-TTS supports meaningful batching with the chosen inference backend.
- Prediction errors when estimating inference duration.
- Distinguishing queueing improvements from actual GPU throughput improvements.
- Whether the proposed method is genuinely novel.
- Generalizability from a single model and GPU.
- Experimental reproducibility.
- Avoiding misleading comparisons and exaggerated performance claims.
- Time constraints alongside university coursework and co-op applications.

The project must not use private customer data, proprietary research data, or voice samples without appropriate permission.

## 12. Publication Philosophy

I do not want to publish a paper merely to claim that I am a published researcher.

A weak publication that exaggerates novelty, uses poor methodology, or appears in a questionable journal could be worse than publishing nothing.

My preferred approach is:

**First:** Learn the field and reproduce meaningful baselines.

**Second:** Build and validate a legitimate experimental study.

**Third:** Seek feedback from an experienced researcher or professor.

**Fourth:** Determine whether the work qualifies as an original research contribution.

**Fifth:** Publish appropriately if justified.

If the project does not produce publishable research, I can still release a well-documented technical report and open-source benchmarking system.

## 13. How This Relates to My Career

I am preparing for Summer 2027 software engineering and cloud computing co-op positions, potentially including CIBC.

This project could demonstrate skills in:

- Systems programming
- Performance engineering
- AI infrastructure
- Queueing and scheduling algorithms
- GPU resource management
- Backend architecture
- Experimental benchmarking
- Technical communication
- Independent research

However, research should not interfere with applying to co-op positions, interview preparation, or strengthening my existing software projects.

## 14. Questions I Need to Answer Before Committing

- What scheduling research already exists for TTS inference specifically?
- Is my proposed research question sufficiently distinct from existing work?
- Which GPU hardware and inference backend can I realistically use?
- Can I collect enough reliable benchmark data?
- Will my experiments investigate real system behavior rather than only simulations?
- Can I identify a narrow research question with clear, measurable outcomes?
- Can I maintain reproducible methodology?
- Can I get feedback from a professor or experienced researcher?
- Would I still consider the project worthwhile if it never resulted in a publication?
- Is the required time commitment reasonable alongside school and co-op applications?

## 15. Project Success Criteria

**Minimum success:** I understand inference scheduling, implement several scheduling algorithms, and build a working benchmark.

**Strong success:** I produce reproducible real-GPU measurements, compare algorithms rigorously, and release a well-documented technical report.

**Excellent success:** I identify and validate a meaningful original contribution, receive expert feedback, and develop a paper suitable for submission to a credible research venue.

**The main goal is to learn to conduct good research, not to obtain a publication at any cost.**

---

## Instructions for AI Research Assistants

I am a second-year Computer Science student evaluating whether this is a feasible and worthwhile independent research project.

Please help me critically investigate this proposal rather than automatically encouraging me.

When assisting me:

1. Identify existing academic papers that overlap with my proposal.
2. Explain what has already been solved and what remains open.
3. Critique my research question for originality, feasibility, and appropriate scope.
4. Distinguish established research findings from speculation or proposed ideas.
5. Prefer primary academic sources, official implementations, and reproducible benchmarks.
6. Do not invent citations, experimental results, or claims of novelty.
7. Suggest manageable experiments suitable for a second-year student with limited hardware.
8. Identify flaws, confounding variables, or unrealistic assumptions in my experimental design.
9. Explain concepts clearly without oversimplifying their technical significance.
10. Be willing to conclude that the project is not novel enough to publish.
11. Help me prioritize learning, technical rigor, and reproducibility over resume value.
12. When suggesting improvements, explain how they differ from existing research.

**My immediate goal is to evaluate the project's feasibility and research gap before committing to a full implementation.**
