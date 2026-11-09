# Picchio MoE — Global Engineering Improvement Proposal

**Status:** proposal for maintainer discussion; not an implementation claim.  
**Scope:** a staged, evidence-led roadmap for improving Picchio's ability to run large sparse MoE models on machines with constrained RAM and VRAM (o no GPU), while preserving correctness, portability, and predictable failure behaviour.

## 1. Goals and guiding principles

Picchio should make the best use of available CPU memory, GPU memory, storage bandwidth, and compute without admitting workloads on optimistic assumptions or rejecting workloads solely because a conservative estimate exceeds a nominal budget.

The work should follow five principles:

1. **Measure before tuning.** Distinguish estimates, reservations, committed allocations, and observed peaks.
2. **Correctness before throughput.** Every optimisation must preserve numerical and behavioural correctness within an explicit tolerance.
3. **Bound every resource.** RAM, VRAM, pinned memory, scratch space, queues, and caches need explicit limits and observable accounting.
4. **Make asynchronous ownership explicit.** No buffer may be freed, reused, or sent through a fallback path while device work may still reference it.
5. **Keep optional complexity optional.** Hardware-specific paths must have a safe baseline and be independently testable.

## 2. Memory accounting and admission control

### Problem to address

A model's estimated or reserved memory is not necessarily its eventual runtime consumption. Lazy allocation, sparse expert activation, cache eviction, quantisation, and workload shape can make actual usage materially lower than a worst-case estimate. Conversely, under-accounting can cause OOM failures after admission.

### Proposed work

Introduce a unified memory ledger with separate categories:

- model weights: resident CPU, resident GPU, mapped, and on-disk;
- expert storage/cache: host cache, device cache, in-flight loads, and evictable entries;
- KV cache: allocated capacity versus currently occupied tokens;
- scratch/workspace: reserved capacity versus high-water mark;
- pinned host slabs and staging buffers;
- CUDA allocations, allocator/runtime overhead, and safety reserve.

For each category report:

- estimated requirement;
- reserved capacity;
- committed/allocated bytes;
- current use where measurable;
- peak observed use;
- reclaimable/evictable bytes;
- allocation failures and their cause.

Admission control should use a documented policy rather than one undifferentiated total. Where safe, allow a workload whose measured/lazy allocation path fits even if a conservative upper bound does not. Retain hard ceilings, safety margins, and rollback/fail-stop behaviour for genuinely unsafe conditions. Never treat swap as equivalent to VRAM or fast host RAM.

### Acceptance evidence

- A reproducible report compares predicted, reserved, committed, and peak memory for the same workload.
- Tests cover just-below, at-limit, and over-limit cases.
- An admission decision explains which budget blocked it and by how much.
- No OOM or resource leak is hidden by silently weakening limits.

## 3. Expert I/O, cache, and storage pipeline

### Proposed work

- Define a single expert-load lifecycle: request, queue, read, decode/dequantise if required, transfer, publish, and evict.
- Measure storage read latency, transfer latency, queue wait, cache hit rate, and p50/p95/p99 load latency separately.
- Track bytes read from disk independently from bytes copied to host or device.
- Evaluate aligned flat expert stores and sequential/positional I/O against the current file layout.
- If prefetch is introduced, insert completed prefetched experts directly into the normal cache/LRU rather than reading or materialising them a second time.
- Preserve a safetensors or existing-format fallback until alternative storage paths have equivalent correctness and recovery coverage.
- Bound queue depth and in-flight bytes; cancellation and shutdown must drain or safely invalidate outstanding work.

### Acceptance evidence

- Cold-cache and warm-cache benchmarks are reported separately.
- Results include throughput, tail latency, cache hit/miss/eviction counts, and actual disk bytes read.
- A stress test proves bounded memory under sustained expert churn.
- The baseline path remains available when an optional storage backend is disabled.

## 4. GPU execution, streams, and failure semantics

### Proposed work

- Give each CUDA stream a clear owner and document which operations are ordered on it.
- Keep public C APIs synchronous until buffer-lifetime and completion guarantees justify exposing asynchronous behaviour.
- Ensure H2D copies, kernels, and D2H copies use the intended stream consistently.
- Separate recoverable errors (where completion and ownership are known) from uncertain asynchronous failures (where the device may still reference host/device buffers).
- Do not free, unregister, reuse, or CPU-fallback from buffers after an uncertain completion failure. Prefer a deliberate fail-stop over use-after-free or silent corruption.
- Test router, dense projections, expert transfers, and shutdown independently.
- Make GPU tests hardware-gated in CI: compile and static checks can run everywhere; runtime tests should run only on suitable CUDA runners or documented local hardware.

### Acceptance evidence

- Fault injection verifies that a failed stream wait cannot return control to a caller that then reuses or frees an in-flight buffer.
- Normal shutdown drains work and releases registered resources.
- Router and dense outputs are compared with CPU references.
- CI clearly distinguishes compile-only checks from actual GPU execution.

## 5. Quantisation and kernel efficiency

### Proposed work

- Profile the existing INT3/INT4/INT8 paths and dense projections before adding new kernels.
- Prioritise kernels by end-to-end contribution, not isolated microbenchmark speed.
- Evaluate SIMD/CPU and CUDA implementations behind narrow interfaces.
- Compare dequantisation-on-load, fused dequantisation/matmul, and resident representations where the memory/compute trade-off is measurable.
- Record format-specific numerical error, throughput, memory footprint, and supported tensor shapes.
- Keep unsupported or unvalidated formats out of automatic selection.

### Acceptance evidence

- Every kernel has deterministic small fixtures and CPU-reference comparisons.
- Report max absolute error and a scale-aware metric such as relative RMS error.
- Benchmarks include prefill and token-generation workloads, not only synthetic matrix multiplication.
- The selected path and fallback reason are observable.

## 6. Scheduling and overlap — later phase

Only after memory, I/O, and GPU ownership are reliable should Picchio consider more aggressive overlap:

- expert prefetch concurrent with current-layer compute;
- bounded transfer/compute pipelines;
- adaptive cache and prefetch policies;
- queue-depth tuning based on measured storage and device behaviour.

These changes must be incremental. First prove ordering and lifetime correctness; then measure whether overlap improves end-to-end token latency without increasing peak memory beyond configured limits.

## 7. External implementation research

TensorFold and tandem.cpp are useful sources of engineering patterns, not code to transplant wholesale. Review candidate techniques individually against:

- Picchio's streaming execution model;
- the target machine's limited RAM/VRAM;
- dependency and build complexity;
- model-format compatibility;
- numerical equivalence and failure behaviour;
- portability to CPU-only and different CUDA hardware.

Any adopted technique should have a short design note, a benchmark baseline, and a clear reason it fits Picchio better than the existing implementation.

## 8. Staged validation gates

| Gate | Required evidence | Exit condition |
|---|---|---|
| G0 — Reproducibility | Versioned model/config, command, hardware, logs, and baseline metrics | Another developer can reproduce the baseline |
| G1 — Correctness | CPU reference, deterministic fixtures, numerical tolerances | No unexplained output divergence |
| G2 — Resource safety | Memory ledger, hard limits, fault injection, clean shutdown | No unbounded growth, unsafe fallback, or ownership ambiguity |
| G3 — Performance | Cold/warm I/O, prefill, generation, peak RAM/VRAM | Improvement measured end-to-end without regressions hidden |
| G4 — Integration | Real model smoke test, documented supported configurations | Feature is safe to enable by default or remains opt-in |

Synthetic tests, compile checks, and real-model runs must be labelled separately. A successful build is not evidence of runtime correctness; a synthetic benchmark is not evidence of real-model performance.

## 9. Recommended implementation order

1. Establish reproducible baseline telemetry and the memory ledger.
2. Close correctness and resource-lifetime gaps in the existing CPU/GPU paths.
3. Improve expert I/O and cache accounting, then benchmark storage alternatives.
4. Optimise the highest-impact quantisation and dense kernels.
5. Add bounded prefetch/overlap only when earlier gates pass.
6. Evaluate external backend or format integrations only with measured benefit.

## 10. Decisions requested from maintainers

1. Is the staged roadmap aligned with Picchio's intended direction?
2. Which current bottleneck should be prioritised first: memory admission, expert I/O/cache, GPU correctness, or kernel throughput?
3. What hardware and model configurations should define the first supported validation matrix?
4. Should the memory ledger and validation gates become permanent project requirements?

## 11. Non-goals

This proposal does not request an immediate rewrite, an unconditional CUDA dependency, unbounded prefetching, weakening safety limits, or enabling experimental backends by default. It does not claim that the proposed work has already been implemented or that performance gains have been demonstrated.

The objective is a measurable sequence of small changes, each independently reviewable and reversible.
