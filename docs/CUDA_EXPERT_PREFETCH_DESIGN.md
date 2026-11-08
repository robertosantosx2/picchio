# CUDA expert prefetch: implementation design

**Status:** research/design only. No CUDA asynchronous transfer or performance gain is claimed.
**Target:** Picchio MoE, including the MSI Thin 15 B12UC / RTX 3050 Laptop GPU (sm_86, 4 GiB).
**Baseline branch reviewed:** `feature/cuda-pinned-host-slabs`, PR #1 head `c52dfcf1107063e40bae75b4a3c2b11d164b1764`.
**Purpose:** specify the next optimization without duplicating the existing host/disk prefetch or weakening cache correctness.

## 1. Audit finding: part of P1 already exists

Picchio already has a CPU-side predictive prefetch path:

- `predict_topk_into()` computes a proxy top-k for a later layer.
- `prefetch_issue()` reserves host expert-cache slots and schedules reads on the prefetch worker.
- `prefetch_run_job()` loads the selected experts from the SafeTensors source into those reserved slots.
- `slot_loading` / `slot_set_loading` publish slot readiness with acquire/release ordering.
- `cache_load_batch()` and `cache_load_batch_async()` avoid evicting slots while a host prefetch is in flight.
- `g_gpu_prefetch` uses this prediction path for GPU-offload runs.

This is **host/disk prefetch**, not GPU upload prefetch. It can make expert bytes available in RAM before the layer needs them, but it does not upload those bytes into the CUDA expert LRU on a separate stream.

The current CUDA upload path in `get_expert()` uses `cudaMalloc` and synchronous `cudaMemcpy` calls. Kernels are launched on the default stream, and `pgpu_moe_layer()` ends with a blocking device-to-host copy. There is no dedicated upload stream, no per-expert upload event, and no pending-upload state.

Pinned host registration alone does not create overlap. The current PR #1 feature calls `cudaHostRegister`, but transfers remain synchronous. It must not be described as asynchronous transfer or as a measured speedup.

## 2. Intended pipeline

The desired pipeline is a two-stage pipeline with separate ownership:

1. **Host stage (existing):** predict future experts and read their packed weights from disk into reusable host slabs.
2. **Device stage (new):** when the host slabs are ready and registered, enqueue H2D copies on a dedicated CUDA upload stream while the current layer's expert kernels execute on the compute stream.
3. **Join:** when the next layer needs a prefetched expert, make the compute stream wait on that expert's upload event. Do not globally synchronize the device.
4. **Fallback:** if prediction misses, host memory is pageable, upload allocation fails, or an event cannot be recorded, use the existing synchronous `get_expert()` path.

The first implementation should target only a small bounded number of experts (initially at most the model's top-k for one future layer). No speculative expert may affect routing, weights, selected IDs, or numerical output.

## 3. Required per-expert state machine

Each device-cache entry needs an explicit state:

| State | Meaning | May be consumed? | May be evicted? |
|---|---|---:|---:|
| ABSENT | No device allocation | No | Yes |
| ALLOCATING | Device buffers are being allocated | No | No |
| UPLOADING | H2D copies are queued; event not complete | No, unless compute stream waits on its event | No |
| READY | Upload event completed successfully | Yes | Yes, subject to active-use lease |
| FAILED | Prefetch failed; no valid device copy | No | Yes |

Transitions:

- `ABSENT -> ALLOCATING -> UPLOADING -> READY` on successful prefetch.
- Any allocation/copy/event failure transitions to `FAILED`, releases partial device allocations, and leaves the host copy intact.
- A normal demand miss may synchronously create an entry and transition directly to `READY`.
- An entry in `UPLOADING` must never be returned as ready without an event wait.
- An entry in `UPLOADING` must never be selected by LRU eviction.
- An entry actively referenced by a compute submission must not be freed until that submission has completed.

A simple boolean `resident` flag is insufficient once uploads become asynchronous.

## 4. CUDA streams and events

Use two explicit streams:

- `compute_stream`: all MoE kernels and their dependent operations.
- `upload_stream`: H2D copies for future-layer experts.

For each expert upload:

1. Allocate all device buffers before publishing the entry.
2. Enqueue copies on `upload_stream` using `cudaMemcpyAsync`.
3. Record a per-entry `cudaEvent_t ready_event` after the final copy.
4. Mark the entry UPLOADING.
5. Before use, enqueue `cudaStreamWaitEvent(compute_stream, ready_event, 0)`; then the kernels can consume it without a host-side global wait.
6. Recycle the event and buffers only after completion and after all compute leases have ended.

Do not use `cudaDeviceSynchronize()` in the token loop. A device-wide synchronization would erase the intended overlap.

The compute path must launch kernels on the explicit compute stream. Mixing implicit default-stream operations with explicit streams risks hidden synchronization or incorrect ordering depending on default-stream semantics. Every kernel launch and copy involved in this pipeline must use the intended stream explicitly.

## 5. Host-memory lifetime and pinning

The host slab is the source of an asynchronous DMA operation. It must remain allocated and unchanged until the H2D event completes.

Required host-slot ownership:

- Add a small DMA lease/reference count to the host expert slot, or an equivalent explicit pin/unpin API.
- Eviction/recycling must skip a slot while its DMA lease is non-zero.
- Release the lease only after the upload event has completed, not immediately after enqueueing `cudaMemcpyAsync`.
- On shutdown, stop new prefetch submissions, drain upload events, release host leases, then unregister pinned slabs and free allocations.

Only call `cudaMemcpyAsync` from registered/pinned slabs in the first version. If the slab is not pinned, retain the existing synchronous upload path. This avoids relying on runtime staging of pageable memory and gives the experiment a clean A/B boundary.

Pinned registration should remain bounded and optional. Do not increase the 128 MiB cap as part of the first async-transfer patch. Keep the cap configurable only in a later, separately measured experiment.

## 6. Device-cache admission and eviction

The current `g_ecache` is an LRU map and `get_expert()` uploads synchronously. It is not currently thread-safe and has no in-flight ownership model.

For the first implementation:

- Keep all cache-map mutation on the inference/main thread.
- The upload stream may execute DMA asynchronously, but do not introduce a second host thread that mutates `g_ecache`.
- Reserve capacity before starting an upload.
- Count UPLOADING allocations against the VRAM budget immediately.
- Never evict UPLOADING entries.
- Never evict entries with active compute leases.
- Keep current-layer experts protected while the layer's kernels are being enqueued.
- If the predicted set cannot fit without evicting protected/current entries, skip prefetch rather than overcommit VRAM.
- On any demand-time cache miss, use the current synchronous upload route.

This is intentionally conservative. A background CUDA worker thread, lock-free map, or doorbell protocol is out of scope.

## 7. Prediction and admission policy

The existing predictor is a proxy: it scores the next layer using an earlier activation. Prediction misses are expected.

Initial policy:

- Prefetch only the predicted top-k for one future layer.
- Do not prefetch more than one layer ahead.
- Do not prefetch an expert already in the device cache or already pending.
- Do not evict a READY entry solely to make room for a speculative upload if the demand cache is under pressure.
- Track predicted hits, prediction misses, upload bytes, upload duration, bytes skipped due to budget, and demand-time waits.
- Add an environment switch (default OFF) for the new GPU-upload prefetch independently of the existing host-prefetch switch.
- Keep a kill switch that disables uploads while leaving the current CUDA path untouched.

The first benchmark should use the same prompt, context, output-token count, model, cache warmness and thread settings across paired OFF/ON runs.

## 8. Instrumentation required before performance claims

Per run, report:

- predicted experts, already-resident skips, and duplicate/pending skips;
- host slots pinned vs pageable;
- bytes and expert count submitted to the upload stream;
- upload enqueue time and event-completion time;
- number of demand-time event waits and their total/max duration;
- upload failures by CUDA API and allocation stage;
- device-cache bytes in READY and UPLOADING states separately;
- LRU evictions by reason (normal demand vs speculative admission);
- host DMA leases acquired/released and any shutdown drain time;
- existing disk reads, host-cache hit rate, VRAM-cache hit rate, RSS, swap, GPU power and temperature.

Do not conflate enqueue latency with transfer completion. Report both.

## 9. Correctness and failure tests

Before running model benchmarks, add deterministic tests for:

1. Event ordering: compute cannot read a test buffer before its upload event.
2. Upload lifetime: the host slab is not recycled while a DMA lease is active.
3. LRU safety: UPLOADING and leased READY entries are not evicted.
4. Budget: speculative admission never exceeds the configured VRAM budget.
5. Duplicate requests: one pending upload per (layer, expert).
6. Failure rollback: partial allocations are freed and host fallback remains valid.
7. Prediction miss: normal demand upload returns the same result as the prefetch-disabled path.
8. Shutdown: pending uploads complete or are safely drained before host unregister/free.
9. INT4 and INT3 packed formats: byte counts and scales match the existing format calculations.
10. Numerical parity: output remains within the existing exact/tolerance contract for the same expert ordering.

A synthetic CUDA test should use small buffers and should not require downloading a model checkpoint.

## 10. Rollout plan

### P0 — baseline and invariants
- Keep PR #1 pinned registration isolated and draft.
- Add/verify request-size counters on the target GPU.
- Record a repeatable no-prefetch CUDA baseline.

### P1a — stream plumbing only
- Add explicit compute and upload streams.
- Route current kernels/copies through the compute stream without changing behavior.
- Add teardown and stream-error tests.
- Verify output parity and performance neutrality before enabling any async prefetch.

### P1b — one-expert asynchronous upload
- Add per-entry state/event and host DMA lease.
- Implement one predicted expert upload on the upload stream.
- Keep feature disabled by default.
- Test all failure paths and exact ownership rules.

### P1c — bounded top-k prefetch
- Expand to top-k only after P1b is correct.
- Add admission accounting and metrics.
- Run paired OFF/ON measurements, at least three repetitions per configuration.

### P2 — adaptive hybrid CPU/GPU
- Consider only after P1 measurements establish whether host reads, PCIe transfer, or GPU compute is limiting.
- Do not assume CPU compute is faster than PCIe or that expert weights are RAM-resident.
- Keep disk misses out of any CPU-vs-GPU split heuristic until measured separately.

## 11. Hardware-specific constraints and non-goals

The target RTX 3050 Laptop GPU has 4 GiB VRAM. The host has approximately 14 GiB RAM and 32 GiB swap. The existing workload has shown significant host-cache misses and disk reads. Therefore:

- Avoid large pinned-memory increases.
- Avoid prefetching multiple layers ahead.
- Avoid duplicating the existing host cache or loader.
- Avoid allocating all predicted experts if the VRAM budget cannot safely hold them.
- Do not assume the 4090 results from other MoE runtimes transfer to sm_86.
- Do not change the CPU baseline or overwrite the known-good CUDA shared library during target validation.

## 12. Acceptance criteria

The feature is eligible to become default only if all of the following hold:

- Synthetic CUDA tests pass on the supported target architecture.
- Numerical parity is demonstrated on INT4 and INT3.
- No host-slot use-after-free, device use-after-free, or cache overcommit is observed.
- At least three paired repetitions show a consistent improvement in median decode throughput or latency.
- Disk reads, cache hit rates, RSS and swap are reported alongside tok/s.
- The feature can be disabled independently and cleanly falls back to the current synchronous path.

Until then, this is an implementation design, not a performance claim.
