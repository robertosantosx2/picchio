# CUDA pinned-host diagnostics and validation plan

**Status:** experimental; no performance benefit has been established.
**Target used for the initial diagnostic:** Ubuntu 26.04, MSI Thin 15 B12UC, RTX 3050 Laptop GPU (4 GiB VRAM), CUDA Toolkit 13.4.
**Source branch:** `feature/cuda-pinned-host-slabs`.

## What the feature does

Picchio can optionally register reusable, 4 KiB-aligned expert slabs with CUDA via
`cudaHostRegister`. Registration is bounded by a separate 128 MiB aggregate
budget. If registration is rejected, the allocation remains ordinary pageable
host memory and inference continues. The transfer path remains synchronous;
this change does not add CUDA streams or asynchronous copies.

The CPU baseline and the pre-existing CUDA shared library are outside this
experiment and must not be overwritten during validation.

## Observed target-hardware diagnostic

One isolated run used Qwen3-30B-A3B INT4, expert GPU offload enabled, expert
reuse enabled, context 512, four CPU threads, four I/O threads and eight output
tokens. The instrumented library was loaded from a separate temporary path.

| Metric | Observed result |
|---|---:|
| Generated tokens | 8 |
| Elapsed time | 17.5 s |
| Throughput | 0.46 tok/s |
| Host slabs registered | 50 |
| Registration failures | 1,277 |
| Peak registered host memory | 127.3 / 128 MiB |
| Cap rejections | 1,277 |
| Alignment failures | 0 |
| CUDA API failures | 0 |
| Bookkeeping failures | 0 |
| Expert VRAM cache | 1,485 hits / 1,292 misses (53.5%) |
| Host cache | 1,052 / 4,502 hits (23.4%) |
| Disk reads | 3,450 |

**Interpretation:** every failed registration in this run was rejected by the
aggregate 128 MiB cap. The CUDA registration call itself reported no failures.
This proves the cap was reached, not that raising it will improve inference.
The 0.46 tok/s result is a short diagnostic sample, not a stable benchmark and
not evidence of a speedup or slowdown caused by pinned memory.

## Preliminary sizing estimates (not measurements)

The current log does not yet include the distribution of slab sizes. The new
instrumentation reports request count, aggregate requested bytes, minimum and
maximum request size, and bytes rejected by the cap. Until those values are
collected, a defensible optimal cap cannot be calculated.

For planning only, the following are candidate configurations to compare:

| Candidate cap | Extra host-pinned budget vs current | Purpose |
|---|---:|---|
| 128 MiB | baseline | Existing safety ceiling |
| 192 MiB | +64 MiB | Conservative next point |
| 256 MiB | +128 MiB | Medium point |
| 384 MiB | +256 MiB | Upper experimental point; only if host memory headroom is confirmed |

These are test points, not recommendations or predicted performance outcomes.
On a machine with limited physical RAM and active swap, increasing pinned memory
can reduce memory available to the OS and may make the system less stable. Do
not raise the cap based only on the number of cap rejections.

### What can be estimated now

- **Registration coverage:** at least 127.3 MiB was pinned in the observed run.
  The total eligible bytes and the coverage percentage are unknown.
- **Potential transfer benefit:** unknown. The present path is synchronous and
  the measured run is disk/cache constrained; pinned memory alone does not
  remove disk reads or expert-cache misses.
- **Performance range:** no reliable numerical range can be estimated from one
  eight-token run. Treat any claimed tok/s improvement as unsubstantiated until
  repeated paired tests are complete.
- **Risk:** a larger cap consumes additional non-pageable host memory and can
  worsen memory pressure. This is particularly important on low-RAM systems.

## Validation plan

### GitHub-hosted checks (no local Ubuntu required)

The repository CI builds the CPU executable on Ubuntu and runs the synthetic
self-test. This validates the portable C build and CPU-side invariants, but does
not validate CUDA runtime behavior or the target GPU.

### Target GPU checks (must eventually run on RTX 3050)

1. Build the current CUDA source into a separate diagnostic shared library.
2. Run the pinned-host standalone self-test on a CUDA-capable device.
3. Run one eight-token diagnostic with the new request-size counters.
4. Use those request sizes to choose a small cap matrix; do not assume 192/256/
   384 MiB is safe until host-memory headroom is checked.
5. Run paired A/B trials at each accepted cap, with at least three repetitions
   per configuration and identical model, prompt, context, token count, thread
   count, cache warmness and environment.
6. Record median tok/s, total latency, disk reads, host cache hit rate, expert
   VRAM hit rate, peak RSS, swap activity, pinned bytes, GPU power and
   temperature.
7. Keep the original CUDA library and CPU baseline unchanged throughout.

### Decision criteria

- Reject a cap increase if it increases swap activity, causes OOM, destabilizes
  inference, or gives no repeatable throughput/latency benefit.
- Prefer the smallest cap that gives a statistically consistent improvement.
- If pinned registration does not improve results, retain the safe pageable
  fallback and consider disabling the feature by default.

## What GitHub CI cannot establish

A GitHub-hosted runner does not have access to the MSI's RTX 3050 or its local
model files. Hosted CI can validate source-level and CPU-only behavior. The
hardware-specific CUDA self-test and inference measurements remain pending
until a CUDA-capable runner or the target workstation is used.
