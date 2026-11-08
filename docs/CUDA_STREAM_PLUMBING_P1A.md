# CUDA stream plumbing: P1a audit and implementation contract

**Status:** design/audit plus experimental implementation in PR #5; not merged. Initial target-GPU smoke tests passed, but cross-layer error-path ownership remains a blocking defect.
**Parent:** `feature/cuda-pinned-host-slabs` (PR #1).
**Scope:** one explicit CUDA stream while preserving the current synchronous C ABI.
**Non-goal:** overlap, prefetch, concurrent expert uploads, or performance claims.

## Executive finding

The CUDA backend currently relies on the legacy default stream and synchronous
`cudaMemcpy` calls. Kernels are launched without an explicit stream. The public
C entry points return only after their device-to-host result is available, so
callers currently assume all host input buffers may be reused immediately after
return.

A safe P1a must preserve that contract. Merely creating a stream is incorrect:
work submitted to a non-blocking stream is not ordered with legacy default-stream
work. Every transfer and kernel in a call must use the same explicit stream,
and the API must drain it before returning. P1a therefore establishes explicit
ordering, not asynchronous overlap.

## Complete call-site inventory in `picchio_cuda.cu`

| Area | Current operations | P1a requirement |
|---|---|---|
| Initialization | device selection, memory query | Create one non-blocking compute stream; fail initialization cleanly if creation fails |
| `pgpu_matmul_i8` | cached weight/scales H2D, input H2D, lm-head kernel, output D2H | Queue all work on the stream; synchronize before return |
| `pgpu_router_upload` | router weights/bias H2D | Queue copies on the stream; synchronize before return because model-owned host pointers are borrowed |
| `pgpu_router_scores` | input H2D, router kernel, scores D2H | Same-stream ordering; synchronize before return |
| `pgpu_matmul_i4gs` | input/weights/scales H2D, expert kernel, output D2H | Same-stream ordering; synchronize before return |
| `pgpu_matmul_i3gs` | input/weights/scales H2D, expert kernel, output D2H | Same-stream ordering; synchronize before return |
| `get_expert` | up to six expert H2D uploads; VRAM LRU eviction | Uploads use the compute stream; any failure drains queued work before returning or freeing partial allocations |
| `pgpu_moe_layer` | input H2D, zero/gate-up/SwiGLU/down/accumulate kernels, output D2H | All kernels and copies use the same stream; synchronize before success or CPU fallback |
| Host registration | `cudaHostRegister` / `cudaHostUnregister` | Unregister must not race outstanding DMA; retain the existing bounded registration accounting |
| Shutdown | unregister pinned slabs, free caches/scratch | Drain stream first, then unregister/free, then destroy stream |
| Standalone tests | two `cudaDeviceSynchronize` calls in timing harness | Keep valid; API calls themselves remain synchronous |

The source contains no other runtime kernel launch or `cudaMemcpy` call sites
outside this inventory. The CUDA runtime also allocates/frees device memory in
`ensure`, cache insertion/eviction, and shutdown; those paths must not free a
buffer while queued work can still reference it.

## Required ordering and lifetime invariants

1. Exactly one backend-owned compute stream exists between successful init and
   shutdown. P1a does not add worker threads or additional streams.
2. Every kernel launch explicitly names that stream.
3. Every H2D and D2H transfer is enqueued on that same stream. A helper should
   report enqueue failures and drain already queued work before the caller
   releases any borrowed host pointer.
4. Every public operation that borrows host memory must synchronize before returning,
   including error/fallback returns after any transfer has been enqueued. PR #5 does
   not yet satisfy this on all error paths.
5. Expert host slabs cannot be unregistered, recycled, or freed while a queued
   transfer might still read them. P1a's synchronous API boundary is the lease:
   it must be upheld on every exit path. The current void unregister ABI does not
   let the C owner know whether unregister/drain succeeded.
6. LRU eviction cannot free a device expert still referenced by queued kernels.
   Drain the stream before freeing an eviction victim unless a later event/lease
   mechanism proves a narrower safe boundary.
7. Shutdown drains outstanding work before unregistering host slabs and freeing
   device allocations. Stream destruction happens last.
8. CPU fallback remains valid after any CUDA error. No partial GPU result may be
   consumed as a successful result.
9. The CPU-only build, CPU self-test, and original CUDA library remain untouched.

## Error-path review points

The most dangerous paths are not successful returns but partial submission:

- A sequence of expert H2D copies can fail after earlier copies were enqueued.
- A later expert allocation can fail after earlier experts' kernels were queued.
- A kernel launch error can occur after input copies or earlier kernels entered
  the stream.
- An output D2H enqueue or stream synchronization can fail after computation.
- A pinned host slab can be recycled immediately after `pgpu_moe_layer` returns
  failure and the C engine starts CPU fallback.

Each such path must drain queued work before returning control to the C engine.
A single success-only synchronization is insufficient.

## Validation gates before any performance work

### Hosted CI (compile only; no GPU claim)

- Compile the runtime shared library for `sm_86`.
- Compile `PGPU_PINNED_TEST`.
- Compile `PGPU_ROUTER_TEST`.
- Run the existing CPU build/self-test and `git diff --check`.
- State explicitly that compilation does not execute CUDA or validate runtime
  behavior.

### RTX 3050 target

- Build to a new temporary library path; never overwrite the known-good library.
- Run pinned-host and router numeric self-tests.
- Run CPU-vs-GPU correctness checks for INT4 and INT3, including failure/fallback
  cases where reproducible.
- Run repeated short inference with the existing CPU baseline unchanged.
- Verify clean shutdown, zero leaked pinned registrations, and no OOM/segfault.
- Compare output token IDs against the current synchronous CUDA implementation
  with deterministic sampling.

Only after these gates pass should P1b introduce a separate upload stream,
completion events, per-expert UPLOADING state, host DMA leases, and eviction
protection. P1a must not claim speedup: the required stream synchronization at
each API boundary intentionally preserves the current serialization contract.


## Implementation audit addendum — 2026-11-08

PR #5 implements the explicit compute stream, asynchronous copies, and explicit
kernel stream arguments. Hosted compile/static checks passed. On the MSI RTX
3050, `PGPU_PINNED_TEST` passed and `PGPU_ROUTER_TEST` passed twice (CPU/GPU
top-1 matched; max absolute error 7.629395e-06). These are limited smoke tests,
not full runtime or failure-path validation.

A source-level cross-layer audit then found that the error-path invariants above
are not yet met:

- `pgpu_host_unregister()` discards a failed `stream_wait()`.
- The function returns `void`; `picchio.c:free_slot_buffers()` unconditionally
  clears the pinned state and calls `flat_aligned_free()` after the callback.
  Returning early in CUDA alone would therefore not protect the host slab.
- `pgpu_shutdown()` discards the drain result and proceeds to unregister host
  memory, free device caches/scratch, and destroy the stream.
- `stream_copy()` and `CU_TRY` ignore recovery-wait results. Partial upload
  cleanup in lm-head/router/expert paths can free allocations without a
  confirmed safe completion boundary.
- `ensure()` releases old scratch allocations during resize; its safety depends
  on the backend's error state being known-good.
- `get_expert()` checks the wait before normal LRU eviction, but ignores the
  wait result before freeing a partially uploaded expert.

The public C API currently has no explicit poisoned-backend state or teardown
status channel. The host owner must participate in the policy: a backend-only
early return is insufficient because the C caller will still free/recycle the
slab.

**Required remediation:** define a backend poison/fail-stop policy; propagate
unregister and teardown outcomes across the C ABI; make the host owner retain
or deliberately fail-stop rather than free a slab after failed unregister;
apply the same policy to partial device uploads, scratch resizing, CPU fallback,
and shutdown; and add deterministic stream-wait fault-injection tests.

Until that work and its tests exist, PR #5 remains Draft and must not be
approved or merged. The two successful target-GPU smoke tests do not close this
gate.
