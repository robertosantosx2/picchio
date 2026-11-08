/* picchio_cuda.cu — optional CUDA backend for Picchio (prototype).
 *
 * Kernels:
 *   lm_head_i8     : one block per output row, coalesced INT8 reads, block reduce.
 *   expert_i4gs    : one thread per output row, INT4 group-scaled dequant matmul.
 *   expert_i3gs    : one thread per output row, INT3 gs64 bit-plane matmul.
 *
 * The math mirrors quant.h exactly (matmul_q8 / matmul_i4_gs) so the GPU top-1
 * token matches the CPU oracle; only the FP reduction order differs, so results
 * are NOT byte-identical (that contract belongs to the CPU path).
 *
 * Build the standalone numeric self-test:
 *   nvcc -O3 -arch=sm_75 -DPGPU_TEST -o pgpu_test.exe picchio_cuda.cu
 *   ./pgpu_test.exe
 *
 * Build the runtime DLL for the engine (integration, next step):
 *   nvcc -O3 -arch=sm_75 -shared -o picchio_cuda.dll picchio_cuda.cu
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <mutex>

#include "picchio_cuda.h"

/* ── error plumbing ─────────────────────────────────────────────────────── */

static int   g_ok = 0;          /* 1 once a device is initialized */
static char  g_name[256] = "";
static cudaStream_t g_compute_stream = nullptr;
#ifdef PGPU_STREAM_FAULT_TEST
static int g_test_fail_next_stream_wait = 0;
extern "C" void pgpu_test_fail_next_stream_wait(void) {
    g_test_fail_next_stream_wait = 1;
}
#endif

/* P1a keeps the public C API synchronous while making stream ordering explicit.
 * A failed drain is fail-stop: returning to the C owner could let it recycle a
 * host slab or free storage still referenced by queued CUDA work. Do not clear
 * the CUDA error and continue with fallback/cleanup when completion is unknown. */
static int stream_wait(const char *where) {
    if (!g_compute_stream) {
        if (g_ok) {
            fprintf(stderr, "[gpu] FATAL: %s has no live compute stream\n", where);
            fflush(stderr);
            abort();
        }
        return 0; /* no stream was created, or shutdown already completed */
    }
#ifdef PGPU_STREAM_FAULT_TEST
    if (g_test_fail_next_stream_wait) {
        g_test_fail_next_stream_wait = 0;
        fprintf(stderr,
                "[gpu] FATAL: injected stream-wait failure at %s; "
                "aborting to preserve host/device buffer ownership\n", where);
        fflush(stderr);
        abort();
    }
#endif
    cudaError_t e = cudaStreamSynchronize(g_compute_stream);
    if (e != cudaSuccess) {
        fprintf(stderr,
                "[gpu] FATAL: %s stream synchronize failed: %s; "
                "aborting to preserve host/device buffer ownership\n",
                where, cudaGetErrorString(e));
        fflush(stderr);
        abort();
    }
    return 0;
}
static cudaError_t stream_copy(void *dst, const void *src, size_t bytes,
                              cudaMemcpyKind kind, const char *where) {
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, g_compute_stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "[gpu] %s async copy enqueue failed: %s\n",
                where, cudaGetErrorString(e));
        (void)stream_wait(where);
        (void)cudaGetLastError();
        return e;
    }
    return cudaSuccess;
}

/* Bounded CUDA-pinned host slabs. Pageable memory remains the fallback. */
static std::mutex g_host_reg_mutex;
static std::unordered_map<void *, size_t> g_host_regs;
static size_t g_host_reg_bytes = 0, g_host_reg_peak_bytes = 0;
static uint64_t g_host_reg_ok = 0, g_host_reg_fail = 0;
static uint64_t g_host_reg_align_fail = 0;
static uint64_t g_host_reg_cap_fail = 0;
static uint64_t g_host_reg_cuda_fail = 0;
static uint64_t g_host_reg_bookkeeping_fail = 0;
static uint64_t g_host_reg_attempts = 0;
static uint64_t g_host_reg_attempt_bytes = 0;
static uint64_t g_host_reg_cap_fail_bytes = 0;
static size_t g_host_reg_min_attempt = (size_t)-1;
static size_t g_host_reg_max_attempt = 0;
static cudaError_t g_host_reg_last_cuda_error = cudaSuccess;
static const size_t G_HOST_REG_LIMIT = (size_t)128 * 1024 * 1024;

/* Overflow-safe aggregate pinned-memory budget check. */
static int host_reg_fits(size_t used, size_t bytes) {
    return used <= G_HOST_REG_LIMIT &&
           bytes <= G_HOST_REG_LIMIT - used;
}
static const uintptr_t G_HOST_REG_ALIGN_MASK = (uintptr_t)4095;

#define CU_TRY(call) do {                                            \
        cudaError_t _e = (call);                                     \
        if (_e != cudaSuccess) {                                     \
            fprintf(stderr, "[gpu] %s failed: %s\n", #call,          \
                    cudaGetErrorString(_e));                         \
            (void)stream_wait(#call);                                \
            return -1;                                               \
        }                                                            \
    } while (0)

/* ── device kernels ─────────────────────────────────────────────────────── */

/* lm_head: y[o] = scale[o] * dot(x, W_int8[o]). One block per row o. */
__global__ void lm_head_i8(const int8_t *__restrict__ W,
                           const float *__restrict__ scale,
                           const float *__restrict__ x,
                           float *__restrict__ y, int O, int I) {
    int o = blockIdx.x;
    if (o >= O) return;
    const int8_t *w = W + (long long)o * I;

    float partial = 0.f;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        partial += x[i] * (float)w[i];

    __shared__ float sm[256];
    sm[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = sm[0] * scale[o];
}

/* One resident router row per block. Only E floats return to the host; the
 * [E,D] matrix remains in VRAM across every token and prediction. */
__global__ void router_f32(const float *__restrict__ W,
                           const float *__restrict__ bias,
                           const float *__restrict__ x,
                           float *__restrict__ scores, int E, int D) {
    int e = blockIdx.x;
    if (e >= E) return;
    const float *w = W + (long long)e * D;
    float a = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        a += x[i] * w[i];
    __shared__ float sm[256];
    sm[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[e] = sm[0] + (bias ? bias[e] : 0.f);
}

/* expert INT4 group-scaled: y[o] = sum_g sc[o,g] * sum_{i in g} x[i]*(nib(i)-8).
 * One thread per output row (O is small: 5760 gate_up / 2880 down). */
__global__ void expert_i4gs(const uint8_t *__restrict__ Q,
                            const float *__restrict__ S,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int O, int I, int gs) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;
    const uint8_t *w = Q + (long long)o * rb;
    const float   *sc = S + (long long)o * ng;

    float a = 0.f;
    for (int g = 0; g < ng; g++) {
        int g0 = g * gs;
        int g1 = g0 + gs; if (g1 > I) g1 = I;
        float ga = 0.f;
        for (int i = g0; i < g1; i++) {
            unsigned char byte = w[i >> 1];
            int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            ga += x[i] * (float)(nib - 8);
        }
        a += ga * sc[g];
    }
    y[o] = a;
}

/* Picchio INT3 gs64. Per output row and group of 64 values the low two bits
 * occupy 16 bytes and the high bit occupies 8 bytes. Code [0,7] maps to [-4,3]. */
__global__ void expert_i3gs(const uint8_t *__restrict__ Q,
                            const float *__restrict__ S,
                            const float *__restrict__ x,
                            float *__restrict__ y,
                            int O, int I) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= O) return;
    int ng = (I + 63) / 64;
    const uint8_t *row = Q + (long long)o * ng * 24;
    const float *sc = S + (long long)o * ng;
    float a = 0.f;
    for (int g = 0; g < ng; g++) {
        const uint8_t *lo = row + (long long)g * 24;
        const uint8_t *hi = lo + 16;
        int base = g * 64;
        int n = I - base; if (n > 64) n = 64;
        float ga = 0.f;
        for (int j = 0; j < n; j++) {
            int low2 = (lo[j >> 2] >> ((j & 3) * 2)) & 3;
            int high1 = (hi[j >> 3] >> (j & 7)) & 1;
            ga += x[base + j] * (float)((low2 | (high1 << 2)) - 4);
        }
        a += ga * sc[g];
    }
    y[o] = a;
}

/* SwiGLU over interleaved (gate, up) pairs, mirroring moe_forward exactly.
 * bias (optional) is added to gate/up before the activation. */
__global__ void swiglu_k(const float *__restrict__ gu, const float *__restrict__ bias,
                         float *__restrict__ h, int half,
                         int clipped, float limit, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    float gate = gu[2 * i], up = gu[2 * i + 1];
    if (bias) { gate += bias[2 * i]; up += bias[2 * i + 1]; }
    if (clipped) {
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        if (up < -limit) up = -limit;
        h[i] = (up + 1.0f) * gate * (1.0f / (1.0f + expf(-alpha * gate)));
    } else {
        h[i] = gate * (1.0f / (1.0f + expf(-gate))) * up;
    }
}

/* out[i] += w * (eo[i] + d_bias[i]) */
__global__ void accum_k(float *__restrict__ out, const float *__restrict__ eo,
                        const float *__restrict__ dbias, float w, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    float v = eo[i];
    if (dbias) v += dbias[i];
    out[i] += w * v;
}

__global__ void zero_k(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}

/* ── host state ─────────────────────────────────────────────────────────── */

struct DevW { int8_t *w; float *s; };
static std::unordered_map<const void *, DevW> g_wcache;   /* resident weights */
struct DevRouter { float *w; float *bias; int E, D; };
static std::unordered_map<const void *, DevRouter> g_rcache;

static float   *g_dx   = nullptr;  size_t g_dx_cap   = 0;  /* activation scratch */
static float   *g_dy   = nullptr;  size_t g_dy_cap   = 0;  /* output scratch */
static uint8_t *g_dq4  = nullptr;  size_t g_dq4_cap  = 0;  /* expert weight scratch */
static float   *g_ds   = nullptr;  size_t g_ds_cap   = 0;  /* expert scale scratch */

/* G2 per-layer scratch (activations only; expert weights live in g_ecache). */
static float   *g_dgu  = nullptr;  size_t g_dgu_cap  = 0;  /* gate_up output */
static float   *g_dh   = nullptr;  size_t g_dh_cap   = 0;  /* swiglu output */
static float   *g_deo  = nullptr;  size_t g_deo_cap  = 0;  /* down output */
static float   *g_dout = nullptr;  size_t g_dout_cap = 0;  /* accumulated token out */

/* G2 VRAM expert cache: hot experts kept resident, keyed by (layer,eid), with a
 * byte-bounded LRU (PIN_VRAM_GB). Independent of the CPU RAM cache. */
struct DevExpert {
    uint8_t *gu; float *gu_s; float *gu_bias;
    uint8_t *d;  float *d_s;  float *d_bias;
    size_t   bytes;
    unsigned long long used;
};
static std::unordered_map<uint64_t, DevExpert> g_ecache;
static unsigned long long g_eclock  = 0;
static size_t             g_vram_used   = 0;
static size_t             g_vram_budget = 0;  /* 0 = set at init from PIN_VRAM_GB */
static unsigned long long g_ehit = 0, g_emiss = 0;

static void free_dev_expert(DevExpert &e) {
    cudaFree(e.gu); cudaFree(e.gu_s); cudaFree(e.gu_bias);
    cudaFree(e.d);  cudaFree(e.d_s);  cudaFree(e.d_bias);
}

static int ensure(void **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*buf) cudaFree(*buf);
    *buf = nullptr; *cap = 0;
    if (cudaMalloc(buf, need) != cudaSuccess) { *buf = nullptr; return -1; }
    *cap = need;
    return 0;
}

/* ── C API ──────────────────────────────────────────────────────────────── */

extern "C" int pgpu_init(void) {
    if (g_ok) return 1;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return 0;
    if (cudaSetDevice(0) != cudaSuccess) return 0;
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        snprintf(g_name, sizeof(g_name), "%s", p.name);
        fprintf(stderr, "[gpu] %s · %.1f/%.1f GB free · sm_%d%d\n",
                p.name, freeb / 1e9, totb / 1e9, p.major, p.minor);
    }

    /* Expert VRAM budget (PIN_VRAM_GB). Default: most of the free VRAM, minus a
     * ~0.6 GB margin for the display / driver / per-layer scratch. */
    const char *v = getenv("PIN_VRAM_GB");
    double gb = v ? atof(v) : 0.0;
    if (gb <= 0.0) { gb = (double)freeb / 1e9 - 0.6; if (gb < 0.2) gb = 0.2; }
    g_vram_budget = (size_t)(gb * 1e9);
    fprintf(stderr, "[gpu] expert VRAM budget: %.2f GB\n", g_vram_budget / 1e9);

    cudaError_t se = cudaStreamCreateWithFlags(&g_compute_stream, cudaStreamNonBlocking);
    if (se != cudaSuccess) {
        fprintf(stderr, "[gpu] compute stream creation failed: %s\n", cudaGetErrorString(se));
        (void)cudaGetLastError();
        g_compute_stream = nullptr;
        return 0;
    }
    g_ok = 1;
    return 1;
}

extern "C" int pgpu_available(void) { return g_ok; }

extern "C" int pgpu_host_register(void *ptr, size_t bytes) {
    if (!g_ok || !ptr || bytes == 0) return 0;
    std::lock_guard<std::mutex> lock(g_host_reg_mutex);
    auto it = g_host_regs.find(ptr);
    if (it != g_host_regs.end()) return it->second >= bytes ? 1 : 0;

    /* Record each new slab registration request to guide cap sizing. */
    g_host_reg_attempts++;
    g_host_reg_attempt_bytes += (uint64_t)bytes;
    if (bytes < g_host_reg_min_attempt) g_host_reg_min_attempt = bytes;
    if (bytes > g_host_reg_max_attempt) g_host_reg_max_attempt = bytes;

    /* flat_aligned_alloc currently returns 4 KiB-aligned slabs. Reject any
     * other pointer instead of asking CUDA to register an invalid range. */
    if (((uintptr_t)ptr & G_HOST_REG_ALIGN_MASK) != 0) {
        g_host_reg_fail++;
        g_host_reg_align_fail++;
        return 0;
    }
    if (!host_reg_fits(g_host_reg_bytes, bytes)) {
        g_host_reg_fail++;
        g_host_reg_cap_fail++;
        g_host_reg_cap_fail_bytes += (uint64_t)bytes;
        return 0;
    }

    cudaError_t e = cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
    if (e != cudaSuccess) {
        g_host_reg_fail++;
        g_host_reg_cuda_fail++;
        g_host_reg_last_cuda_error = e;
        (void)cudaGetLastError(); /* clear the runtime's sticky error state */
        return 0;                 /* caller keeps using pageable memory */
    }

    /* If bookkeeping allocation fails, undo the CUDA registration immediately. */
    try {
        g_host_regs.emplace(ptr, bytes);
    } catch (...) {
        cudaError_t undo = cudaHostUnregister(ptr);
        if (undo != cudaSuccess) {
            fprintf(stderr,
                    "[gpu] FATAL: pinned-registration bookkeeping failed and "
                    "cudaHostUnregister rollback also failed: %s; aborting "
                    "to preserve host allocation ownership\n",
                    cudaGetErrorString(undo));
            fflush(stderr);
            abort();
        }
        g_host_reg_fail++;
        g_host_reg_bookkeeping_fail++;
        return 0;
    }
    g_host_reg_bytes += bytes;
    if (g_host_reg_bytes > g_host_reg_peak_bytes)
        g_host_reg_peak_bytes = g_host_reg_bytes;
    g_host_reg_ok++;
    return 1;
}

extern "C" void pgpu_host_unregister(void *ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_host_reg_mutex);
    auto it = g_host_regs.find(ptr);
    if (it == g_host_regs.end()) return;
    (void)stream_wait("host unregister");
    cudaError_t e = cudaHostUnregister(ptr);
    if (e != cudaSuccess) {
        fprintf(stderr,
                "[gpu] FATAL: cudaHostUnregister failed: %s; aborting because "
                "the C owner must not free or recycle this slab\n",
                cudaGetErrorString(e));
        fflush(stderr);
        abort();
    }
    g_host_reg_bytes -= it->second;
    g_host_regs.erase(it);
}

extern "C" void pgpu_shutdown(void) {
    (void)stream_wait("shutdown");
    {
        std::lock_guard<std::mutex> lock(g_host_reg_mutex);
        for (auto &kv : g_host_regs) {
            cudaError_t e = cudaHostUnregister(kv.first);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "[gpu] FATAL: shutdown cudaHostUnregister failed: %s; "
                        "aborting before releasing any remaining resources\n",
                        cudaGetErrorString(e));
                fflush(stderr);
                abort();
            }
        }
        g_host_regs.clear();
        g_host_reg_bytes = 0;
    }
    fprintf(stderr, "[gpu] pinned host slabs: %llu registered, %llu failed, peak %.1f MiB / 128 MiB cap\n",
            (unsigned long long)g_host_reg_ok, (unsigned long long)g_host_reg_fail,
            g_host_reg_peak_bytes / 1048576.0);
    fprintf(stderr, "[gpu] pinned failures: cap=%llu align=%llu cuda=%llu bookkeeping=%llu\n",
            (unsigned long long)g_host_reg_cap_fail,
            (unsigned long long)g_host_reg_align_fail,
            (unsigned long long)g_host_reg_cuda_fail,
            (unsigned long long)g_host_reg_bookkeeping_fail);
    if (g_host_reg_attempts > 0)
        fprintf(stderr, "[gpu] pinned requests: %llu slabs, %.1f MiB total, min %.1f MiB, max %.1f MiB; cap-blocked %.1f MiB\n",
                (unsigned long long)g_host_reg_attempts,
                g_host_reg_attempt_bytes / 1048576.0,
                g_host_reg_min_attempt / 1048576.0,
                g_host_reg_max_attempt / 1048576.0,
                g_host_reg_cap_fail_bytes / 1048576.0);
    if (g_host_reg_last_cuda_error != cudaSuccess)
        fprintf(stderr, "[gpu] last cudaHostRegister error: %s\n",
                cudaGetErrorString(g_host_reg_last_cuda_error));
    if (g_emiss + g_ehit > 0)
        fprintf(stderr, "[gpu] expert VRAM cache: %llu hit / %llu miss (%.1f%%), %.2f GB resident\n",
                (unsigned long long)g_ehit, (unsigned long long)g_emiss,
                100.0 * g_ehit / (g_ehit + g_emiss), g_vram_used / 1e9);
    for (auto &kv : g_wcache)  { cudaFree(kv.second.w); cudaFree(kv.second.s); }
    for (auto &kv : g_rcache)  { cudaFree(kv.second.w); cudaFree(kv.second.bias); }
    for (auto &kv : g_ecache)  { free_dev_expert(kv.second); }
    g_wcache.clear(); g_rcache.clear(); g_ecache.clear();
    g_vram_used = 0;
    if (g_dx)   cudaFree(g_dx);
    if (g_dy)   cudaFree(g_dy);
    if (g_dq4)  cudaFree(g_dq4);
    if (g_ds)   cudaFree(g_ds);
    if (g_dgu)  cudaFree(g_dgu);
    if (g_dh)   cudaFree(g_dh);
    if (g_deo)  cudaFree(g_deo);
    if (g_dout) cudaFree(g_dout);
    g_dx = g_dy = g_dgu = g_dh = g_deo = g_dout = nullptr; g_ds = nullptr; g_dq4 = nullptr;
    g_dx_cap = g_dy_cap = g_dq4_cap = g_ds_cap = 0;
    g_dgu_cap = g_dh_cap = g_deo_cap = g_dout_cap = 0;
    if (g_compute_stream) {
        cudaError_t e = cudaStreamDestroy(g_compute_stream);
        if (e != cudaSuccess)
            fprintf(stderr, "[gpu] compute stream destroy failed: %s\n", cudaGetErrorString(e));
        g_compute_stream = nullptr;
    }
    g_ok = 0;
}

extern "C" int pgpu_matmul_i8(float *y, const float *x,
                              const int8_t *w, const float *scale,
                              int O, int I, const void *weight_id) {
    if (!g_ok) return -1;

    /* Upload (and cache) the weight the first time we see this weight_id. */
    auto it = g_wcache.find(weight_id);
    if (it == g_wcache.end()) {
        DevW dw{nullptr, nullptr};
        if (cudaMalloc(&dw.w, (size_t)O * I) != cudaSuccess) return -1;
        if (cudaMalloc(&dw.s, (size_t)O * sizeof(float)) != cudaSuccess) {
            cudaFree(dw.w); return -1;
        }
        if (stream_copy(dw.w, w, (size_t)O * I, cudaMemcpyHostToDevice, "lm_head weights") != 0 ||
            stream_copy(dw.s, scale, (size_t)O * sizeof(float), cudaMemcpyHostToDevice, "lm_head scales") != 0 ||
            stream_wait("lm_head weight upload") != 0) {
            cudaFree(dw.w); cudaFree(dw.s); return -1;
        }
        g_wcache[weight_id] = dw;
        it = g_wcache.find(weight_id);
    }

    if (ensure((void **)&g_dx, &g_dx_cap, (size_t)I * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dy, &g_dy_cap, (size_t)O * sizeof(float)) != 0) return -1;
    CU_TRY(stream_copy(g_dx, x, (size_t)I * sizeof(float), cudaMemcpyHostToDevice, "lm_head input"));

    lm_head_i8<<<O, 128, 0, g_compute_stream>>>(it->second.w, it->second.s, g_dx, g_dy, O, I);
    CU_TRY(cudaGetLastError());
    CU_TRY(stream_copy(y, g_dy, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost, "matmul output"));
    return stream_wait("matmul output");
}

extern "C" int pgpu_router_upload(const float *w, const float *bias,
                                   int E, int D, const void *weight_id) {
    if (!g_ok || !w || !weight_id || E <= 0 || D <= 0) return -1;
    if (g_rcache.find(weight_id) != g_rcache.end()) return 0;
    DevRouter r{nullptr, nullptr, E, D};
    if (cudaMalloc(&r.w, (size_t)E * D * sizeof(float)) != cudaSuccess)
        return -1;
    if (bias && cudaMalloc(&r.bias, (size_t)E * sizeof(float)) != cudaSuccess) {
        cudaFree(r.w); return -1;
    }
    if (stream_copy(r.w, w, (size_t)E * D * sizeof(float), cudaMemcpyHostToDevice, "router weights") != 0 ||
        (bias && stream_copy(r.bias, bias, (size_t)E * sizeof(float), cudaMemcpyHostToDevice, "router bias") != 0) ||
        stream_wait("router upload") != 0) {
        cudaFree(r.w); cudaFree(r.bias); return -1;
    }
    g_rcache.emplace(weight_id, r);
    return 0;
}

extern "C" int pgpu_router_scores(float *scores, const float *x,
                                   const float *w, const float *bias,
                                   int E, int D, const void *weight_id) {
    if (!g_ok || !scores || !x) return -1;
    if (pgpu_router_upload(w, bias, E, D, weight_id) != 0) return -1;
    auto it = g_rcache.find(weight_id);
    if (it == g_rcache.end() || it->second.E != E || it->second.D != D) return -1;
    if (ensure((void **)&g_dx, &g_dx_cap, (size_t)D * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dy, &g_dy_cap, (size_t)E * sizeof(float)) != 0) return -1;
    CU_TRY(stream_copy(g_dx, x, (size_t)D * sizeof(float), cudaMemcpyHostToDevice, "router input"));
    router_f32<<<E, 256, 0, g_compute_stream>>>(it->second.w, it->second.bias, g_dx, g_dy, E, D);
    CU_TRY(cudaGetLastError());
    CU_TRY(stream_copy(scores, g_dy, (size_t)E * sizeof(float), cudaMemcpyDeviceToHost, "router scores"));
    return stream_wait("router scores");
}

extern "C" int pgpu_matmul_i4gs(float *y, const float *x,
                                const uint8_t *q4, const float *scale,
                                int O, int I, int gs) {
    if (!g_ok) return -1;
    int rb = (I + 1) / 2;
    int ng = (I + gs - 1) / gs;

    if (ensure((void **)&g_dx,  &g_dx_cap,  (size_t)I * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dy,  &g_dy_cap,  (size_t)O * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dq4, &g_dq4_cap, (size_t)O * rb) != 0) return -1;
    if (ensure((void **)&g_ds,  &g_ds_cap,  (size_t)O * ng * sizeof(float)) != 0) return -1;

    CU_TRY(stream_copy(g_dx, x, (size_t)I * sizeof(float), cudaMemcpyHostToDevice, "expert input"));
    CU_TRY(stream_copy(g_dq4, q4, (size_t)O * rb, cudaMemcpyHostToDevice, "INT4 weights"));
    CU_TRY(stream_copy(g_ds, scale, (size_t)O * ng * sizeof(float), cudaMemcpyHostToDevice, "expert scales"));

    int threads = 128, blocks = (O + threads - 1) / threads;
    expert_i4gs<<<blocks, threads, 0, g_compute_stream>>>(g_dq4, g_ds, g_dx, g_dy, O, I, gs);
    CU_TRY(cudaGetLastError());
    CU_TRY(stream_copy(y, g_dy, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost, "matmul output"));
    return stream_wait("matmul output");
}

extern "C" int pgpu_matmul_i3gs(float *y, const float *x,
                                 const uint8_t *q3, const float *scale,
                                 int O, int I, int gs) {
    if (!g_ok || gs != 64) return -1;
    int ng = (I + 63) / 64;
    size_t qbytes = (size_t)O * ng * 24;

    if (ensure((void **)&g_dx,  &g_dx_cap,  (size_t)I * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dy,  &g_dy_cap,  (size_t)O * sizeof(float)) != 0) return -1;
    if (ensure((void **)&g_dq4, &g_dq4_cap, qbytes) != 0) return -1;
    if (ensure((void **)&g_ds,  &g_ds_cap,  (size_t)O * ng * sizeof(float)) != 0) return -1;

    CU_TRY(stream_copy(g_dx, x, (size_t)I * sizeof(float), cudaMemcpyHostToDevice, "expert input"));
    CU_TRY(stream_copy(g_dq4, q3, qbytes, cudaMemcpyHostToDevice, "INT3 weights"));
    CU_TRY(stream_copy(g_ds, scale, (size_t)O * ng * sizeof(float), cudaMemcpyHostToDevice, "expert scales"));

    int threads = 128, blocks = (O + threads - 1) / threads;
    expert_i3gs<<<blocks, threads, 0, g_compute_stream>>>(g_dq4, g_ds, g_dx, g_dy, O, I);
    CU_TRY(cudaGetLastError());
    CU_TRY(stream_copy(y, g_dy, (size_t)O * sizeof(float), cudaMemcpyDeviceToHost, "matmul output"));
    return stream_wait("matmul output");
}

/* Resolve an expert in the VRAM cache, uploading (and evicting LRU) on a miss.
 * Returns a stable pointer into g_ecache, or NULL on allocation failure. */
static DevExpert *get_expert(uint64_t key, const PgpuExpert *e,
                             size_t gu_wb, size_t gu_sb, size_t gu_bb,
                             size_t d_wb, size_t d_sb, size_t d_bb) {
    auto it = g_ecache.find(key);
    if (it != g_ecache.end()) { g_ehit++; return &it->second; }
    /* A resident-only expert (host weights intentionally not loaded, NULL ptrs)
     * that is no longer cached — e.g. evicted before we reached it. Signal the
     * caller to fall back rather than dereference the absent host pointers. */
    if (!e->gu_q4 || !e->d_q4) return NULL;
    g_emiss++;

    size_t need = gu_wb + gu_sb + d_wb + d_sb
                + (e->gu_bias ? gu_bb : 0) + (e->d_bias ? d_bb : 0);

    /* Evict least-recently-used experts until the newcomer fits the budget. */
    while (g_vram_budget > 0 && g_vram_used + need > g_vram_budget && !g_ecache.empty()) {
        auto victim = g_ecache.begin();
        for (auto i = g_ecache.begin(); i != g_ecache.end(); ++i)
            if (i->second.used < victim->second.used) victim = i;
        if (stream_wait("expert eviction") != 0) return NULL;
        g_vram_used -= victim->second.bytes;
        free_dev_expert(victim->second);
        g_ecache.erase(victim);
    }

    DevExpert de; memset(&de, 0, sizeof(de)); de.bytes = need;
    bool ok = cudaMalloc(&de.gu,   gu_wb) == cudaSuccess
           && cudaMalloc(&de.gu_s, gu_sb) == cudaSuccess
           && cudaMalloc(&de.d,    d_wb)  == cudaSuccess
           && cudaMalloc(&de.d_s,  d_sb)  == cudaSuccess;
    if (ok && e->gu_bias) ok = cudaMalloc(&de.gu_bias, gu_bb) == cudaSuccess;
    if (ok && e->d_bias)  ok = cudaMalloc(&de.d_bias,  d_bb)  == cudaSuccess;
    if (ok)
        ok = stream_copy(de.gu, e->gu_q4, gu_wb, cudaMemcpyHostToDevice, "expert gate weights") == 0
          && stream_copy(de.gu_s, e->gu_s, gu_sb, cudaMemcpyHostToDevice, "expert gate scales") == 0
          && stream_copy(de.d, e->d_q4, d_wb, cudaMemcpyHostToDevice, "expert down weights") == 0
          && stream_copy(de.d_s, e->d_s, d_sb, cudaMemcpyHostToDevice, "expert down scales") == 0;
    if (ok && e->gu_bias)
        ok = stream_copy(de.gu_bias, e->gu_bias, gu_bb, cudaMemcpyHostToDevice, "expert gate bias") == 0;
    if (ok && e->d_bias)
        ok = stream_copy(de.d_bias, e->d_bias, d_bb, cudaMemcpyHostToDevice, "expert down bias") == 0;
    if (ok) ok = stream_wait("expert upload") == 0;
    else (void)stream_wait("partial expert upload");
    if (!ok) { free_dev_expert(de); return NULL; }

    g_vram_used += need;
    auto res = g_ecache.emplace(key, de);
    return &res.first->second;
}

extern "C" int pgpu_moe_layer(float *out, const float *x, int D, int moe_inter,
                              int layer, const PgpuExpert *experts,
                              const float *wsum, int nexp, int gs, int bits,
                              int swiglu_clipped, float swiglu_limit,
                              float swiglu_alpha) {
    if (!g_ok || (bits != 3 && bits != 4) || gs != 64) return -1;
    int half  = moe_inter / 2;
    int gu_ng = (D + gs - 1) / gs;
    int d_ng  = (half + gs - 1) / gs;
    size_t gu_rowb = bits == 3 ? (size_t)gu_ng * 24 : (size_t)(D + 1) / 2;
    size_t d_rowb  = bits == 3 ? (size_t)d_ng * 24 : (size_t)(half + 1) / 2;
    size_t gu_wb = (size_t)moe_inter * gu_rowb, gu_sb = (size_t)moe_inter * gu_ng * 4;
    size_t d_wb  = (size_t)D * d_rowb,          d_sb  = (size_t)D * d_ng * 4;
    size_t gu_bb = (size_t)moe_inter * 4,     d_bb  = (size_t)D * 4;

    if (ensure((void **)&g_dx,   &g_dx_cap,   (size_t)D * 4)         != 0) return -1;
    if (ensure((void **)&g_dgu,  &g_dgu_cap,  (size_t)moe_inter * 4) != 0) return -1;
    if (ensure((void **)&g_dh,   &g_dh_cap,   (size_t)half * 4)      != 0) return -1;
    if (ensure((void **)&g_deo,  &g_deo_cap,  (size_t)D * 4)         != 0) return -1;
    if (ensure((void **)&g_dout, &g_dout_cap, (size_t)D * 4)         != 0) return -1;

    if (stream_copy(g_dx, x, (size_t)D * 4, cudaMemcpyHostToDevice, "MoE input") != 0) return -1;
    zero_k<<<(D + 127) / 128, 128, 0, g_compute_stream>>>(g_dout, D);

    /* Protect this call's already-resident experts from being evicted by the
     * misses in this same call: bump their LRU stamp before any upload evicts. */
    for (int k = 0; k < nexp; k++) {
        uint64_t rk = ((uint64_t)(unsigned)layer << 32) | (unsigned)experts[k].eid;
        auto rit = g_ecache.find(rk);
        if (rit != g_ecache.end()) rit->second.used = ++g_eclock;
    }

    for (int k = 0; k < nexp; k++) {
        const PgpuExpert *e = &experts[k];
        uint64_t key = ((uint64_t)(unsigned)layer << 32) | (unsigned)e->eid;
        DevExpert *de = get_expert(key, e, gu_wb, gu_sb, gu_bb, d_wb, d_sb, d_bb);
        if (!de) { (void)stream_wait("MoE expert lookup failure"); return -1; }
        de->used = ++g_eclock;

        if (bits == 3)
            expert_i3gs<<<(moe_inter + 127) / 128, 128, 0, g_compute_stream>>>(de->gu, de->gu_s, g_dx, g_dgu, moe_inter, D);
        else
            expert_i4gs<<<(moe_inter + 127) / 128, 128, 0, g_compute_stream>>>(de->gu, de->gu_s, g_dx, g_dgu, moe_inter, D, gs);
        swiglu_k<<<(half + 127) / 128, 128, 0, g_compute_stream>>>(g_dgu, de->gu_bias, g_dh, half,
                                              swiglu_clipped, swiglu_limit, swiglu_alpha);
        if (bits == 3)
            expert_i3gs<<<(D + 127) / 128, 128, 0, g_compute_stream>>>(de->d, de->d_s, g_dh, g_deo, D, half);
        else
            expert_i4gs<<<(D + 127) / 128, 128, 0, g_compute_stream>>>(de->d, de->d_s, g_dh, g_deo, D, half, gs);
        accum_k<<<(D + 127) / 128, 128, 0, g_compute_stream>>>(g_dout, g_deo, de->d_bias, wsum[k], D);
    }
    CU_TRY(cudaGetLastError());
    CU_TRY(stream_copy(out, g_dout, (size_t)D * 4, cudaMemcpyDeviceToHost, "MoE output"));
    return stream_wait("MoE layer");
}

extern "C" int pgpu_moe_resident(int layer, int eid) {
    if (!g_ok) return 0;
    uint64_t key = ((uint64_t)(unsigned)layer << 32) | (unsigned)eid;
    return g_ecache.find(key) != g_ecache.end() ? 1 : 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  STANDALONE NUMERIC SELF-TEST + MICROBENCH (nvcc -DPGPU_TEST)
 * ══════════════════════════════════════════════════════════════════════════ */
#if defined(PGPU_STREAM_FAULT_TEST)
int main(void) {
    if (!pgpu_init()) {
        fprintf(stderr, "no CUDA device; stream fault test skipped\n");
        return 77;
    }
    const int E = 8, D = 64;
    float *w = (float *)malloc((size_t)E * D * sizeof(float));
    float *b = (float *)malloc((size_t)E * sizeof(float));
    float *scores = (float *)malloc((size_t)E * sizeof(float));
    float *x = nullptr;
    if (!w || !b || !scores ||
        posix_memalign((void **)&x, 4096, (size_t)D * sizeof(float)) != 0) {
        fprintf(stderr, "fault-test allocation failed\n");
        return 2;
    }
    for (int i = 0; i < E * D; i++) w[i] = 0.001f * (float)(i % 17);
    for (int i = 0; i < E; i++) b[i] = 0.01f * (float)i;
    for (int i = 0; i < D; i++) x[i] = 0.02f * (float)(i % 13);

    if (!pgpu_host_register(x, (size_t)D * sizeof(float))) {
        fprintf(stderr, "fault-test pinned host registration failed\n");
        return 3;
    }
    if (pgpu_router_upload(w, b, E, D, w) != 0) {
        fprintf(stderr, "fault-test router upload failed\n");
        return 4;
    }

    /* router_scores queues pinned-host H2D, a kernel, and D2H. The injected
     * failed drain must abort before this caller can unregister/free x. */
    pgpu_test_fail_next_stream_wait();
    (void)pgpu_router_scores(scores, x, w, b, E, D, w);
    fprintf(stderr, "FAULT TEST FAIL: backend returned after failed drain\n");
    return 5;
}
#elif defined(PGPU_PINNED_TEST)
int main(void) {
    if (!pgpu_init()) { fprintf(stderr, "no CUDA device\n"); return 1; }
    void *p = nullptr;
    if (posix_memalign(&p, 4096, 4096) != 0 || !p) return 2;

    int first = pgpu_host_register(p, 4096);
    int duplicate = pgpu_host_register(p, 4096);
    pgpu_host_unregister(p);
    int again = pgpu_host_register(p, 4096);
    int unaligned = pgpu_host_register((char *)p + 1, 4096);
    int over_cap = pgpu_host_register(p, G_HOST_REG_LIMIT + 1);
    int exact_cap = host_reg_fits(0, G_HOST_REG_LIMIT);
    int cap_plus_one = !host_reg_fits(0, G_HOST_REG_LIMIT + 1);
    int used_at_cap = !host_reg_fits(G_HOST_REG_LIMIT, 1);
    int used_over_cap = !host_reg_fits(G_HOST_REG_LIMIT + 1, 0);
    int size_overflow = !host_reg_fits(0, (size_t)-1);
    pgpu_host_unregister(p);
    pgpu_shutdown();
    free(p);

    int ok = first == 1 && duplicate == 1 && again == 1 &&
             unaligned == 0 && over_cap == 0 && exact_cap &&
             cap_plus_one && used_at_cap && used_over_cap && size_overflow;
    fprintf(stderr, "[gpu] pinned host self-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}
#elif defined(PGPU_ROUTER_TEST)
int main(void) {
    if (!pgpu_init()) { fprintf(stderr, "no CUDA device\n"); return 1; }
    const int E = 128, D = 2880;
    float *w = (float *)malloc((size_t)E * D * sizeof(float));
    float *b = (float *)malloc((size_t)E * sizeof(float));
    float *x = (float *)malloc((size_t)D * sizeof(float));
    float *cpu = (float *)malloc((size_t)E * sizeof(float));
    float *gpu = (float *)malloc((size_t)E * sizeof(float));
    if (!w || !b || !x || !cpu || !gpu) return 2;
    srand(1234);
    for (int i = 0; i < E * D; i++) w[i] = (float)(rand() % 2001 - 1000) / 10000.f;
    for (int i = 0; i < D; i++) x[i] = (float)(rand() % 2001 - 1000) / 1000.f;
    for (int e = 0; e < E; e++) {
        b[e] = (float)(rand() % 201 - 100) / 1000.f;
        float a = b[e];
        for (int i = 0; i < D; i++) a += x[i] * w[(long long)e * D + i];
        cpu[e] = a;
    }
    if (pgpu_router_upload(w, b, E, D, w) != 0 ||
        pgpu_router_scores(gpu, x, w, b, E, D, w) != 0) return 3;
    float md = 0.f; int ct = 0, gt = 0;
    for (int e = 0; e < E; e++) {
        float d = fabsf(cpu[e] - gpu[e]); if (d > md) md = d;
        if (cpu[e] > cpu[ct]) ct = e;
        if (gpu[e] > gpu[gt]) gt = e;
    }
    printf("router F32 [%d x %d]: CPU top1=%d GPU top1=%d max_abs=%.7g %s\n",
           E, D, ct, gt, md, ct == gt && md < 1e-4f ? "OK" : "FAIL");
    pgpu_shutdown();
    free(w); free(b); free(x); free(cpu); free(gpu);
    return ct == gt && md < 1e-4f ? 0 : 4;
}
#elif defined(PGPU_TEST)
#include <time.h>

static double wall(void) { return (double)clock() / CLOCKS_PER_SEC; }

static void ref_i8(float *y, const float *x, const int8_t *w,
                   const float *sc, int O, int I) {
    for (int o = 0; o < O; o++) {
        double a = 0;
        const int8_t *wr = w + (long long)o * I;
        for (int i = 0; i < I; i++) a += (double)x[i] * (double)wr[i];
        y[o] = (float)(a * sc[o]);
    }
}

static void ref_i4gs(float *y, const float *x, const uint8_t *q4,
                     const float *sc, int O, int I, int gs) {
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q4 + (long long)o * rb;
        const float *s = sc + (long long)o * ng;
        double a = 0;
        for (int g = 0; g < ng; g++) {
            int g0 = g * gs, g1 = g0 + gs; if (g1 > I) g1 = I;
            double ga = 0;
            for (int i = g0; i < g1; i++) {
                unsigned char byte = w[i >> 1];
                int nib = (i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                ga += (double)x[i] * (double)(nib - 8);
            }
            a += ga * (double)s[g];
        }
        y[o] = (float)a;
    }
}

static int argmax(const float *y, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (y[i] > y[b]) b = i; return b;
}
static float maxabs(const float *a, const float *b, int n) {
    float m = 0; for (int i = 0; i < n; i++) { float d = fabsf(a[i]-b[i]); if (d>m) m=d; } return m;
}

int main(void) {
    if (!pgpu_init()) { fprintf(stderr, "no CUDA device\n"); return 1; }

    const int D = 2880;

    /* ---- lm_head INT8: [vocab, D] ---- */
    const int V = 201088;
    float  *x   = (float  *)malloc((size_t)D * 4);
    int8_t *w8  = (int8_t *)malloc((size_t)V * D);
    float  *s8  = (float  *)malloc((size_t)V * 4);
    float  *yc  = (float  *)malloc((size_t)V * 4);
    float  *yg  = (float  *)malloc((size_t)V * 4);
    srand(1234);
    for (int i = 0; i < D; i++) x[i] = (float)(rand() % 2001 - 1000) / 1000.f;
    for (long long i = 0; i < (long long)V * D; i++) w8[i] = (int8_t)(rand() % 255 - 127);
    for (int o = 0; o < V; o++) s8[o] = 0.002f + (float)(rand() % 100) / 100000.f;

    ref_i8(yc, x, w8, s8, V, D);
    /* warm-up + upload */
    pgpu_matmul_i8(yg, x, w8, s8, V, D, (const void *)w8);
    double t0 = wall();
    int iters = 20;
    for (int k = 0; k < iters; k++) pgpu_matmul_i8(yg, x, w8, s8, V, D, (const void *)w8);
    cudaDeviceSynchronize();
    double tg = (wall() - t0) / iters * 1000.0;

    double t1 = wall(); ref_i8(yc, x, w8, s8, V, D); double tc = (wall() - t1) * 1000.0;

    printf("lm_head INT8 [%d x %d]\n", V, D);
    printf("  top-1  CPU=%d  GPU=%d  %s\n", argmax(yc, V), argmax(yg, V),
           argmax(yc, V) == argmax(yg, V) ? "MATCH" : "MISMATCH");
    printf("  max|dCPU-GPU| = %.4g\n", maxabs(yc, yg, V));
    printf("  GPU %.2f ms/tok (resident) · CPU(1thr) %.1f ms/tok\n\n", tg, tc);

    /* ---- expert INT4 gs64: gate_up [5760, D] ---- */
    const int O = 5760, gs = 64;
    int rb = (D + 1) / 2, ng = (D + gs - 1) / gs;
    uint8_t *q4 = (uint8_t *)malloc((size_t)O * rb);
    float   *se = (float   *)malloc((size_t)O * ng * 4);
    float   *ec = (float   *)malloc((size_t)O * 4);
    float   *eg = (float   *)malloc((size_t)O * 4);
    for (long long i = 0; i < (long long)O * rb; i++) q4[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < O * ng; i++) se[i] = 0.01f + (float)(rand() % 100) / 100000.f;

    ref_i4gs(ec, x, q4, se, O, D, gs);
    pgpu_matmul_i4gs(eg, x, q4, se, O, D, gs);
    double t2 = wall();
    for (int k = 0; k < iters; k++) pgpu_matmul_i4gs(eg, x, q4, se, O, D, gs);
    cudaDeviceSynchronize();
    double te = (wall() - t2) / iters * 1000.0;

    printf("expert INT4 gs64 [%d x %d] (streamed via scratch)\n", O, D);
    printf("  max|dCPU-GPU| = %.4g\n", maxabs(ec, eg, O));
    printf("  GPU %.3f ms/expert (upload+compute)\n", te);

    pgpu_shutdown();
    return 0;
}
#endif
