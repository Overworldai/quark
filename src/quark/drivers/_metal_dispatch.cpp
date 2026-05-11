/**
 * _metal_dispatch — nanobind-based Metal compute dispatch.
 *
 * Uses Apple's metal-cpp + nanobind (matching MLX's binding stack)
 * for low-overhead Python→C calls. Eager encoding: each launch
 * encodes immediately into the persistent command encoder; sync()
 * commits and waits.
 */

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
// Single-header amalgamation of Apple's metal-cpp, compute path only.
// Regenerate from a fresh metal-cpp release with:
//   python3 SingleHeader/MakeSingleHeader.py -o Metal-cpp.hpp \
//     Metal/MTLDevice.hpp Metal/MTLBuffer.hpp Metal/MTLCommandQueue.hpp \
//     Metal/MTLCommandBuffer.hpp Metal/MTLComputeCommandEncoder.hpp \
//     Metal/MTLComputePipeline.hpp Metal/MTLLibrary.hpp
// The seed list above covers every MTL::* and NS::* symbol this file uses.
// If a new Metal subsystem is needed (raytracing, render, etc.) add its
// top-level header to the seed list and regenerate.
#include "_metal_cpp/Metal-cpp.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Threading model + Clang thread-safety annotations
// ---------------------------------------------------------------------------
//
// All dispatcher entry points are called from Python under the GIL, so the
// "main thread" sees serialized access to every global below by default.
// The one exception is ``addCompletedHandler`` callbacks installed on
// ``MTL::CommandBuffer``: those fire on Metal's callback queue (off-thread)
// once the GPU finishes the buffer.
//
// **Single-owner mutation.** No completion handler may mutate a "main"
// global directly; instead it appends a tiny task to a lock-protected
// pending queue, and the main thread drains that queue at well-defined
// points. The interesting state (e.g. ``g_buf_to_fence``) stays
// single-threaded, the only cross-thread mutation is on a small
// append-only queue with an O(1) critical section. This was a hash-table
// corruption bug for ``g_buf_to_fence`` once — the off-thread completion
// handler was mid-``erase`` while the main thread was mid-``find`` — that
// surfaced as SIGSEGV after 200-1500 frames in long-running consumers.
//
// To make the contract enforceable in code review (and at compile time),
// every cross-thread global is tagged with ``TSA_GUARDED_BY(<lock>)`` and
// the locks are wrapped in a ``Mutex`` capability type. ``-Wthread-safety``
// in setup.py turns these into compile-time errors when an unprotected
// access slips through. Plain ``std::mutex`` is intentionally NOT used as
// the global-protector type — the standard library doesn't tag it with
// capability attributes, so Clang's analysis can't reason through it.
//
// When adding a new global that's touched from a completion handler:
//   1. Don't. Have the handler enqueue a task to a pending queue and let
//      the main thread mutate the real state.
//   2. The pending queue itself is the only cross-thread global; protect
//      it with ``Mutex`` + ``TSA_GUARDED_BY`` + ``MutexGuard`` at every
//      access. The compiler will refuse to build until all three are in
//      place.

#if defined(__clang__)
#define TSA_CAPABILITY(x)     __attribute__((capability(x)))
#define TSA_SCOPED_CAPABILITY __attribute__((scoped_lockable))
#define TSA_GUARDED_BY(x)     __attribute__((guarded_by(x)))
#define TSA_ACQUIRE(...)      __attribute__((acquire_capability(__VA_ARGS__)))
#define TSA_RELEASE(...)      __attribute__((release_capability(__VA_ARGS__)))
#define TSA_REQUIRES(...)     __attribute__((requires_capability(__VA_ARGS__)))
#define TSA_NO_ANALYSIS       __attribute__((no_thread_safety_analysis))
#else
#define TSA_CAPABILITY(x)
#define TSA_SCOPED_CAPABILITY
#define TSA_GUARDED_BY(x)
#define TSA_ACQUIRE(...)
#define TSA_RELEASE(...)
#define TSA_REQUIRES(...)
#define TSA_NO_ANALYSIS
#endif

// Thin capability-tagged wrapper around ``std::mutex``. Identical
// runtime behaviour (no extra fields, methods are inlined into the
// std::mutex calls); the attributes only affect Clang's static
// thread-safety analysis. Use ``MutexGuard`` rather than
// ``std::lock_guard<std::mutex>`` so the analysis can track the
// scope of held capabilities.
class TSA_CAPABILITY("mutex") Mutex {
public:
    void lock() TSA_ACQUIRE() { mu_.lock(); }
    void unlock() TSA_RELEASE() { mu_.unlock(); }
private:
    std::mutex mu_;
};

class TSA_SCOPED_CAPABILITY MutexGuard {
public:
    explicit MutexGuard(Mutex& m) TSA_ACQUIRE(m) : mu_(m) { mu_.lock(); }
    ~MutexGuard() TSA_RELEASE() { mu_.unlock(); }
    MutexGuard(const MutexGuard&) = delete;
    MutexGuard& operator=(const MutexGuard&) = delete;
private:
    Mutex& mu_;
};

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
//
// Dispatch is eager (encoder lives across calls): ``launch`` opens an
// encoder on first call and writes to it directly. Auto-commit at
// g_max_ops closes the encoder and submits the command buffer, freeing
// the GPU to start processing while Python is still encoding the next
// batch. A purely lazy queue (defer all encoder work to eval) starves
// the GPU on chained dispatches and runs ~30% slower — MLX's "lazy" is
// Python-side graph construction, not deferred encoding.
//
// MLX's ``eval`` walks its graph and dispatches into a command buffer
// the same way we do here; the speedup comes from cheap per-call work,
// not from the eval pass.

static MTL::Device* g_device = nullptr;
static MTL::CommandQueue* g_queue = nullptr;
static MTL::CommandBuffer* g_cmd = nullptr;
static MTL::ComputeCommandEncoder* g_enc = nullptr;
static int g_buffer_ops = 0;
static int g_max_ops = 50;
static std::vector<MTL::CommandBuffer*> g_committed_uncompleted;

// ─────────────────────────────────────────────────────────────────────
// MLX-style cross-encoder fence machinery (mirrors
// mlx/backend/metal/device.cpp:CommandEncoder::end_encoding).
// ─────────────────────────────────────────────────────────────────────
//
// Untracked buffers do NOT propagate memory visibility across encoder
// (or command-buffer) boundaries on Apple Silicon. The auto-commit at
// g_max_ops splits dispatches across multiple command buffers, so any
// buffer written by encoder A and read by encoder B needs an explicit
// fence handshake or B reads stale memory.
//
// Pattern (per encoder):
//   • Each encoder owns one MTL::Fence.
//   • Track all unique buffers this encoder reads (g_curr_inputs)
//     and all it writes (g_curr_outputs).
//   • At endEncoding:
//       – For every input that appears in g_buf_to_fence, waitForFence
//         on the fence that produced it (deduped — one wait per fence).
//       – Register every output as produced by THIS encoder's fence.
//       – updateFence on this encoder's fence.
//       – Add a completion handler that prunes those outputs from the
//         map once the command buffer completes (so the map doesn't
//         grow unboundedly across the program lifetime).
//
// Within an encoder, the existing memoryBarrier(BarrierScopeBuffers)
// path (g_pending_outputs RAW check) handles read-after-write between
// dispatches — fences only handle the cross-encoder gap.
static MTL::Fence* g_curr_fence = nullptr;               // main thread only
static std::unordered_set<MTL::Buffer*> g_curr_inputs;   // main thread only
static std::unordered_set<MTL::Buffer*> g_curr_outputs;  // main thread only

// Single-threaded — only ``encode_fence_handshake`` (main thread) reads
// or writes this map. Completion handlers on Metal's callback queue
// don't touch it directly; they enqueue cleanup tasks onto
// ``g_pending_fence_cleanups`` (below) and the next main-thread
// ``encode_fence_handshake`` drains them. No lock needed here.
static std::unordered_map<MTL::Buffer*, MTL::Fence*> g_buf_to_fence;

// Cleanup tasks owed to the main thread by completion handlers. Each
// entry is ``(fence_ref, outputs_to_drop)``: the main thread will, on
// its next ``encode_fence_handshake``, look up each output in
// ``g_buf_to_fence`` and erase it iff the entry still points to
// ``fence_ref`` (a later writer may have already overwritten it).
//
// This is the ONLY global the completion-handler queue mutates.
// Critical sections are O(1) — emplace_back on the producer side,
// swap on the consumer side — so the lock is uncontended in steady
// state. Tagged ``TSA_GUARDED_BY`` so any new access has to grab
// the mutex (or the build fails).
static Mutex g_pending_cleanups_mu;
using FenceCleanup = std::pair<MTL::Fence*, std::unordered_set<MTL::Buffer*>>;
static std::vector<FenceCleanup> g_pending_fence_cleanups
    TSA_GUARDED_BY(g_pending_cleanups_mu);

static void track_input(MTL::Buffer* b) {
    if (b) g_curr_inputs.insert(b);
}
static void track_output(MTL::Buffer* b) {
    if (b) g_curr_outputs.insert(b);
}

static void ensure_curr_fence() {
    if (!g_curr_fence) {
        g_curr_fence = g_device->newFence();
    }
}

// Apply any cleanup tasks that completion handlers have appended to
// ``g_pending_fence_cleanups`` since the last call. Runs on the main
// thread so it can mutate ``g_buf_to_fence`` without locking it. The
// queue handoff itself takes ``g_pending_cleanups_mu`` for an O(1)
// swap so the completion handler's append never blocks on the main
// thread's iteration.
static void drain_pending_fence_cleanups() {
    std::vector<FenceCleanup> batch;
    {
        MutexGuard lk(g_pending_cleanups_mu);
        batch.swap(g_pending_fence_cleanups);
    }
    for (auto& [fence_ref, outputs] : batch) {
        for (auto* o : outputs) {
            auto it = g_buf_to_fence.find(o);
            if (it != g_buf_to_fence.end() && it->second == fence_ref) {
                g_buf_to_fence.erase(it);
            }
        }
    }
}

// Called RIGHT BEFORE g_enc->endEncoding() — wires up the fence
// handshake. Mirrors MLX's CommandEncoder::end_encoding.
static void encode_fence_handshake() {
    if (!g_enc) return;
    ensure_curr_fence();

    // Drain any cleanup work owed by completion handlers that fired
    // since the last handshake. This is the ONLY place ``g_buf_to_fence``
    // is mutated based on completion-handler output, so it's where the
    // map state catches up with finished GPU work.
    drain_pending_fence_cleanups();

    // Wait on all upstream fences that produced our inputs (dedup).
    std::unordered_set<MTL::Fence*> waited;
    for (auto* in : g_curr_inputs) {
        auto it = g_buf_to_fence.find(in);
        if (it != g_buf_to_fence.end()) {
            MTL::Fence* f = it->second;
            if (waited.insert(f).second) {
                g_enc->waitForFence(f);
            }
        }
    }

    // Register our outputs as produced by our fence.
    for (auto* out : g_curr_outputs) {
        g_buf_to_fence[out] = g_curr_fence;
    }

    // Always signal our fence so downstream encoders can wait on it.
    g_enc->updateFence(g_curr_fence);

    // Enqueue a cleanup task for when the command buffer completes.
    // The handler runs on Metal's callback queue (off-thread); it does
    // NOT touch ``g_buf_to_fence`` directly — it just appends to the
    // pending queue, and the next ``encode_fence_handshake`` on the
    // main thread will run the actual erases (via
    // ``drain_pending_fence_cleanups`` above). The encoder's
    // updateFence call above retains the fence; the block captures it
    // as a raw pointer for the post-completion lookup. We don't
    // retain/release here because the amalgamated metal-cpp header
    // only forward-declares MTL::Fence; the encoder/cmd buffer handle
    // the lifetime in practice.
    MTL::Fence* fence_ref = g_curr_fence;
    auto outputs_copy = std::move(g_curr_outputs);
    g_cmd->addCompletedHandler(^(MTL::CommandBuffer*) {
        MutexGuard lk(g_pending_cleanups_mu);
        g_pending_fence_cleanups.emplace_back(fence_ref, std::move(outputs_copy));
    });

    // Reset for next encoder.
    g_curr_fence = nullptr;
    g_curr_inputs.clear();
    g_curr_outputs.clear();
}

static MTL::Device* get_device() {
    if (!g_device) {
        g_device = MTL::CreateSystemDefaultDevice();
        if (!g_device) throw std::runtime_error("No Metal device available");
        g_device->retain();
    }
    return g_device;
}

static MTL::CommandQueue* get_queue() {
    if (!g_queue) g_queue = get_device()->newCommandQueue();
    return g_queue;
}

// ---------------------------------------------------------------------------
// Buffer caches
// ---------------------------------------------------------------------------

static std::unordered_map<uintptr_t, MTL::Buffer*> g_input_cache;
static std::unordered_map<uintptr_t, MTL::Buffer*> g_output_alias_map;
static std::unordered_map<size_t, std::vector<MTL::Buffer*>> g_output_pool;
static std::unordered_map<size_t, std::vector<MTL::Buffer*>> g_input_pool;
static std::vector<MTL::Buffer*> g_pending_outputs;
static std::unordered_map<uint64_t, MTL::Buffer*> g_meta_cache;

// MLX-style numpy → Metal: copy data into a Metal-owned buffer instead of
// using newBufferWithBytesNoCopy. NoCopy wraps the numpy page directly,
// forcing per-dispatch CPU↔GPU cache coherence; the stall scales with
// buffer size (~2.6ms/dispatch at 4MB on M5 Max). A Metal-owned shared
// buffer decouples the layout. See MLX's nd_array_to_mlx → mx::array::init:
// set_data(malloc(nbytes)); std::copy.
//
// The (ptr → buf) cache is only valid within a single eval() boundary —
// across evals, Python's allocator can reuse addresses for unrelated
// arrays, so a stale cache hit would feed wrong data to the kernel. On
// eval() we return cached buffers to g_input_pool and clear the map.
// Within an eval, repeat lookups (same weight passed to 288 chained
// dispatches) reuse the same buffer with no recopy.
static MTL::Buffer* cache_wrap_input(void* ptr, size_t nbytes) {
    uintptr_t key = (uintptr_t)ptr;
    auto alias_it = g_output_alias_map.find(key);
    if (alias_it != g_output_alias_map.end()) return alias_it->second;
    auto it = g_input_cache.find(key);
    if (it != g_input_cache.end()) return it->second;
    size_t alloc_size = std::max(nbytes, (size_t)1);
    MTL::Buffer* buf = nullptr;
    auto& pool = g_input_pool[alloc_size];
    if (!pool.empty()) {
        buf = pool.back();
        pool.pop_back();
    } else {
        buf = get_device()->newBuffer(alloc_size, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
    }
    if (nbytes > 0) memcpy(buf->contents(), ptr, nbytes);
    g_input_cache[key] = buf;
    return buf;
}

static MTL::Buffer* pool_alloc_output(size_t nbytes) {
    nbytes = std::max(nbytes, (size_t)1);
    auto& pool = g_output_pool[nbytes];
    if (!pool.empty()) {
        auto* buf = pool.back();
        pool.pop_back();
        return buf;
    }
    return get_device()->newBuffer(nbytes, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
}

static MTL::Buffer* get_meta_buf_int32(const std::vector<int>& vals, int param_idx) {
    uint64_t key = (uint64_t)param_idx * 1000003ULL + 0x5318ULL;
    for (int v : vals) key = key * 31 + (uint64_t)v;
    auto it = g_meta_cache.find(key);
    if (it != g_meta_cache.end()) return it->second;
    size_t nbytes = std::max(vals.size() * 4, (size_t)4);
    auto* buf = get_device()->newBuffer(nbytes, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
    auto* p = (int32_t*)buf->contents();
    for (size_t i = 0; i < vals.size(); i++) p[i] = vals[i];
    g_meta_cache[key] = buf;
    return buf;
}

static MTL::Buffer* get_meta_buf_int64(const std::vector<int64_t>& vals, int param_idx) {
    uint64_t key = (uint64_t)param_idx * 1000003ULL + 0xC0DEULL;
    for (int64_t v : vals) key = key * 31 + (uint64_t)v;
    auto it = g_meta_cache.find(key);
    if (it != g_meta_cache.end()) return it->second;
    size_t nbytes = std::max(vals.size() * 8, (size_t)8);
    auto* buf = get_device()->newBuffer(nbytes, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
    auto* p = (int64_t*)buf->contents();
    for (size_t i = 0; i < vals.size(); i++) p[i] = vals[i];
    g_meta_cache[key] = buf;
    return buf;
}

static MTL::Buffer* get_ndim_buf(int ndim) {
    uint64_t key = (uint64_t)ndim + 0xFFFF0000ULL;
    auto it = g_meta_cache.find(key);
    if (it != g_meta_cache.end()) return it->second;
    auto* buf = get_device()->newBuffer(4, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
    *(int32_t*)buf->contents() = ndim;
    g_meta_cache[key] = buf;
    return buf;
}

// ---------------------------------------------------------------------------
// Lazy queue — MLX-style graph nodes
// ---------------------------------------------------------------------------
//
// queue_launch() records an op without touching the Metal encoder.
// eval_queue() walks the queue, resolves buffer handles to MTL::Buffer*,
// encodes everything in one tight C++ loop with auto-commit, then waits.
// Per-call Python cost: ~1 µs (one FFI call) vs ~7 µs for eager launch.

struct QueuedOp {
    MTL::ComputePipelineState* pipeline;
    // Inputs: index into g_lazy_buffers (for lazy handles) or raw MTL::Buffer*
    // (for eager/weight buffers). Discriminated by is_lazy flags.
    std::vector<int> input_handles;       // -1 = use input_bufs[i]
    std::vector<MTL::Buffer*> input_bufs; // non-lazy (weight) buffers
    int output_handle;                     // index into g_lazy_buffers
    std::vector<std::tuple<int,int,int>> plan; // (kind, param_idx, buf_idx)
    MTL::Size grid;
    MTL::Size tg;
    int smem_bytes;
};

static std::vector<QueuedOp> g_lazy_queue;
static std::vector<MTL::Buffer*> g_lazy_buffers;  // handle → MTL::Buffer*
static std::vector<size_t> g_lazy_buf_sizes;       // for pool return
// Per-handle refcount: parallel to g_lazy_buffers. New entries start
// at 1 (the Python ``_MetalStorage`` that wraps the handle owns one
// reference); ``release_handle`` decrements; when it hits 0 the slot
// is marked for return to the buffer pool. ``eval`` is what actually
// returns released buffers to the pool — that ensures the GPU has
// finished any in-flight work that bound the buffer before it can
// be handed to a new allocation.
//
// HANDLES ARE STABLE FOR THE LIFE OF THE STORAGE. A ``QuarkTensor``
// captured before an ``eval()`` and read after stays valid as long as
// the Python wrapper is alive, because we never reuse handle indices
// while the slot is referenced. Earlier the dispatcher resized
// ``g_lazy_buffers`` back to ``g_pin_count`` on every eval, which
// silently invalidated every transient handle Python still held —
// the fence machinery couldn't help because the bug was at the
// allocator-bookkeeping layer, not the GPU.
static std::vector<int> g_lazy_buf_refcount;
// Free list of handle indices whose buffers were returned to
// ``g_output_pool`` by the eval-sweep and whose refcount is 0. New
// allocations consume from this list before extending the table.
// Without this the table grows by ~thousands of entries per frame
// in long-running consumers (Biome's hot loop) — every eval-sweep
// nulls out the freed slot and ``lazy_alloc_output`` always
// ``push_back``s a fresh index, so after a few hundred frames the
// table is millions deep, the per-eval sweep walks all of it, and
// the dispatcher eventually wedges silently.
static std::vector<int> g_lazy_buf_free_list;
// ``g_pin_count`` is kept for back-compat with diagnostics that read
// it; allocator decisions are now refcount-driven.
static int g_pin_count = 0;

static int lazy_alloc_output(size_t nbytes) {
    auto* buf = pool_alloc_output(nbytes);
    // Refcount starts at 0; the Python ``_MetalStorage`` that wraps
    // the returned handle calls ``retain_handle`` from its
    // ``__init__`` so the count tracks live Python wrappers exactly.
    // Safe because nothing sweeps the slot before Python gets a chance
    // to wrap it (the eval sweep only fires on explicit ``eval()``).
    if (!g_lazy_buf_free_list.empty()) {
        int handle = g_lazy_buf_free_list.back();
        g_lazy_buf_free_list.pop_back();
        g_lazy_buffers[handle] = buf;
        g_lazy_buf_sizes[handle] = nbytes;
        g_lazy_buf_refcount[handle] = 0;
        return handle;
    }
    int handle = (int)g_lazy_buffers.size();
    g_lazy_buffers.push_back(buf);
    g_lazy_buf_sizes.push_back(nbytes);
    g_lazy_buf_refcount.push_back(0);
    return handle;
}

// ---------------------------------------------------------------------------
// Compile / pipeline cache (capsule-equivalent: nanobind opaque pointer)
// ---------------------------------------------------------------------------

struct PipelineHandle {
    MTL::ComputePipelineState* ps;
    ~PipelineHandle() { if (ps) ps->release(); }
};

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

NB_MODULE(_metal_dispatch, m) {
    nb::class_<PipelineHandle>(m, "Pipeline");

    // probe()
    m.def("probe", []() {
        auto* dev = get_device();
        nb::dict d;
        d["name"] = std::string(dev->name()->utf8String());
        d["architecture"] = std::string(dev->architecture()->name()->utf8String());
        d["max_threadgroup_memory"] = (long)dev->maxThreadgroupMemoryLength();
        d["max_threads_per_threadgroup"] = (long)dev->maxThreadsPerThreadgroup().width;
        d["supports_metal4"] = dev->supportsFamily(MTL::GPUFamilyMetal4);
        d["supports_apple10"] = dev->supportsFamily(MTL::GPUFamilyApple10);
        return d;
    });

    // compile(source, name, lang_version) -> Pipeline
    m.def("compile", [](const std::string& source, const std::string& name, int lang_version) {
        auto* dev = get_device();
        auto* opts = MTL::CompileOptions::alloc()->init();
        opts->setLanguageVersion((MTL::LanguageVersion)lang_version);
        // Match MLX: disable fast-math so the compiler generates
        // IEEE-correct paths that Apple's shader optimizer handles better
        // for NAX cooperative-tensor patterns.
        opts->setFastMathEnabled(false);
        NS::Error* err = nullptr;
        auto* lib = dev->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), opts, &err);
        opts->release();
        if (!lib) {
            std::string msg = err ? err->localizedDescription()->utf8String() : "unknown";
            throw std::runtime_error("Compile failed for '" + name + "': " + msg);
        }
        auto* fn = lib->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
        lib->release();
        if (!fn) throw std::runtime_error("Function '" + name + "' not found");
        err = nullptr;
        auto* pipeline = dev->newComputePipelineState(fn, &err);
        fn->release();
        if (!pipeline) {
            std::string msg = err ? err->localizedDescription()->utf8String() : "unknown";
            throw std::runtime_error("Pipeline failed: " + msg);
        }
        return new PipelineHandle{pipeline};
    }, nb::rv_policy::take_ownership);

    // pin_buffer(ptr, nbytes) -> (handle, content_ptr)
    // Allocates a persistent Metal pool buffer and copies host data
    // into it (or leaves it uninitialized when ptr == 0). Returns a
    // (handle, ptr) pair: the handle indexes into ``g_lazy_buffers``
    // for zero-copy use as a queue_launch input; the ptr is the raw
    // ``MTLBuffer->contents()`` for CPU reads after eval.
    //
    // Pinned buffers (those allocated through this function) survive
    // ``eval_queue`` — they are intended for weights loaded once and
    // reused across many forward passes.
    m.def("pin_buffer", [](uintptr_t data_ptr, size_t nbytes) -> std::tuple<int, uintptr_t> {
        size_t alloc = std::max(nbytes, (size_t)1);
        auto* buf = get_device()->newBuffer(alloc, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
        if (data_ptr != 0 && nbytes > 0) {
            memcpy(buf->contents(), (void*)data_ptr, nbytes);
        }
        int handle = (int)g_lazy_buffers.size();
        g_lazy_buffers.push_back(buf);
        g_lazy_buf_sizes.push_back(alloc);
        // Refcount starts at 0: the Python ``_MetalStorage`` that
        // wraps the returned handle bumps it via ``retain_handle``.
        g_lazy_buf_refcount.push_back(0);
        // Mark all buffers up to here as pinned (kept for diagnostics).
        g_pin_count = (int)g_lazy_buffers.size();
        return {handle, (uintptr_t)buf->contents()};
    });

    // retain_handle(handle) / release_handle(handle) — Python
    // ``_MetalStorage`` calls these on creation / destruction. The
    // refcount tracks the number of live Python wrappers pointing at
    // the handle, *not* the number of Metal-side users. ``pin_buffer``
    // and ``lazy_alloc_output`` start the count at 1 because they hand
    // the handle directly to a fresh ``_MetalStorage`` whose
    // ``__init__`` skips the explicit retain. Any other code path that
    // builds a new ``_MetalStorage`` around an existing handle (e.g.
    // ``queue_launch_ir`` rewrapping its output, or future strided
    // views that re-wrap rather than retain the existing storage)
    // MUST call ``retain_handle`` first or the slot will be swept
    // while the original wrapper is still alive.
    m.def("retain_handle", [](int handle) {
        if (handle < 0) return;
        if ((size_t)handle >= g_lazy_buf_refcount.size()) return;
        g_lazy_buf_refcount[handle]++;
    });
    m.def("release_handle", [](int handle) {
        if (handle < 0) return;
        if ((size_t)handle >= g_lazy_buf_refcount.size()) return;
        if (g_lazy_buf_refcount[handle] <= 0) return;
        g_lazy_buf_refcount[handle]--;
    });

    // launch(pipeline, inputs, output_nbytes, output_shapes, scalars,
    //        plan, grid, threadgroup, smem_bytes) -> list[int]
    //
    // Inputs and scalars come in as nb::ndarray — buffer protocol gives
    // ptr/nbytes/shape/stride/ndim in O(1), no Python list construction
    // and no nested-vector marshalling on the FFI boundary. Mirrors how
    // mx::array carries its metadata across MLX's Python binding.
    using NDArr = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;
    m.def("launch", [](
            PipelineHandle* ph,
            const std::vector<NDArr>& inputs,
            const std::vector<int>& input_handles,
            const std::vector<size_t>& output_nbytes,
            const std::vector<std::vector<int>>& output_shapes,
            const std::vector<NDArr>& scalars,
            const std::vector<std::tuple<int, int, int>>& plan,
            std::tuple<size_t, size_t, size_t> grid,
            std::tuple<size_t, size_t, size_t> tg,
            int smem_bytes,
            const std::vector<int>& provided_output_handles = {}) {

        auto* pipeline = ph->ps;
        size_t n_inputs = inputs.size();
        // input_handles is a parallel vector: handle ≥ 0 means use the
        // pinned buffer at g_lazy_buffers[handle] (zero copy); handle == -1
        // means the input is an ndarray to memcpy via cache_wrap_input.
        std::vector<MTL::Buffer*> input_bufs(n_inputs);
        for (size_t i = 0; i < n_inputs; i++) {
            int h = input_handles[i];
            if (h >= 0) {
                input_bufs[i] = g_lazy_buffers[h];
            } else {
                input_bufs[i] = cache_wrap_input(const_cast<void*>(inputs[i].data()),
                                                  inputs[i].nbytes());
            }
            track_input(input_bufs[i]);
        }

        // Outputs: by default allocate through lazy_alloc_output so they
        // get g_lazy_buffers handles — Python can wrap them as QuarkTensor
        // and pass the handle straight to subsequent queue_launch /
        // launch calls (zero copy).
        //
        // ``provided_output_handles[i] >= 0`` overrides this and uses the
        // caller's pre-allocated pinned buffer at that handle as the
        // kernel's output binding. The kernel writes directly into the
        // caller's buffer, so role="out" tensors that hold persistent
        // state across eval boundaries (e.g. KV caches) can survive the
        // pool recycle that fires at end-of-eval.
        size_t n_outputs = output_nbytes.size();
        std::vector<MTL::Buffer*> output_bufs(n_outputs);
        std::vector<int> output_handles(n_outputs);
        bool have_provided_outs = !provided_output_handles.empty();
        for (size_t i = 0; i < n_outputs; i++) {
            int provided = (have_provided_outs && i < provided_output_handles.size())
                ? provided_output_handles[i] : -1;
            if (provided >= 0) {
                output_handles[i] = provided;
                output_bufs[i] = g_lazy_buffers[provided];
            } else {
                output_handles[i] = lazy_alloc_output(output_nbytes[i]);
                output_bufs[i] = g_lazy_buffers[output_handles[i]];
            }
            g_output_alias_map[(uintptr_t)output_bufs[i]->contents()] = output_bufs[i];
            track_output(output_bufs[i]);
        }

        size_t n_scalars = scalars.size();
        std::vector<MTL::Buffer*> scalar_bufs(n_scalars);
        for (size_t i = 0; i < n_scalars; i++)
            scalar_bufs[i] = cache_wrap_input(const_cast<void*>(scalars[i].data()),
                                               scalars[i].nbytes());

        bool fresh_encoder = false;
        if (!g_cmd) {
            g_cmd = get_queue()->commandBuffer();
            g_cmd->retain();
        }
        if (!g_enc) {
            g_enc = g_cmd->computeCommandEncoder();
            g_enc->retain();
            fresh_encoder = true;
        }

        // RAW + WAW barrier: any input OR output buffer that matches a
        // recently-produced output of a prior dispatch needs a fence.
        // Output-buffer is included to catch in-place-mutating kernels
        // where the plan tags the buffer as output (kind=1) but the
        // kernel actually reads from it before writing.
        //
        // Fresh encoder: when auto-commit fires we close the encoder
        // and start a new one. Untracked buffers do NOT get implicit
        // memory visibility across command-buffer boundaries — without
        // an explicit barrier on the first dispatch of the new encoder,
        // any read of a buffer that was just written by the previous
        // command buffer sees stale data. Force a barrier on the first
        // dispatch of a fresh encoder whenever there are pending writes
        // from the previous encoder.
        bool needs_barrier = fresh_encoder && !g_pending_outputs.empty();
        if (!needs_barrier) {
            for (auto& [kind, param_idx, buf_idx] : plan) {
                MTL::Buffer* b = nullptr;
                if (kind == 0) {
                    b = input_bufs[param_idx];
                } else if (kind == 1) {
                    if ((size_t)param_idx < n_outputs) b = output_bufs[param_idx];
                }
                if (b == nullptr) continue;
                for (auto* po : g_pending_outputs) {
                    if (po == b) { needs_barrier = true; break; }
                }
                if (needs_barrier) break;
            }
        }
        if (needs_barrier) g_enc->memoryBarrier(MTL::BarrierScopeBuffers);

        g_enc->setComputePipelineState(pipeline);
        if (smem_bytes > 0) g_enc->setThreadgroupMemoryLength(smem_bytes, 0);

        // Plan walk. For shape/strides/ndim slots, pull metadata from
        // the input ndarray (param_idx < n_inputs) or output_shapes
        // (param_idx >= n_inputs, contiguous strides assumed).
        std::vector<int> shape_scratch;
        std::vector<int64_t> stride_scratch;
        for (auto& [kind, param_idx, buf_idx] : plan) {
            MTL::Buffer* buf = nullptr;
            switch (kind) {
                case 0: buf = input_bufs[param_idx]; break;
                case 1: buf = output_bufs[param_idx]; break;
                case 2: buf = scalar_bufs[param_idx]; break;
                case 3: {  // shape
                    if ((size_t)param_idx < n_inputs) {
                        auto& a = inputs[param_idx];
                        shape_scratch.resize(a.ndim());
                        for (size_t k = 0; k < a.ndim(); k++)
                            shape_scratch[k] = (int)a.shape(k);
                        buf = get_meta_buf_int32(shape_scratch, param_idx);
                    } else {
                        buf = get_meta_buf_int32(output_shapes[param_idx - n_inputs],
                                                  param_idx);
                    }
                    break;
                }
                case 4: {  // strides (in elements)
                    if ((size_t)param_idx < n_inputs) {
                        auto& a = inputs[param_idx];
                        stride_scratch.resize(a.ndim());
                        for (size_t k = 0; k < a.ndim(); k++)
                            stride_scratch[k] = a.stride(k);
                        buf = get_meta_buf_int64(stride_scratch, param_idx);
                    } else {
                        auto& sh = output_shapes[param_idx - n_inputs];
                        stride_scratch.resize(sh.size());
                        int64_t s = 1;
                        for (int k = (int)sh.size() - 1; k >= 0; k--) {
                            stride_scratch[k] = s;
                            s *= sh[k];
                        }
                        buf = get_meta_buf_int64(stride_scratch, param_idx);
                    }
                    break;
                }
                case 5: {  // ndim
                    int nd = (size_t)param_idx < n_inputs
                                 ? (int)inputs[param_idx].ndim()
                                 : (int)output_shapes[param_idx - n_inputs].size();
                    buf = get_ndim_buf(nd);
                    break;
                }
            }
            g_enc->setBuffer(buf, 0, buf_idx);
        }

        {
            auto gs = MTL::Size(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid));
            auto ts = MTL::Size(std::get<0>(tg), std::get<1>(tg), std::get<2>(tg));
            // Use dispatchThreadgroups when grid is an exact multiple of tg
            // (matches MLX's dispatch path). dispatchThreads can introduce
            // non-uniform threadgroups at edges which may affect scheduling.
            bool exact = (gs.width % ts.width == 0) && (gs.height % ts.height == 0) && (gs.depth % ts.depth == 0);
            if (exact) {
                MTL::Size tg_count(gs.width / ts.width, gs.height / ts.height, gs.depth / ts.depth);
                g_enc->dispatchThreadgroups(tg_count, ts);
            } else {
                g_enc->dispatchThreads(gs, ts);
            }
        }

        // Track this op's outputs so a subsequent launch() / queue_launch
        // that reads them will insert a memoryBarrier. Without this,
        // a launch output read by a queue_launch (or another launch)
        // bypasses the RAW guard — the eager path was previously the
        // only writer that didn't register its outputs, so a Linear
        // GEMM (queue_launch) → SiLU (launch) → Linear (queue_launch)
        // chain raced silently for any layer past the first command
        // buffer's worth of dispatches.
        for (size_t i = 0; i < n_outputs; i++) {
            g_pending_outputs.push_back(output_bufs[i]);
        }

        // Auto-commit fires every g_max_ops. Submitting partway through
        // a chain lets the GPU start the first batch while Python is
        // still encoding the next.
        g_buffer_ops++;
        if (g_buffer_ops >= g_max_ops) {
            encode_fence_handshake();
            g_enc->endEncoding(); g_enc->release(); g_enc = nullptr;
            g_cmd->commit();
            g_committed_uncompleted.push_back(g_cmd);
            g_cmd = nullptr;
            g_buffer_ops = 0;
        }

        // Return parallel (handle, ptr) lists so callers can wrap each
        // output as a QuarkTensor with _MetalStorage.
        std::vector<int> handle_result(n_outputs);
        std::vector<uintptr_t> ptr_result(n_outputs);
        for (size_t i = 0; i < n_outputs; i++) {
            handle_result[i] = output_handles[i];
            ptr_result[i] = (uintptr_t)output_bufs[i]->contents();
        }
        return std::make_tuple(handle_result, ptr_result);
    },
    nb::arg("pipeline"),
    nb::arg("inputs"),
    nb::arg("input_handles"),
    nb::arg("output_nbytes"),
    nb::arg("output_shapes"),
    nb::arg("scalars"),
    nb::arg("plan"),
    nb::arg("grid"),
    nb::arg("tg"),
    nb::arg("smem_bytes"),
    nb::arg("provided_output_handles") = std::vector<int>{}
    );

    m.def("eval", []() {
        if (g_buffer_ops == 0 && !g_cmd && g_committed_uncompleted.empty()) return;
        if (g_enc) {
            encode_fence_handshake();
            g_enc->endEncoding(); g_enc->release(); g_enc = nullptr;
        }
        if (g_cmd) {
            g_cmd->commit();
            g_committed_uncompleted.push_back(g_cmd);
            g_cmd = nullptr;
            g_buffer_ops = 0;
        }
        if (!g_committed_uncompleted.empty()) {
            g_committed_uncompleted.back()->waitUntilCompleted();
            for (auto* cb : g_committed_uncompleted) cb->release();
            g_committed_uncompleted.clear();
        }
        // g_pending_outputs is the cross-dispatch RAW barrier set; its
        // buffer ownership is already accounted for by g_lazy_buffers /
        // g_lazy_buf_refcount. Just CLEAR it on eval — do NOT push the
        // buffers into g_output_pool here. Doing so would pool buffers
        // backing live, refcounted handles (e.g. KV-cache state passed
        // via ``provided_output_handles``); pool_alloc_output would then
        // hand those same MTL::Buffer*s out to a *different* handle's
        // kernel write, silently corrupting the persistent state and
        // producing the multi-frame compounding drift the verify rig
        // surfaced (cos 0.999 → 0.49 over 5 frames).
        g_pending_outputs.clear();
        g_output_alias_map.clear();
        for (auto& [k, buf] : g_input_cache)
            g_input_pool[buf->length()].push_back(buf);
        g_input_cache.clear();
        // Refcount-driven sweep: any slot whose refcount has dropped
        // to 0 gets its buffer pooled and the slot becomes nullptr.
        // We do NOT shrink ``g_lazy_buffers`` — handles are stable for
        // the life of the storage so a Python QuarkTensor captured
        // before this eval and read afterwards still resolves to its
        // original buffer (refcount > 0 prevents the sweep here).
        for (size_t i = 0; i < g_lazy_buf_refcount.size(); i++) {
            if (g_lazy_buf_refcount[i] == 0 && g_lazy_buffers[i] != nullptr) {
                g_output_pool[g_lazy_buf_sizes[i]].push_back(g_lazy_buffers[i]);
                g_lazy_buffers[i] = nullptr;
                // Mark this slot reusable. ``lazy_alloc_output`` pops
                // from the free list before extending the table, so
                // the handle space stays bounded even on multi-thousand-
                // frame hot loops. The slot's contents (refcount, size)
                // get overwritten when the handle is reused.
                g_lazy_buf_free_list.push_back((int)i);
            }
        }
    });

    m.def("has_pending", []() {
        return g_buffer_ops > 0 || g_cmd != nullptr || !g_committed_uncompleted.empty();
    });

    // commit_no_wait — commit the current command buffer to the GPU
    // queue and return WITHOUT calling waitUntilCompleted. The next
    // ``eval()`` (or any operation that triggers a sync) drains the
    // pending committed buffer and does the deferred state cleanup.
    //
    // Use case: an iteration's tail GPU work whose output is only
    // read by the *next* iteration's first kernel (so the cross-
    // encoder fence handshake handles ordering — no host
    // synchronization needed). The waipoint world model's commit /
    // ``cache_write`` is the canonical caller: the K_cache it writes
    // is read only by the next frame's ``kv_cache_update`` /
    // ``owl_attn``, both of which fence-wait on the producer
    // automatically. By skipping the wait we let the commit GPU
    // work overlap with (a) the ANE TAEHV decoder running in a
    // worker thread and (b) the main thread's next-frame Python
    // dispatch — same trick MLX uses with its
    // ``cache_write`` lazy-eval pattern.
    //
    // Safety contract:
    //   * The encoder is closed via the fence handshake, so any
    //     buffer written here is registered in ``g_buf_to_fence``;
    //     the next encoder that reads it will ``waitForFence``.
    //   * ``g_pending_outputs`` IS cleared (the cross-encoder
    //     ordering is now via fences, not the in-encoder barrier).
    //   * Buffer-pool sweeps and ``g_input_cache`` recycling are
    //     DEFERRED to the next ``eval()`` — must not happen here
    //     because the GPU may still be using those buffers.
    //   * Any caller that subsequently reads the produced data on
    //     the host (``ctypes.memmove`` from ``data_ptr``, etc.)
    //     MUST first ``eval()`` to wait for completion. The
    //     fence machinery only handles kernel-to-kernel ordering,
    //     not kernel-to-host.
    m.def("commit_no_wait", []() {
        if (g_buffer_ops == 0 && !g_cmd) return;
        if (g_enc) {
            encode_fence_handshake();
            g_enc->endEncoding(); g_enc->release(); g_enc = nullptr;
        }
        if (g_cmd) {
            g_cmd->commit();
            g_committed_uncompleted.push_back(g_cmd);
            g_cmd = nullptr;
            g_buffer_ops = 0;
        }
        // Encoder closed → cross-dispatch RAW barriers no longer apply
        // (we use fences from here on). g_input_cache / g_lazy_buffers
        // refcounts intentionally NOT touched: GPU may still be
        // reading those.
        g_pending_outputs.clear();
        g_output_alias_map.clear();
    });

    m.def("set_max_ops", [](int n) { g_max_ops = n; });

    // -----------------------------------------------------------------
    // Lazy queue: MLX-style graph nodes
    // -----------------------------------------------------------------

    // queue_launch(pipeline, input_handles, input_ptrs, input_nbytes,
    //              out_nbytes, plan, grid, tg, smem) -> (output_handle, output_ptr)
    //
    // Records one op. Input handles ≥ 0 are lazy (index into g_lazy_buffers).
    // Input handles == -1 are eager: wrapped from input_ptrs[i]/input_nbytes[i].
    // Output buffer is pool-allocated immediately (stable ptr for views).
    // Returns (handle, ptr) so Python can wrap them in a QuarkTensor with
    // shape/dtype metadata.
    m.def("queue_launch", [](
            PipelineHandle* ph,
            const std::vector<int>& input_handles,
            const std::vector<uintptr_t>& input_ptrs,
            const std::vector<size_t>& input_nbytes,
            size_t out_nbytes,
            const std::vector<std::tuple<int,int,int>>& plan,
            std::tuple<size_t,size_t,size_t> grid,
            std::tuple<size_t,size_t,size_t> tg,
            int smem_bytes,
            int provided_output_handle = -1) -> std::tuple<int, uintptr_t> {

        // Encode this op INTO THE SAME eager encoder so it's serialized
        // with surrounding eager dispatches in Python order. Previously
        // queue_launch deferred the op into g_lazy_queue and ran it AFTER
        // all pending eager ops, which silently reordered any interleaved
        // pattern: e.g. ``owl_attn`` (queue_launch) followed by
        // ``out_proj`` (eager m8n8k8) had out_proj run first against
        // owl_attn's uninitialized output buffer → all-zero output and
        // garbage frames. Encoding directly preserves order; the caller's
        // ``with quark.lazy()`` block already amortizes commit overhead
        // via the eager auto-commit (g_max_ops) inside ``launch()``.

        // Allocate / pin the output buffer up-front so we have a stable
        // (handle, ptr) to return synchronously to Python.
        int out_handle;
        if (provided_output_handle >= 0) {
            out_handle = provided_output_handle;
        } else {
            out_handle = lazy_alloc_output(out_nbytes);
        }
        MTL::Buffer* out_buf = g_lazy_buffers[out_handle];
        g_output_alias_map[(uintptr_t)out_buf->contents()] = out_buf;
        track_output(out_buf);

        // Resolve input buffers — either pinned (handle ≥ 0 → g_lazy_buffers)
        // or eager-wrapped (memcpy from host).
        size_t n_inputs = input_handles.size();
        std::vector<MTL::Buffer*> input_bufs(n_inputs);
        for (size_t i = 0; i < n_inputs; i++) {
            if (input_handles[i] >= 0) {
                input_bufs[i] = g_lazy_buffers[input_handles[i]];
            } else {
                input_bufs[i] = cache_wrap_input((void*)input_ptrs[i], input_nbytes[i]);
            }
            track_input(input_bufs[i]);
        }

        bool fresh_encoder = false;
        if (!g_cmd) {
            g_cmd = get_queue()->commandBuffer();
            g_cmd->retain();
        }
        if (!g_enc) {
            g_enc = g_cmd->computeCommandEncoder();
            g_enc->retain();
            fresh_encoder = true;
        }

        // RAW + WAW barrier: any input OR the output buffer that is a
        // recently-produced output of a prior dispatch must be fenced
        // before this op runs. Output-buffer is included to catch
        // in-place-mutating kernels (e.g. apply_q_rope) where the
        // binding plan tags the buffer as kind=1 (output) but the MSL
        // kernel reads from it before writing.
        //
        // Fresh encoder: same rationale as in launch() — untracked
        // buffers don't sync across command-buffer boundaries, so
        // force a barrier on the first dispatch of a new encoder
        // whenever there are pending writes from the previous one.
        bool needs_barrier = fresh_encoder && !g_pending_outputs.empty();
        if (!needs_barrier) {
            for (auto& [kind, param_idx, buf_idx] : plan) {
                MTL::Buffer* b = nullptr;
                if (kind == 0 && (size_t)param_idx < n_inputs) {
                    b = input_bufs[param_idx];
                } else if (kind == 1) {
                    b = out_buf;
                }
                if (b == nullptr) continue;
                for (auto* po : g_pending_outputs) {
                    if (po == b) { needs_barrier = true; break; }
                }
                if (needs_barrier) break;
            }
        }
        if (needs_barrier) g_enc->memoryBarrier(MTL::BarrierScopeBuffers);

        g_enc->setComputePipelineState(ph->ps);
        if (smem_bytes > 0) g_enc->setThreadgroupMemoryLength(smem_bytes, 0);

        // Bind buffers via the plan: kind=0 input, kind=1 output.
        for (auto& [kind, param_idx, buf_idx] : plan) {
            MTL::Buffer* buf = nullptr;
            if (kind == 0) {
                buf = input_bufs[param_idx];
            } else if (kind == 1) {
                buf = out_buf;
            }
            if (buf) g_enc->setBuffer(buf, 0, buf_idx);
        }

        {
            auto gs = MTL::Size(std::get<0>(grid), std::get<1>(grid), std::get<2>(grid));
            auto ts = MTL::Size(std::get<0>(tg), std::get<1>(tg), std::get<2>(tg));
            bool exact = (gs.width % ts.width == 0)
                && (gs.height % ts.height == 0)
                && (gs.depth % ts.depth == 0);
            if (exact) {
                MTL::Size tg_count(gs.width / ts.width, gs.height / ts.height, gs.depth / ts.depth);
                g_enc->dispatchThreadgroups(tg_count, ts);
            } else {
                g_enc->dispatchThreads(gs, ts);
            }
        }

        // Track this op's output as recently-written so a subsequent
        // launch() that reads it will insert a barrier.
        g_pending_outputs.push_back(out_buf);

        g_buffer_ops++;
        if (g_buffer_ops >= g_max_ops) {
            encode_fence_handshake();
            g_enc->endEncoding(); g_enc->release(); g_enc = nullptr;
            g_cmd->commit();
            g_committed_uncompleted.push_back(g_cmd);
            g_cmd = nullptr;
            g_buffer_ops = 0;
        }

        uintptr_t out_ptr = (uintptr_t)out_buf->contents();
        return std::make_tuple(out_handle, out_ptr);
    },
    nb::arg("pipeline"),
    nb::arg("input_handles"),
    nb::arg("input_ptrs"),
    nb::arg("input_nbytes"),
    nb::arg("out_nbytes"),
    nb::arg("plan"),
    nb::arg("grid"),
    nb::arg("tg"),
    nb::arg("smem_bytes"),
    nb::arg("provided_output_handle") = -1
    );

    // eval_queue() — encode all queued ops, auto-commit, wait.
    // Single tight C++ loop — no Python between encoder calls.
    m.def("eval_queue", []() {
        if (g_lazy_queue.empty()) return;

        // First, flush any pending eager ops.
        if (g_buffer_ops > 0 || g_cmd) {
            if (g_enc) {
                encode_fence_handshake();
                g_enc->endEncoding(); g_enc->release(); g_enc = nullptr;
            }
            if (g_cmd) {
                g_cmd->commit();
                g_committed_uncompleted.push_back(g_cmd);
                g_cmd = nullptr;
                g_buffer_ops = 0;
            }
        }

        MTL::CommandBuffer* cmd = nullptr;
        MTL::ComputeCommandEncoder* enc = nullptr;
        int ops = 0;

        auto soft_commit = [&]() {
            if (enc) { enc->endEncoding(); enc->release(); enc = nullptr; }
            if (cmd) {
                cmd->commit();
                g_committed_uncompleted.push_back(cmd);
                cmd = nullptr;
            }
            ops = 0;
        };

        // Insert a memoryBarrier BEFORE every dispatch so the previous
        // op's writes are visible to this op's reads. Conservative —
        // we don't track RAW dependencies, just always fence. The
        // alternative (no barrier) silently corrupts data when one op's
        // output is the next op's input. The eager launch() path has
        // the equivalent guard; the lazy queue was missing it entirely.
        bool has_prev_dispatch = false;
        bool debug = std::getenv("QUARK_EVAL_DEBUG") != nullptr;
        int op_idx = 0;
        for (auto& op : g_lazy_queue) {
            if (!cmd) { cmd = get_queue()->commandBuffer(); cmd->retain(); }
            if (!enc) {
                enc = cmd->computeCommandEncoder(); enc->retain();
                has_prev_dispatch = false;
            }

            if (has_prev_dispatch) {
                enc->memoryBarrier(MTL::BarrierScopeBuffers);
            }

            if (debug) {
                fprintf(stderr, "[eval_queue] op %d: out_handle=%d (buf=%p) inputs=[",
                        op_idx, op.output_handle,
                        (void*)g_lazy_buffers[op.output_handle]);
                for (size_t pi = 0; pi < op.input_handles.size(); pi++) {
                    int h = op.input_handles[pi];
                    if (h >= 0) fprintf(stderr, "%d:%p ", h, (void*)g_lazy_buffers[h]);
                    else fprintf(stderr, "raw:%p ", (void*)op.input_bufs[pi]);
                }
                fprintf(stderr, "] g_lazy_buffers.size=%zu\n", g_lazy_buffers.size());
            }
            op_idx++;

            enc->setComputePipelineState(op.pipeline);
            if (op.smem_bytes > 0) enc->setThreadgroupMemoryLength(op.smem_bytes, 0);

            // Bind buffers from the plan.
            for (auto& [kind, param_idx, buf_idx] : op.plan) {
                MTL::Buffer* buf = nullptr;
                if (kind == 0) {  // input
                    if (op.input_handles[param_idx] >= 0)
                        buf = g_lazy_buffers[op.input_handles[param_idx]];
                    else
                        buf = op.input_bufs[param_idx];
                } else if (kind == 1) {  // output
                    buf = g_lazy_buffers[op.output_handle];
                }
                if (buf) enc->setBuffer(buf, 0, buf_idx);
            }

            // Match the eager launch() path: prefer dispatchThreadgroups
            // when the grid is an exact multiple of the threadgroup size.
            // ``dispatchThreads`` introduces non-uniform threadgroups at
            // edges which some kernels (notably NAX matmul2d) don't
            // tolerate cleanly.
            {
                bool exact = (op.grid.width % op.tg.width == 0)
                    && (op.grid.height % op.tg.height == 0)
                    && (op.grid.depth % op.tg.depth == 0);
                if (exact) {
                    MTL::Size tg_count(
                        op.grid.width / op.tg.width,
                        op.grid.height / op.tg.height,
                        op.grid.depth / op.tg.depth);
                    enc->dispatchThreadgroups(tg_count, op.tg);
                } else {
                    enc->dispatchThreads(op.grid, op.tg);
                }
            }
            has_prev_dispatch = true;

            ops++;
            if (ops >= g_max_ops) soft_commit();
        }
        soft_commit();

        // Wait for all.
        if (!g_committed_uncompleted.empty()) {
            g_committed_uncompleted.back()->waitUntilCompleted();
            for (auto* cb : g_committed_uncompleted) cb->release();
            g_committed_uncompleted.clear();
        }

        // Refcount-driven sweep: pool only slots whose Python refcount
        // is already 0. Active QuarkTensors keep their handle valid
        // across this eval (see the rationale on g_lazy_buf_refcount).
        for (size_t i = 0; i < g_lazy_buf_refcount.size(); i++) {
            if (g_lazy_buf_refcount[i] == 0 && g_lazy_buffers[i] != nullptr) {
                g_output_pool[g_lazy_buf_sizes[i]].push_back(g_lazy_buffers[i]);
                g_lazy_buffers[i] = nullptr;
            }
        }
        g_lazy_queue.clear();

        // Also drain eager caches. ``g_pending_outputs`` is the
        // cross-dispatch RAW barrier set — its buffer ownership is
        // already covered by g_lazy_buffers/g_lazy_buf_refcount, so
        // we just reset the set without re-pooling its members. (See
        // the matching note on the eval() path: pooling persistent
        // buffers from here is what produced the multi-frame compound
        // drift.)
        g_pending_outputs.clear();
        g_output_alias_map.clear();
        for (auto& [k, buf] : g_input_cache)
            g_input_pool[buf->length()].push_back(buf);
        g_input_cache.clear();
    });

    m.def("has_lazy_pending", []() {
        return !g_lazy_queue.empty();
    });

    // Internal allocator-bookkeeping snapshot for leak diagnostics. Cheap to
    // call (just reads counters); intended for repro scripts that need to
    // see the dispatcher's hidden state across hundreds of frames.
    m.def("stats", []() {
        nb::dict d;
        d["lazy_buffers"] = (long)g_lazy_buffers.size();
        long live = 0, freed = 0;
        for (size_t i = 0; i < g_lazy_buffers.size(); i++) {
            if (g_lazy_buffers[i] != nullptr) live++;
            else freed++;
        }
        d["lazy_buffers_live"] = live;
        d["lazy_buffers_nulled"] = freed;
        d["lazy_buffers_free_list"] = (long)g_lazy_buf_free_list.size();
        d["lazy_queue"] = (long)g_lazy_queue.size();
        d["pin_count"] = (long)g_pin_count;
        // ``g_buf_to_fence`` is single-threaded (only encode_fence_handshake
        // touches it). ``g_pending_fence_cleanups`` is the cross-thread queue.
        d["buf_to_fence"] = (long)g_buf_to_fence.size();
        {
            MutexGuard lk(g_pending_cleanups_mu);
            d["pending_fence_cleanups"] = (long)g_pending_fence_cleanups.size();
        }
        d["input_cache"] = (long)g_input_cache.size();
        d["output_alias_map"] = (long)g_output_alias_map.size();
        long input_pool_total = 0;
        for (auto& [k, v] : g_input_pool) input_pool_total += (long)v.size();
        d["input_pool_total"] = input_pool_total;
        d["input_pool_buckets"] = (long)g_input_pool.size();
        long output_pool_total = 0;
        for (auto& [k, v] : g_output_pool) output_pool_total += (long)v.size();
        d["output_pool_total"] = output_pool_total;
        d["output_pool_buckets"] = (long)g_output_pool.size();
        long refcount_sum = 0, refcount_max = 0;
        for (int rc : g_lazy_buf_refcount) {
            refcount_sum += rc;
            if (rc > refcount_max) refcount_max = rc;
        }
        d["refcount_sum"] = refcount_sum;
        d["refcount_max"] = refcount_max;
        return d;
    });

}
