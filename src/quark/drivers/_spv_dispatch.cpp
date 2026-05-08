/**
 * _spv_dispatch — nanobind-based Vulkan compute dispatch (SPIR-V backend).
 *
 * Linux-only sibling to ``_metal_dispatch.cpp``. Provides Python →
 * VkInstance / VkPhysicalDevice / VkDevice / VkQueue calls for the
 * Quark SPIR-V backend. PORTABILITY_PLAN §3.1 deliverable.
 *
 * Surface:
 *
 *     enumerate_devices() / probe(device_index)
 *         Identity-only summary + full capability dict per device.
 *
 *     bind_device(device_index) -> None
 *         Choose the active VkPhysicalDevice + create the VkDevice +
 *         compute queue + command pool. Idempotent; second call with
 *         a different index re-creates the device-side state.
 *
 *     allocate_buffer(nbytes) -> (handle, mapped_ptr)
 *         Host-visible coherent storage buffer. ``handle`` is a u64
 *         opaque token Python uses with ``launch``. ``mapped_ptr``
 *         is an address Python's ``ctypes.memmove`` can write into
 *         directly for upload / read from for download.
 *
 *     compile(spirv_bytes, entry, n_buffers, push_constants_size)
 *         -> compile_handle
 *         Build a VkShaderModule + VkDescriptorSetLayout +
 *         VkPipelineLayout + VkPipeline. ``n_buffers`` is the number
 *         of storage-buffer bindings the kernel reads/writes (slot
 *         layout is sequential 0..n_buffers-1). ``push_constants_size``
 *         is how many bytes of scalar args the kernel expects via
 *         ``layout(push_constant)``.
 *
 *     launch(compile_handle, grid_xyz, buffer_handles, push_bytes)
 *         -> None
 *         Record a vkCmdDispatch into the persistent command buffer.
 *         Synchronous-style today: the buffer is committed + waited
 *         on inside ``launch`` itself (no accumulation yet — that's
 *         a follow-up perf optim once a real workload exists).
 *
 *     sync() -> None
 *         Wait for any pending GPU work. No-op in today's eager-
 *         submit shape; lands functionality once accumulation does.
 *
 * The accumulation idiom that mirrors ``_metal_dispatch.cpp``'s
 * MTLCommandBuffer batching (commit on ops threshold / output read /
 * explicit sync) is a perf optim, not a correctness need — an eager
 * submit-per-launch shape is correct, just slower at high dispatch
 * rates. Land alongside the §3.7 v1 perf number that proves the
 * accumulation pays off.
 *
 * Linker: needs ``-lvulkan``. Vulkan headers come from
 * ``libvulkan-dev`` (apt) / equivalent. No SDK install required —
 * loader resolves ICDs at runtime.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#define VK_ENABLE_BETA_EXTENSIONS 1
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Shared instance-level state.
// ---------------------------------------------------------------------------
//
// The instance is created lazily on first use and reused across
// probe() / enumerate_devices() / future compile() calls. The same
// pattern ``_metal_dispatch.cpp`` uses for ``MTL::Device`` / queue
// singletons. Single-threaded today (every entry point is called
// under the GIL); a per-instance mutex will land alongside the first
// off-thread completion handler in the launch path.
// ---------------------------------------------------------------------------

namespace {

// Per-allocation buffer record. The mapped pointer is held for the
// lifetime of the allocation — Vulkan's host-visible coherent memory
// guarantees writes are visible to the GPU without an explicit flush.
struct BufferAlloc {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t nbytes = 0;
};

// Per-compiled-pipeline record. Kept as a single value rather than
// chasing pointers so the launch path's hot loop stays branch-free.
//
// ``cached_set`` / ``cached_buffers`` form a one-entry LRU descriptor-
// set cache: when the next ``launch_internal`` is invoked with the
// same buffer handles as the previous call (extremely common — same
// kernel, same input/output buffers), we skip the
// ``vkAllocateDescriptorSets`` + ``vkUpdateDescriptorSets`` +
// ``vkFreeDescriptorSets`` round-trip. On Battlemage that round-trip
// is ~50–80μs of the launch's ~160μs wall time.
struct CompiledPipeline {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t n_buffers = 0;
    uint32_t push_size = 0;

    VkDescriptorSet cached_set = VK_NULL_HANDLE;
    std::vector<uint64_t> cached_buffers;
};

struct Globals {
    // Instance-level (created lazily on first probe / device call).
    VkInstance instance = VK_NULL_HANDLE;
    bool init_attempted = false;
    std::string init_error;

    // Device-level (created on bind_device(device_index); recreated
    // on a subsequent bind to a different index).
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = ~0u;
    int bound_index = -1;

    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    // Descriptor pool — one set per dispatch today, refilled per
    // sync(). Sized generously enough to absorb a typical Waypoint
    // frame without re-allocating; resize on demand if a real
    // workload exceeds.
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    // Cached memory-type indices for "host-visible coherent storage"
    // (used by every alloc today) and "device-local storage" (the
    // path the steady-state kernel will switch to once we have a
    // staging-buffer loop).
    uint32_t mem_host_visible = ~0u;
    uint32_t mem_device_local = ~0u;

    // u64 → record tables. We hand Python opaque integers so it
    // doesn't have to learn nanobind capsules; the C side resolves
    // back to the record on every call. Indices are dense + start
    // from 1 (0 is reserved for "no handle").
    std::vector<BufferAlloc> buffers;
    std::vector<CompiledPipeline> pipelines;
};

Globals& globals() {
    static Globals g;
    return g;
}

void check(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "%s failed (VkResult=%d)", where, static_cast<int>(r));
        throw std::runtime_error(buf);
    }
}

void ensure_instance() {
    Globals& g = globals();
    if (g.instance != VK_NULL_HANDLE) return;
    if (g.init_attempted) {
        // Prior attempt failed; surface the same error rather than
        // re-attempting (would re-fail with the same root cause).
        throw std::runtime_error(g.init_error);
    }
    g.init_attempted = true;

    // Pin api version 1.3 so chained sType queries (Properties2 /
    // Features2 with attached BFloat16 / AtomicFloat / coopmat structs)
    // dispatch through the modern path. Some Mesa / loader combos drop
    // chained sType structures when the application advertises
    // sub-1.3 through the static dispatcher — same footgun the
    // standalone probe sidesteps via VkApplicationInfo.apiVersion.
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "quark";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkResult r = vkCreateInstance(&ici, nullptr, &g.instance);
    if (r != VK_SUCCESS) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "vkCreateInstance failed (VkResult=%d). "
                      "Is libvulkan + an ICD installed?", static_cast<int>(r));
        g.init_error = buf;
        throw std::runtime_error(g.init_error);
    }
}

std::vector<VkPhysicalDevice> physical_devices() {
    ensure_instance();
    VkInstance instance = globals().instance;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devices(n);
    if (n > 0) vkEnumeratePhysicalDevices(instance, &n, devices.data());
    return devices;
}

// ---------------------------------------------------------------------------
// Encoding helpers — pack Vulkan enums / type tuples into Python-friendly
// strings so the wrapper doesn't need to depend on Vulkan headers.
// ---------------------------------------------------------------------------

const char* component_type_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "s8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "s16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "s32";
        case VK_COMPONENT_TYPE_SINT64_KHR:  return "s64";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "u32";
        case VK_COMPONENT_TYPE_UINT64_KHR:  return "u64";
        case VK_COMPONENT_TYPE_BFLOAT16_KHR:return "bf16";
        default: return "?";
    }
}

const char* scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:       return "device";
        case VK_SCOPE_WORKGROUP_KHR:    return "workgroup";
        case VK_SCOPE_SUBGROUP_KHR:     return "subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "queueFamily";
        default: return "?";
    }
}

// ---------------------------------------------------------------------------
// Device probe.
// ---------------------------------------------------------------------------

nb::dict probe_device(VkInstance instance, VkPhysicalDevice phys) {
    nb::dict out;

    // Basic identity + selected limits.
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    out["device_name"] = std::string(props.deviceName);
    out["vendor_id"] = props.vendorID;
    out["device_id"] = props.deviceID;
    out["device_type"] = static_cast<uint32_t>(props.deviceType);
    out["api_version"] = props.apiVersion;
    out["max_compute_workgroup_invocations"] =
        props.limits.maxComputeWorkGroupInvocations;
    out["max_compute_shared_memory_size"] = props.limits.maxComputeSharedMemorySize;
    out["max_push_constants_size"] = props.limits.maxPushConstantsSize;
    nb::tuple wg_size = nb::make_tuple(
        props.limits.maxComputeWorkGroupSize[0],
        props.limits.maxComputeWorkGroupSize[1],
        props.limits.maxComputeWorkGroupSize[2]);
    out["max_compute_workgroup_size"] = wg_size;

    // Properties2 + Features2 chain — explicit fn-ptr load to dodge
    // the sub-1.3 dispatch quirk. Same fix as the standalone probe.
    auto fn_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
    auto fn_feat2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));

    // Subgroup props.
    if (fn_props2) {
        VkPhysicalDeviceSubgroupProperties sg{};
        sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &sg;
        fn_props2(phys, &props2);
        out["subgroup_size"] = sg.subgroupSize;
        out["subgroup_supported_stages"] = sg.supportedStages;
        out["subgroup_supported_operations"] = sg.supportedOperations;
    } else {
        out["subgroup_size"] = 0u;
        out["subgroup_supported_stages"] = 0u;
        out["subgroup_supported_operations"] = 0u;
    }

    // BFloat16 features.
    bool bf16_type = false, bf16_dot = false, bf16_coopmat = false;
    if (fn_feat2) {
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16{};
        bf16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
        VkPhysicalDeviceFeatures2 feat2{};
        feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feat2.pNext = &bf16;
        fn_feat2(phys, &feat2);
        bf16_type = bf16.shaderBFloat16Type;
        bf16_dot = bf16.shaderBFloat16DotProduct;
        bf16_coopmat = bf16.shaderBFloat16CooperativeMatrix;
    }
    out["bf16_type"] = bf16_type;
    out["bf16_dot_product"] = bf16_dot;
    out["bf16_cooperative_matrix"] = bf16_coopmat;

    // Atomic float / float2.
    bool f32_buf = false, f32_smem = false;
    bool f16_add_buf = false, f16_add_smem = false;
    bool f16_minmax_buf = false, f16_minmax_smem = false;
    if (fn_feat2) {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af1{};
        af1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2{};
        af2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
        af2.pNext = &af1;
        VkPhysicalDeviceFeatures2 feat2{};
        feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feat2.pNext = &af2;
        fn_feat2(phys, &feat2);
        f32_buf = af1.shaderBufferFloat32AtomicAdd;
        f32_smem = af1.shaderSharedFloat32AtomicAdd;
        f16_add_buf = af2.shaderBufferFloat16AtomicAdd;
        f16_add_smem = af2.shaderSharedFloat16AtomicAdd;
        f16_minmax_buf = af2.shaderBufferFloat16AtomicMinMax;
        f16_minmax_smem = af2.shaderSharedFloat16AtomicMinMax;
    }
    out["atomic_f32_add_buffer"] = f32_buf;
    out["atomic_f32_add_smem"] = f32_smem;
    out["atomic_f16_add_buffer"] = f16_add_buf;
    out["atomic_f16_add_smem"] = f16_add_smem;
    out["atomic_f16_minmax_buffer"] = f16_minmax_buf;
    out["atomic_f16_minmax_smem"] = f16_minmax_smem;

    // Cooperative-matrix shapes.
    auto fn_coopmat = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance,
                              "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    nb::list shapes;
    if (fn_coopmat) {
        uint32_t n_props = 0;
        if (fn_coopmat(phys, &n_props, nullptr) == VK_SUCCESS && n_props > 0) {
            std::vector<VkCooperativeMatrixPropertiesKHR> props_list(n_props);
            for (auto& p : props_list)
                p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            fn_coopmat(phys, &n_props, props_list.data());
            for (const auto& p : props_list) {
                nb::dict shape;
                shape["m"] = p.MSize;
                shape["n"] = p.NSize;
                shape["k"] = p.KSize;
                shape["a_dtype"] = std::string(component_type_name(p.AType));
                shape["b_dtype"] = std::string(component_type_name(p.BType));
                shape["c_dtype"] = std::string(component_type_name(p.CType));
                shape["result_dtype"] =
                    std::string(component_type_name(p.ResultType));
                shape["scope"] = std::string(scope_name(p.scope));
                shape["saturating_accumulation"] =
                    static_cast<bool>(p.saturatingAccumulation);
                shapes.append(shape);
            }
        }
    }
    out["cooperative_matrix_shapes"] = shapes;

    return out;
}

// ---------------------------------------------------------------------------
// Device binding — pick a VkPhysicalDevice and stand up its VkDevice +
// compute queue + command pool + descriptor pool + per-launch fence.
// Re-enterable: a second call with a different index tears the prior
// state down first.
// ---------------------------------------------------------------------------

uint32_t pick_compute_queue_family(VkPhysicalDevice phys) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> qf(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf.data());
    // Prefer compute-only family (often higher-priority on the GPU's
    // scheduler than the universal graphics+compute one). Fall back
    // to any family that advertises COMPUTE.
    for (uint32_t i = 0; i < n; ++i) {
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            return i;
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    }
    throw std::runtime_error("no Vulkan queue family with COMPUTE");
}

void cache_memory_type_indices(Globals& g) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
    g.mem_host_visible = ~0u;
    g.mem_device_local = ~0u;
    // Walk types in order; prefer COHERENT (no manual flush needed)
    // for the host-visible slot.
    constexpr VkMemoryPropertyFlags HV =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags pf = mp.memoryTypes[i].propertyFlags;
        if (g.mem_host_visible == ~0u && (pf & HV) == HV) {
            g.mem_host_visible = i;
        }
        if (g.mem_device_local == ~0u &&
            (pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            g.mem_device_local = i;
        }
    }
    if (g.mem_host_visible == ~0u) {
        throw std::runtime_error(
            "no HOST_VISIBLE | HOST_COHERENT memory type — required for "
            "spv_dispatch's first-cut buffer allocator");
    }
}

void teardown_device_state(Globals& g) {
    if (g.device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(g.device);
    for (auto& p : g.pipelines) {
        if (p.cached_set != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(g.device, g.desc_pool, 1, &p.cached_set);
            p.cached_set = VK_NULL_HANDLE;
        }
        if (p.pipeline) vkDestroyPipeline(g.device, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(g.device, p.layout, nullptr);
        if (p.dsl) vkDestroyDescriptorSetLayout(g.device, p.dsl, nullptr);
        if (p.module) vkDestroyShaderModule(g.device, p.module, nullptr);
    }
    g.pipelines.clear();
    for (auto& b : g.buffers) {
        if (b.mapped) vkUnmapMemory(g.device, b.memory);
        if (b.buffer) vkDestroyBuffer(g.device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(g.device, b.memory, nullptr);
    }
    g.buffers.clear();
    if (g.desc_pool) vkDestroyDescriptorPool(g.device, g.desc_pool, nullptr);
    if (g.fence) vkDestroyFence(g.device, g.fence, nullptr);
    if (g.cmd_buf) vkFreeCommandBuffers(g.device, g.cmd_pool, 1, &g.cmd_buf);
    if (g.cmd_pool) vkDestroyCommandPool(g.device, g.cmd_pool, nullptr);
    vkDestroyDevice(g.device, nullptr);
    g.device = VK_NULL_HANDLE;
    g.queue = VK_NULL_HANDLE;
    g.cmd_pool = VK_NULL_HANDLE;
    g.cmd_buf = VK_NULL_HANDLE;
    g.fence = VK_NULL_HANDLE;
    g.desc_pool = VK_NULL_HANDLE;
    g.bound_index = -1;
}

void bind_device_internal(uint32_t index) {
    Globals& g = globals();
    if (g.bound_index == static_cast<int>(index)) return;  // already there
    teardown_device_state(g);

    auto devices = physical_devices();
    if (index >= devices.size()) {
        throw std::out_of_range(
            "bind_device: device_index out of range");
    }
    g.phys = devices[index];
    g.queue_family = pick_compute_queue_family(g.phys);
    cache_memory_type_indices(g);

    // Build VkDevice with the bf16 feature chain enabled so the
    // SPIR-V we'll emit can declare `BFloat16` capability and
    // `OpCooperativeMatrixMulAddKHR` ops at bf16. Pass any feature
    // we can — the driver will silently drop ones it doesn't
    // advertise (no failure mode there) — but only the ones we
    // actually plan to use today.
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16{};
    bf16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    bf16.shaderBFloat16Type = VK_TRUE;
    bf16.shaderBFloat16CooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat{};
    coopmat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopmat.cooperativeMatrix = VK_TRUE;
    coopmat.pNext = &bf16;
    // PORTABILITY_PLAN §3.5: pin SIMD32 via subgroupSizeControl so the
    // lowerer's hardcoded ``subgroup_width = 32`` matches what the
    // driver actually dispatches at. Required for the ``OpExecutionMode
    // SubgroupSize 32`` decoration the lowerer emits to take effect.
    VkPhysicalDeviceSubgroupSizeControlFeatures sgsc{};
    sgsc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    sgsc.subgroupSizeControl = VK_TRUE;
    sgsc.computeFullSubgroups = VK_TRUE;
    sgsc.pNext = &coopmat;
    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.shaderBufferInt64Atomics = VK_FALSE;
    v12.pNext = &sgsc;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // Enable the extensions the SPIR-V backend will need. Surface
    // them via `vkEnumerateDeviceExtensionProperties` first so
    // we don't request anything the driver doesn't expose
    // (otherwise vkCreateDevice fails). Mesa Battlemage does
    // ship all these.
    std::vector<const char*> ext_names = {
        "VK_KHR_cooperative_matrix",
        "VK_KHR_shader_bfloat16",
        "VK_EXT_subgroup_size_control",
    };
    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> avail(n_ext);
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &n_ext, avail.data());
    auto has_ext = [&](const char* name) {
        for (const auto& e : avail) {
            if (std::strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };
    std::vector<const char*> enabled_ext;
    for (auto* e : ext_names) {
        if (has_ext(e)) enabled_ext.push_back(e);
    }

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(enabled_ext.size());
    dci.ppEnabledExtensionNames = enabled_ext.data();
    dci.pNext = &v12;
    check(vkCreateDevice(g.phys, &dci, nullptr, &g.device), "vkCreateDevice");

    vkGetDeviceQueue(g.device, g.queue_family, 0, &g.queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g.queue_family;
    check(vkCreateCommandPool(g.device, &cpci, nullptr, &g.cmd_pool),
          "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = g.cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(g.device, &cbai, &g.cmd_buf),
          "vkAllocateCommandBuffers");

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    check(vkCreateFence(g.device, &fci, nullptr, &g.fence), "vkCreateFence");

    // Descriptor pool — sized for ~64 unique pipeline launches
    // before reset; 8 storage-buffer slots per launch is plenty
    // for the kernels in tree (qkv_proj at 5 buffers is the upper
    // bound today). Resize on demand later.
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 64 * 8;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 64;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    check(vkCreateDescriptorPool(g.device, &dpci, nullptr, &g.desc_pool),
          "vkCreateDescriptorPool");

    g.bound_index = static_cast<int>(index);
}

void ensure_device() {
    Globals& g = globals();
    if (g.device != VK_NULL_HANDLE) return;
    auto devices = physical_devices();
    if (devices.empty()) {
        throw std::runtime_error("no Vulkan physical devices");
    }
    bind_device_internal(0);
}

// ---------------------------------------------------------------------------
// Buffer allocation. Host-visible coherent storage today — fine for
// iGPU dev (Battlemage's unified memory means HV is also DEVICE_LOCAL
// in practice). Add a staging-buffer loop + pure DEVICE_LOCAL pool
// once the §3.7 v2 sweep on discrete Arc demands it.
// ---------------------------------------------------------------------------

uint64_t allocate_buffer_internal(uint64_t nbytes) {
    ensure_device();
    Globals& g = globals();

    BufferAlloc rec{};
    rec.nbytes = static_cast<size_t>(nbytes);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = nbytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(g.device, &bci, nullptr, &rec.buffer),
          "vkCreateBuffer");

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g.device, rec.buffer, &mr);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = g.mem_host_visible;
    check(vkAllocateMemory(g.device, &mai, nullptr, &rec.memory),
          "vkAllocateMemory");
    check(vkBindBufferMemory(g.device, rec.buffer, rec.memory, 0),
          "vkBindBufferMemory");

    check(vkMapMemory(g.device, rec.memory, 0, mr.size, 0, &rec.mapped),
          "vkMapMemory");
    std::memset(rec.mapped, 0, rec.nbytes);

    g.buffers.push_back(rec);
    // Hand back a 1-based index so 0 stays reserved for "no handle".
    return static_cast<uint64_t>(g.buffers.size());
}

BufferAlloc& resolve_buffer(uint64_t handle) {
    Globals& g = globals();
    if (handle == 0 || handle > g.buffers.size()) {
        throw std::out_of_range("buffer handle out of range");
    }
    return g.buffers[static_cast<size_t>(handle) - 1];
}

// ---------------------------------------------------------------------------
// Pipeline compilation.
//
// Layout convention (locked for v1):
//   * All buffers are storage buffers, bound at sequential
//     descriptor-set #0 bindings 0..n_buffers-1.
//   * Push constants occupy a single VK_SHADER_STAGE_COMPUTE range
//     of size ``push_constants_size`` bytes (zero → no PC range).
//   * SPIR-V binary's entry point name is ``entry`` (typically "main").
//   * No specialisation constants today — the SubgroupSize pin per
//     §3.5 lands as a spec const here in a follow-up.
// ---------------------------------------------------------------------------

uint64_t compile_internal(
    const void* spirv_blob, size_t blob_nbytes,
    const std::string& entry, uint32_t n_buffers,
    uint32_t push_constants_size
) {
    ensure_device();
    Globals& g = globals();

    if (blob_nbytes % 4 != 0) {
        throw std::runtime_error(
            "compile: SPIR-V blob length must be a multiple of 4 bytes");
    }

    CompiledPipeline rec{};
    rec.n_buffers = n_buffers;
    rec.push_size = push_constants_size;

    // 1. Shader module from the SPIR-V binary.
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = blob_nbytes;
    smci.pCode = static_cast<const uint32_t*>(spirv_blob);
    check(vkCreateShaderModule(g.device, &smci, nullptr, &rec.module),
          "vkCreateShaderModule");

    // 2. Descriptor set layout — one storage-buffer binding per slot.
    std::vector<VkDescriptorSetLayoutBinding> bindings(n_buffers);
    for (uint32_t i = 0; i < n_buffers; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = n_buffers;
    dslci.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(g.device, &dslci, nullptr, &rec.dsl),
          "vkCreateDescriptorSetLayout");

    // 3. Pipeline layout — one descriptor set + maybe push constants.
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_constants_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &rec.dsl;
    plci.pushConstantRangeCount = (push_constants_size > 0) ? 1 : 0;
    plci.pPushConstantRanges = (push_constants_size > 0) ? &pcr : nullptr;
    check(vkCreatePipelineLayout(g.device, &plci, nullptr, &rec.layout),
          "vkCreatePipelineLayout");

    // 4. Compute pipeline.
    VkPipelineShaderStageCreateInfo ssci{};
    ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = rec.module;
    ssci.pName = entry.c_str();

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = ssci;
    cpci.layout = rec.layout;
    check(vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &cpci,
                                   nullptr, &rec.pipeline),
          "vkCreateComputePipelines");

    g.pipelines.push_back(rec);
    return static_cast<uint64_t>(g.pipelines.size());  // 1-based handle
}

CompiledPipeline& resolve_pipeline(uint64_t handle) {
    Globals& g = globals();
    if (handle == 0 || handle > g.pipelines.size()) {
        throw std::out_of_range("pipeline handle out of range");
    }
    return g.pipelines[static_cast<size_t>(handle) - 1];
}

// ---------------------------------------------------------------------------
// Launch — record + submit + wait. Eager today (§3.1 MVP); accumulate
// in a follow-up commit when a perf number demands it.
// ---------------------------------------------------------------------------

void launch_internal(
    uint64_t pipeline_handle,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    const std::vector<uint64_t>& buffer_handles,
    const void* push_bytes, uint32_t push_nbytes
) {
    Globals& g = globals();
    CompiledPipeline& p = resolve_pipeline(pipeline_handle);

    if (buffer_handles.size() != p.n_buffers) {
        throw std::runtime_error(
            "launch: buffer_handles count != n_buffers from compile()");
    }
    if (push_nbytes != p.push_size) {
        throw std::runtime_error(
            "launch: push_bytes size != push_constants_size from compile()");
    }

    // One-entry LRU on the descriptor set: when the same buffer
    // handles come in twice in a row (the steady-state case for any
    // bench / inference loop), reuse the previously-written set
    // verbatim. ~60μs/call on Battlemage by skipping the
    // alloc/update/free round-trip.
    bool reuse = (p.cached_set != VK_NULL_HANDLE) &&
                 (p.cached_buffers.size() == buffer_handles.size()) &&
                 (p.cached_buffers == buffer_handles);

    VkDescriptorSet ds = p.cached_set;
    if (!reuse) {
        // Free the stale cached set if any. Vulkan's
        // ``FREE_DESCRIPTOR_SET_BIT`` flag on the pool makes this
        // legal (and cheap — just returns the slot to the pool).
        if (p.cached_set != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(g.device, g.desc_pool, 1, &p.cached_set);
            p.cached_set = VK_NULL_HANDLE;
        }

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = g.desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &p.dsl;
        check(vkAllocateDescriptorSets(g.device, &dsai, &ds),
              "vkAllocateDescriptorSets");

        std::vector<VkDescriptorBufferInfo> buf_infos(p.n_buffers);
        std::vector<VkWriteDescriptorSet> writes(p.n_buffers);
        for (uint32_t i = 0; i < p.n_buffers; ++i) {
            BufferAlloc& b = resolve_buffer(buffer_handles[i]);
            buf_infos[i].buffer = b.buffer;
            buf_infos[i].offset = 0;
            buf_infos[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(g.device, p.n_buffers, writes.data(), 0, nullptr);

        p.cached_set = ds;
        p.cached_buffers = buffer_handles;
    }

    // Record + submit + wait. Eager.
    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkResetCommandBuffer(g.cmd_buf, 0), "vkResetCommandBuffer");
    check(vkBeginCommandBuffer(g.cmd_buf, &cbbi), "vkBeginCommandBuffer");
    vkCmdBindPipeline(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(g.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p.layout, 0, 1, &ds, 0, nullptr);
    if (push_nbytes > 0) {
        vkCmdPushConstants(g.cmd_buf, p.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_nbytes, push_bytes);
    }
    vkCmdDispatch(g.cmd_buf, grid_x, grid_y, grid_z);
    check(vkEndCommandBuffer(g.cmd_buf), "vkEndCommandBuffer");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd_buf;
    check(vkResetFences(g.device, 1, &g.fence), "vkResetFences");
    check(vkQueueSubmit(g.queue, 1, &si, g.fence), "vkQueueSubmit");
    check(vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");

    // The descriptor set stays cached on the pipeline for the next
    // ``launch_internal`` to reuse. It's freed when the pipeline is
    // destroyed (``free_pipeline``) or when a subsequent launch
    // arrives with a different buffer-handle tuple.
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API exposed to Python.
// ---------------------------------------------------------------------------

NB_MODULE(_spv_dispatch, m) {
    m.doc() =
        "Quark SPIR-V backend Python↔Vulkan bindings. Probe-only in this "
        "first cut; compile/launch lands incrementally per "
        "PORTABILITY_PLAN §3.1.";

    m.def("enumerate_devices", []() {
        auto devices = physical_devices();
        VkInstance instance = globals().instance;
        nb::list out;
        for (size_t i = 0; i < devices.size(); ++i) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            nb::dict d;
            d["index"] = static_cast<uint32_t>(i);
            d["device_name"] = std::string(props.deviceName);
            d["vendor_id"] = props.vendorID;
            d["device_id"] = props.deviceID;
            d["device_type"] = static_cast<uint32_t>(props.deviceType);
            d["api_version"] = props.apiVersion;
            (void)instance;  // silence unused-var if we never need instance later here
            out.append(d);
        }
        return out;
    },
    "Return identity dicts for every Vulkan physical device.");

    m.def("probe", [](uint32_t device_index) {
        auto devices = physical_devices();
        if (device_index >= devices.size()) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "device_index %u out of range (have %zu Vulkan devices)",
                device_index, devices.size());
            throw std::out_of_range(buf);
        }
        return probe_device(globals().instance, devices[device_index]);
    },
    nb::arg("device_index") = 0,
    "Return a capability dict for the Vulkan physical device at "
    "``device_index``. Same data the standalone ``scripts/spirv/"
    "probe_coopmat.c`` dumps to stdout, but parsed into Python types "
    "so ``drivers/spv.py`` can populate ``DeviceCaps`` directly.");

    m.def("bind_device", [](uint32_t device_index) {
        bind_device_internal(device_index);
    },
    nb::arg("device_index") = 0,
    "Stand up VkDevice + compute queue + command pool for the device "
    "at ``device_index``. Idempotent; re-binding to a different index "
    "tears the prior state down first.");

    m.def("allocate_buffer", [](uint64_t nbytes) {
        uint64_t handle = allocate_buffer_internal(nbytes);
        BufferAlloc& b = resolve_buffer(handle);
        return nb::make_tuple(handle, reinterpret_cast<uintptr_t>(b.mapped));
    },
    nb::arg("nbytes"),
    "Allocate a host-visible coherent storage buffer of ``nbytes`` "
    "bytes. Returns ``(handle, mapped_ptr)`` — handle is opaque, "
    "mapped_ptr is an integer address callers (typically Python's "
    "``ctypes.memmove``) write into / read from for upload / "
    "download.");

    m.def("compile", [](nb::bytes spirv,
                        const std::string& entry,
                        uint32_t n_buffers,
                        uint32_t push_constants_size) {
        return compile_internal(spirv.c_str(), spirv.size(),
                                entry, n_buffers, push_constants_size);
    },
    nb::arg("spirv"),
    nb::arg("entry") = "main",
    nb::arg("n_buffers"),
    nb::arg("push_constants_size") = 0,
    "Compile a SPIR-V binary blob to a dispatchable Vulkan compute "
    "pipeline. ``n_buffers`` storage-buffer bindings at descriptor "
    "set 0 (sequential bindings 0..n_buffers-1). "
    "``push_constants_size`` bytes of push constants if non-zero. "
    "Returns an opaque pipeline handle for ``launch``.");

    m.def("launch", [](uint64_t pipeline,
                       nb::tuple grid,
                       std::vector<uint64_t> buffers,
                       nb::bytes push_bytes) {
        if (grid.size() != 3) {
            throw std::runtime_error("launch: grid must be (x, y, z)");
        }
        uint32_t gx = nb::cast<uint32_t>(grid[0]);
        uint32_t gy = nb::cast<uint32_t>(grid[1]);
        uint32_t gz = nb::cast<uint32_t>(grid[2]);
        launch_internal(pipeline, gx, gy, gz, buffers,
                        push_bytes.c_str(),
                        static_cast<uint32_t>(push_bytes.size()));
    },
    nb::arg("pipeline"),
    nb::arg("grid"),
    nb::arg("buffers"),
    nb::arg("push_bytes") = nb::bytes(""),
    "Record + submit + wait on one vkCmdDispatch. ``grid`` is "
    "``(x, y, z)`` workgroup count. ``buffers`` is a list of buffer "
    "handles in binding-slot order. ``push_bytes`` is the raw push-"
    "constant payload (empty if push_constants_size was 0).");

    m.def("sync", []() {
        Globals& g = globals();
        if (g.device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(g.device);
        }
    },
    "Wait for all pending GPU work to complete. No-op in the eager-"
    "submit shape today (every launch already waits inline); turns "
    "into a real wait once command-buffer accumulation lands.");
}
