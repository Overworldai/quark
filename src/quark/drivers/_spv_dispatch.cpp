/**
 * _spv_dispatch — nanobind-based Vulkan compute dispatch (SPIR-V backend).
 *
 * Linux-only sibling to ``_metal_dispatch.cpp``. Provides Python →
 * VkInstance / VkPhysicalDevice / VkDevice / VkQueue calls for the
 * Quark SPIR-V backend. PORTABILITY_PLAN §3.1 deliverable.
 *
 * Surface (kept tiny in this first cut — only ``probe`` is wired):
 *
 *     probe(device_index: int = 0) -> dict
 *         Init a VkInstance, enumerate VkPhysicalDevice, return a
 *         capability dict for the device at ``device_index``. Mirrors
 *         the standalone ``scripts/spirv/probe_coopmat.c`` probe but
 *         delivers the data into Python so ``drivers/spv.py`` can
 *         populate ``DeviceCaps`` without shelling out.
 *
 *     enumerate_devices() -> list[dict]
 *         Identity-only summary for every Vulkan physical device.
 *         Used by ``Engine.__new__`` family detection + by tests.
 *
 * The compile/launch surface (``compile``, ``launch``, ``sync``,
 * buffer-pool helpers, fence-graph wiring) lands in subsequent
 * commits per PORTABILITY_PLAN §3.1's sub-phase map. Until then
 * ``SpvDriver`` raises NotImplementedError on any call past
 * device-probe — same shape ``SpirVLowerer`` had at §3.0.
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

struct Globals {
    VkInstance instance = VK_NULL_HANDLE;
    bool init_attempted = false;
    std::string init_error;
};

Globals& globals() {
    static Globals g;
    return g;
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
}
