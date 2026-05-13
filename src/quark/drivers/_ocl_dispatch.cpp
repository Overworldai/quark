// Quark OpenCL backend — Python↔OpenCL bindings (nanobind).
//
// API surface: probe / bind_device / allocate_buffer / compile /
// launch / sync.
//
// Target: Intel iGPU/dGPU via Intel NEO OpenCL runtime (intel-compute-
// runtime) + IGC. SPIR-V binary input via cl_khr_il_program. Storage
// buffers as USM-shared allocations via cl_intel_unified_shared_memory
// (matches the iGPU's UMA model and gives us host-mapped pointers
// like Vulkan's vkMapMemory + HOST_VISIBLE_COHERENT).
//
// Phase 1 scope: minimal mirror covering the API surface the launcher
// touches today. Skipped for now (land in Phase 1.5):
//   * GPU-side clEnqueueCopyBuffer (batched submission accumulator)
//   * Kernel timings via cl_event profiling
//   * Push-constant support beyond raw kernel-arg pass-through
//
// Build: link -lOpenCL. Headers from opencl-headers package.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace nb = nanobind;

// ─────────────────────────────────────────────────────────────────────
// cl_intel_unified_shared_memory function pointers (extension; resolved
// at runtime via clGetExtensionFunctionAddressForPlatform).
// ─────────────────────────────────────────────────────────────────────

typedef void* (CL_API_CALL *clHostMemAllocINTEL_fn)(
    cl_context context,
    const cl_mem_properties_intel* properties,
    size_t size, cl_uint alignment, cl_int* errcode_ret);

typedef void* (CL_API_CALL *clSharedMemAllocINTEL_fn)(
    cl_context context,
    cl_device_id device,
    const cl_mem_properties_intel* properties,
    size_t size, cl_uint alignment, cl_int* errcode_ret);

typedef void* (CL_API_CALL *clDeviceMemAllocINTEL_fn)(
    cl_context context,
    cl_device_id device,
    const cl_mem_properties_intel* properties,
    size_t size, cl_uint alignment, cl_int* errcode_ret);

typedef cl_int (CL_API_CALL *clMemFreeINTEL_fn)(
    cl_context context, void* ptr);

typedef cl_int (CL_API_CALL *clSetKernelArgMemPointerINTEL_fn)(
    cl_kernel kernel, cl_uint arg_index, const void* arg_value);

struct UsmFns {
    clHostMemAllocINTEL_fn host_alloc = nullptr;
    clSharedMemAllocINTEL_fn shared_alloc = nullptr;
    clDeviceMemAllocINTEL_fn device_alloc = nullptr;
    clMemFreeINTEL_fn mem_free = nullptr;
    clSetKernelArgMemPointerINTEL_fn set_kernel_arg_mem_pointer = nullptr;

    bool resolve(cl_platform_id plat) {
        host_alloc = (clHostMemAllocINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
        shared_alloc = (clSharedMemAllocINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(plat, "clSharedMemAllocINTEL");
        device_alloc = (clDeviceMemAllocINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(plat, "clDeviceMemAllocINTEL");
        mem_free = (clMemFreeINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(plat, "clMemFreeINTEL");
        set_kernel_arg_mem_pointer = (clSetKernelArgMemPointerINTEL_fn)
            clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL");
        return host_alloc && shared_alloc && mem_free && set_kernel_arg_mem_pointer;
    }
};

// ─────────────────────────────────────────────────────────────────────
// Globals — one process, one bound device.
// ─────────────────────────────────────────────────────────────────────

struct BufferAlloc {
    void* usm_ptr = nullptr;   // USM-shared pointer; valid on both host + device
    size_t nbytes = 0;
};

struct CompiledKernel {
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    uint32_t n_buffers = 0;
    uint32_t push_size = 0;
    // Local workgroup size pinned at compile time (matches the SPIR-V's
    // declared LocalSize). Set via clEnqueueNDRangeKernel's local_work_size.
    size_t local_size[3] = {1, 1, 1};
};

struct Globals {
    bool initialized = false;
    std::vector<cl_platform_id> platforms;
    std::vector<cl_device_id> devices;          // flat across all platforms
    std::vector<cl_platform_id> device_plat;    // parallel: which platform each device belongs to
    cl_device_id active_device = nullptr;
    cl_platform_id active_platform = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    UsmFns usm;
    std::mutex lock;

    std::unordered_map<uint64_t, BufferAlloc> buffers;
    std::unordered_map<uint64_t, CompiledKernel> kernels;
    uint64_t next_buffer_id = 1;
    uint64_t next_kernel_id = 1;
};

static Globals& g() {
    static Globals globals;
    return globals;
}

static const char* cl_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case -1099: return "CL_INVALID_IL_INTEL (SPIR-V parse failed)";
        default: return "CL_UNKNOWN";
    }
}

static void check_cl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s failed: %s (%d)",
                      what, cl_err_str(err), err);
        throw std::runtime_error(buf);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Initialization helpers.
// ─────────────────────────────────────────────────────────────────────

static void ensure_initialized() {
    Globals& gl = g();
    if (gl.initialized) return;
    std::lock_guard<std::mutex> lk(gl.lock);
    if (gl.initialized) return;

    cl_uint n_plats = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &n_plats);
    if (err != CL_SUCCESS || n_plats == 0) {
        // No platforms — leave initialized=false; enumerate_devices()
        // will return an empty list.
        return;
    }
    gl.platforms.resize(n_plats);
    err = clGetPlatformIDs(n_plats, gl.platforms.data(), nullptr);
    if (err != CL_SUCCESS) return;

    // Walk every platform, collect all GPU devices.
    for (cl_platform_id plat : gl.platforms) {
        cl_uint n_devs = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devs) != CL_SUCCESS) continue;
        if (n_devs == 0) continue;
        std::vector<cl_device_id> devs(n_devs);
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, n_devs, devs.data(), nullptr) != CL_SUCCESS) continue;
        for (cl_device_id d : devs) {
            gl.devices.push_back(d);
            gl.device_plat.push_back(plat);
        }
    }
    gl.initialized = true;
}

static nb::dict probe_device_dict(uint32_t device_index) {
    Globals& gl = g();
    ensure_initialized();
    if (device_index >= gl.devices.size()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "device_index %u out of range (%zu OpenCL GPU devices)",
                      device_index, gl.devices.size());
        throw std::out_of_range(buf);
    }
    cl_device_id dev = gl.devices[device_index];
    cl_platform_id plat = gl.device_plat[device_index];

    nb::dict out;
    out["index"] = device_index;

    auto str_info = [&](cl_device_info info, const char* key) {
        char buf[1024];
        size_t sz = 0;
        if (clGetDeviceInfo(dev, info, sizeof(buf), buf, &sz) == CL_SUCCESS) {
            out[key] = std::string(buf, sz ? sz - 1 : 0);
        }
    };
    str_info(CL_DEVICE_NAME, "device_name");
    str_info(CL_DEVICE_VERSION, "device_version");
    str_info(CL_DRIVER_VERSION, "driver_version");
    str_info(CL_DEVICE_EXTENSIONS, "extensions");

    auto u32_info = [&](cl_device_info info, const char* key) {
        cl_uint v = 0;
        if (clGetDeviceInfo(dev, info, sizeof(v), &v, nullptr) == CL_SUCCESS) {
            out[key] = static_cast<uint32_t>(v);
        }
    };
    u32_info(CL_DEVICE_VENDOR_ID, "vendor_id");
    u32_info(CL_DEVICE_MAX_COMPUTE_UNITS, "max_compute_units");

    auto sz_info = [&](cl_device_info info, const char* key) {
        size_t v = 0;
        if (clGetDeviceInfo(dev, info, sizeof(v), &v, nullptr) == CL_SUCCESS) {
            out[key] = static_cast<uint64_t>(v);
        }
    };
    sz_info(CL_DEVICE_MAX_WORK_GROUP_SIZE, "max_compute_workgroup_invocations");
    sz_info(CL_DEVICE_LOCAL_MEM_SIZE, "max_compute_shared_memory_size");

    // Device type
    cl_device_type dtype = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr) == CL_SUCCESS) {
        // Mirror Vulkan VkPhysicalDeviceType encoding:
        //   1 = INTEGRATED_GPU, 2 = DISCRETE_GPU, 0 = OTHER
        uint32_t vk_type = 0;
        if (dtype & CL_DEVICE_TYPE_GPU) {
            cl_bool unified_host_memory = CL_FALSE;
            clGetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY,
                            sizeof(unified_host_memory),
                            &unified_host_memory, nullptr);
            vk_type = unified_host_memory ? 1u : 2u;  // INTEGRATED vs DISCRETE
        }
        out["device_type"] = vk_type;
    }

    // Device ID (Intel PCI device ID — exposed via extension if available)
    cl_uint dev_id = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_ID_INTEL, sizeof(dev_id), &dev_id, nullptr) == CL_SUCCESS) {
        out["device_id"] = static_cast<uint32_t>(dev_id);
    } else {
        out["device_id"] = 0u;
    }

    // Subgroup sizes (Intel exposes a list via CL_DEVICE_SUB_GROUP_SIZES_INTEL)
    {
        size_t sz = 0;
        clGetDeviceInfo(dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL, 0, nullptr, &sz);
        if (sz > 0) {
            std::vector<size_t> sgs(sz / sizeof(size_t));
            if (clGetDeviceInfo(dev, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sz,
                                sgs.data(), nullptr) == CL_SUCCESS) {
                nb::list lst;
                for (size_t s : sgs) lst.append(static_cast<uint32_t>(s));
                out["subgroup_sizes"] = lst;
                // Default: pick 32 if available, otherwise the max available.
                uint32_t default_sg = 32;
                bool has_32 = false;
                uint32_t max_sg = 0;
                for (size_t s : sgs) {
                    if (s == 32) has_32 = true;
                    if (s > max_sg) max_sg = (uint32_t)s;
                }
                out["subgroup_size"] = has_32 ? 32u : max_sg;
            }
        } else {
            out["subgroup_size"] = 32u;
        }
    }

    // Extension probes — surfaced as bools the Python caps layer reads.
    std::string ext;
    {
        size_t sz = 0;
        clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &sz);
        ext.resize(sz);
        clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sz, ext.data(), nullptr);
    }
    auto has = [&](const char* tok) -> bool {
        return ext.find(tok) != std::string::npos;
    };
    out["has_il_program"] = has("cl_khr_il_program");
    out["has_subgroup_matrix_mma"] = has("cl_intel_subgroup_matrix_multiply_accumulate");
    out["has_usm"] = has("cl_intel_unified_shared_memory");
    out["has_subgroups"] = has("cl_intel_subgroups");
    out["has_bf16_conversions"] = has("cl_intel_bfloat16_conversions");
    // Matched against Vulkan's bf16_cooperative_matrix probe surface so
    // caps_from_probe in the Python wrapper can stay uniform.
    out["bf16_cooperative_matrix"] = has("cl_intel_subgroup_matrix_multiply_accumulate")
                                      && has("cl_intel_bfloat16_conversions");

    out["platform"] = (uint64_t)(uintptr_t)plat;
    return out;
}

static void bind_device_internal(uint32_t device_index) {
    Globals& gl = g();
    ensure_initialized();
    if (device_index >= gl.devices.size()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "device_index %u out of range (%zu OpenCL GPU devices)",
                      device_index, gl.devices.size());
        throw std::out_of_range(buf);
    }
    std::lock_guard<std::mutex> lk(gl.lock);

    // Idempotent re-bind to the same device: skip teardown so kernels
    // + USM buffers allocated by an earlier OclDriver instance remain
    // valid. Without this, multiple OclDriver()s in the same process
    // (runtime tensor allocator + launcher driver + explicit user
    // driver) would tear each other's compiled-kernel table down
    // mid-execution.
    if (gl.ctx && gl.active_device == gl.devices[device_index]) {
        return;
    }

    // Tear down prior state if rebinding to a different device.
    if (gl.queue) { clReleaseCommandQueue(gl.queue); gl.queue = nullptr; }
    if (gl.ctx) { clReleaseContext(gl.ctx); gl.ctx = nullptr; }
    for (auto& kv : gl.kernels) {
        if (kv.second.kernel) clReleaseKernel(kv.second.kernel);
        if (kv.second.program) clReleaseProgram(kv.second.program);
    }
    gl.kernels.clear();
    if (gl.usm.mem_free) {
        for (auto& kv : gl.buffers) {
            gl.usm.mem_free(gl.ctx, kv.second.usm_ptr);
        }
    }
    gl.buffers.clear();

    gl.active_device = gl.devices[device_index];
    gl.active_platform = gl.device_plat[device_index];

    if (!gl.usm.resolve(gl.active_platform)) {
        throw std::runtime_error(
            "OCL device does not expose cl_intel_unified_shared_memory "
            "(required for quark's OCL backend). Need Intel NEO runtime "
            "with the USM extension.");
    }

    cl_int err;
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)gl.active_platform, 0
    };
    gl.ctx = clCreateContext(props, 1, &gl.active_device, nullptr, nullptr, &err);
    check_cl(err, "clCreateContext");

    cl_queue_properties qprops[] = {
        CL_QUEUE_PROPERTIES,
        (cl_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                              | CL_QUEUE_PROFILING_ENABLE),
        0
    };
    gl.queue = clCreateCommandQueueWithProperties(
        gl.ctx, gl.active_device, qprops, &err);
    check_cl(err, "clCreateCommandQueueWithProperties");
}

// ─────────────────────────────────────────────────────────────────────
// Buffer allocation (USM-shared).
// ─────────────────────────────────────────────────────────────────────

static uint64_t allocate_buffer_internal(uint64_t nbytes) {
    Globals& gl = g();
    if (!gl.ctx) {
        throw std::runtime_error(
            "OCL: bind_device() must be called before allocate_buffer()");
    }
    cl_int err = CL_SUCCESS;
    void* ptr = gl.usm.shared_alloc(gl.ctx, gl.active_device, nullptr,
                                     (size_t)nbytes, 64u /* alignment */, &err);
    check_cl(err, "clSharedMemAllocINTEL");
    if (!ptr) throw std::runtime_error("clSharedMemAllocINTEL returned NULL");

    std::lock_guard<std::mutex> lk(gl.lock);
    uint64_t h = gl.next_buffer_id++;
    BufferAlloc& b = gl.buffers[h];
    b.usm_ptr = ptr;
    b.nbytes = (size_t)nbytes;
    return h;
}

static BufferAlloc& resolve_buffer(uint64_t handle) {
    Globals& gl = g();
    auto it = gl.buffers.find(handle);
    if (it == gl.buffers.end()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "OCL: unknown buffer handle %lu",
                      (unsigned long)handle);
        throw std::runtime_error(buf);
    }
    return it->second;
}

// ─────────────────────────────────────────────────────────────────────
// SPIR-V compile.
// ─────────────────────────────────────────────────────────────────────

static uint64_t compile_internal(const void* spirv, size_t spirv_size,
                                  const std::string& entry,
                                  uint32_t n_buffers,
                                  uint32_t push_size,
                                  uint32_t subgroup_size,
                                  uint32_t local_x,
                                  uint32_t local_y,
                                  uint32_t local_z) {
    Globals& gl = g();
    if (!gl.ctx) {
        throw std::runtime_error("OCL: bind_device() must be called before compile()");
    }

    cl_int err;
    cl_program prog = clCreateProgramWithIL(gl.ctx, spirv, spirv_size, &err);
    check_cl(err, "clCreateProgramWithIL");

    // Build flags — pin the required subgroup size if requested.
    // Intel-specific flags: -cl-intel-required-sub-group-size sets the
    // SIMD width per kernel; matches Vulkan's
    // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo behavior.
    std::string flags;
    if (subgroup_size > 0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
                      "-cl-intel-required-sub-group-size %u", subgroup_size);
        flags += buf;
    }

    err = clBuildProgram(prog, 1, &gl.active_device, flags.c_str(),
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Pull the build log for the error message.
        size_t lsz = 0;
        clGetProgramBuildInfo(prog, gl.active_device,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz);
        clGetProgramBuildInfo(prog, gl.active_device,
                              CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        clReleaseProgram(prog);
        std::string msg = "clBuildProgram failed: ";
        msg += cl_err_str(err);
        msg += "\nbuild log:\n";
        msg.append(log.data(), lsz);
        throw std::runtime_error(msg);
    }

    cl_kernel kern = clCreateKernel(prog, entry.c_str(), &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        check_cl(err, "clCreateKernel");
    }

    std::lock_guard<std::mutex> lk(gl.lock);
    uint64_t h = gl.next_kernel_id++;
    CompiledKernel& ck = gl.kernels[h];
    ck.program = prog;
    ck.kernel = kern;
    ck.n_buffers = n_buffers;
    ck.push_size = push_size;
    ck.local_size[0] = local_x;
    ck.local_size[1] = local_y;
    ck.local_size[2] = local_z;
    return h;
}

static CompiledKernel& resolve_kernel(uint64_t handle) {
    Globals& gl = g();
    auto it = gl.kernels.find(handle);
    if (it == gl.kernels.end()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "OCL: unknown kernel handle %lu",
                      (unsigned long)handle);
        throw std::runtime_error(buf);
    }
    return it->second;
}

// ─────────────────────────────────────────────────────────────────────
// Launch.
// ─────────────────────────────────────────────────────────────────────

static void launch_internal(uint64_t kernel_handle,
                            uint32_t gx, uint32_t gy, uint32_t gz,
                            const std::vector<uint64_t>& buffer_handles,
                            const void* push_data, uint32_t push_size,
                            bool sync) {
    Globals& gl = g();
    if (!gl.queue) {
        throw std::runtime_error("OCL: bind_device() required before launch()");
    }
    CompiledKernel& ck = resolve_kernel(kernel_handle);

    if (buffer_handles.size() != ck.n_buffers) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "OCL launch: kernel expects %u buffers, got %zu",
                      ck.n_buffers, buffer_handles.size());
        throw std::runtime_error(buf);
    }

    // Bind buffer args (USM pointers via the Intel extension).
    for (uint32_t i = 0; i < ck.n_buffers; ++i) {
        BufferAlloc& b = resolve_buffer(buffer_handles[i]);
        cl_int err = gl.usm.set_kernel_arg_mem_pointer(ck.kernel, i, b.usm_ptr);
        check_cl(err, "clSetKernelArgMemPointerINTEL");
    }

    // Push constants — if any, pass as the trailing arg as raw bytes.
    if (push_size > 0) {
        cl_int err = clSetKernelArg(ck.kernel, ck.n_buffers,
                                     push_size, push_data);
        check_cl(err, "clSetKernelArg (push)");
    }

    // Global work size: (grid * local). OpenCL semantics are
    // (total_threads, ...) not (workgroups, ...).
    size_t global[3] = {
        (size_t)gx * ck.local_size[0],
        (size_t)gy * ck.local_size[1],
        (size_t)gz * ck.local_size[2],
    };

    cl_int err = clEnqueueNDRangeKernel(gl.queue, ck.kernel, 3,
                                          nullptr, global, ck.local_size,
                                          0, nullptr, nullptr);
    check_cl(err, "clEnqueueNDRangeKernel");

    if (sync) {
        err = clFinish(gl.queue);
        check_cl(err, "clFinish");
    }
}

static void sync_internal() {
    Globals& gl = g();
    if (gl.queue) {
        cl_int err = clFinish(gl.queue);
        check_cl(err, "clFinish");
    }
}

// ─────────────────────────────────────────────────────────────────────
// nanobind module.
// ─────────────────────────────────────────────────────────────────────

NB_MODULE(_ocl_dispatch, m) {
    m.doc() = "Quark OpenCL backend Python↔OpenCL bindings.";

    m.def("enumerate_devices", []() {
        ensure_initialized();
        Globals& gl = g();
        nb::list out;
        for (size_t i = 0; i < gl.devices.size(); ++i) {
            cl_device_id dev = gl.devices[i];
            nb::dict d;
            d["index"] = static_cast<uint32_t>(i);
            char name[256];
            size_t sz = 0;
            if (clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, &sz) == CL_SUCCESS) {
                d["device_name"] = std::string(name, sz ? sz - 1 : 0);
            }
            cl_uint vid = 0;
            clGetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(vid), &vid, nullptr);
            d["vendor_id"] = static_cast<uint32_t>(vid);
            cl_uint did = 0;
            clGetDeviceInfo(dev, CL_DEVICE_ID_INTEL, sizeof(did), &did, nullptr);
            d["device_id"] = static_cast<uint32_t>(did);
            cl_device_type dt = 0;
            clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dt), &dt, nullptr);
            cl_bool unified = CL_FALSE;
            clGetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified),
                            &unified, nullptr);
            d["device_type"] = (dt & CL_DEVICE_TYPE_GPU) ? (unified ? 1u : 2u) : 0u;
            out.append(d);
        }
        return out;
    }, "Enumerate OpenCL GPU devices across all platforms.");

    m.def("probe", &probe_device_dict, nb::arg("device_index") = 0,
          "Return capability dict for the device at ``device_index``.");

    m.def("bind_device", &bind_device_internal, nb::arg("device_index") = 0,
          "Stand up cl_context + cl_command_queue for the device.");

    m.def("allocate_buffer", [](uint64_t nbytes) {
        uint64_t h = allocate_buffer_internal(nbytes);
        BufferAlloc& b = resolve_buffer(h);
        return nb::make_tuple(h, reinterpret_cast<uintptr_t>(b.usm_ptr));
    }, nb::arg("nbytes"),
       "Allocate a USM-shared buffer of ``nbytes`` bytes. Returns "
       "``(handle, host_ptr)`` — host_ptr is a uintptr_t the caller "
       "uses with ctypes.memmove for upload/download. Same pointer is "
       "the GPU-side address (UMA).");

    m.def("compile", [](nb::bytes spirv,
                        const std::string& entry,
                        uint32_t n_buffers,
                        uint32_t push_constants_size,
                        uint32_t subgroup_size,
                        uint32_t local_x,
                        uint32_t local_y,
                        uint32_t local_z) {
        return compile_internal(spirv.c_str(), spirv.size(),
                                entry, n_buffers, push_constants_size,
                                subgroup_size, local_x, local_y, local_z);
    },
    nb::arg("spirv"),
    nb::arg("entry") = "main",
    nb::arg("n_buffers"),
    nb::arg("push_constants_size") = 0,
    nb::arg("subgroup_size") = 32,
    nb::arg("local_x") = 32, nb::arg("local_y") = 1, nb::arg("local_z") = 1,
    "Compile SPIR-V binary blob to a clCreateProgramWithIL + "
    "clBuildProgram + clCreateKernel pipeline. ``n_buffers`` USM "
    "buffer args; ``push_constants_size`` raw push-bytes arg "
    "appended after the buffers if non-zero. ``subgroup_size`` "
    "passed via -cl-intel-required-sub-group-size at build time. "
    "``local_x/y/z`` is the kernel's LocalSize (matches SPIR-V's "
    "OpExecutionMode LocalSize).");

    m.def("launch", [](uint64_t kernel,
                       nb::tuple grid,
                       std::vector<uint64_t> buffers,
                       nb::bytes push_bytes,
                       bool sync) {
        if (grid.size() != 3) {
            throw std::runtime_error("launch: grid must be (x, y, z)");
        }
        uint32_t gx = nb::cast<uint32_t>(grid[0]);
        uint32_t gy = nb::cast<uint32_t>(grid[1]);
        uint32_t gz = nb::cast<uint32_t>(grid[2]);
        launch_internal(kernel, gx, gy, gz, buffers,
                        push_bytes.c_str(),
                        static_cast<uint32_t>(push_bytes.size()),
                        sync);
    },
    nb::arg("kernel"), nb::arg("grid"), nb::arg("buffers"),
    nb::arg("push_bytes") = nb::bytes(""),
    nb::arg("sync") = true,
    "Enqueue one clEnqueueNDRangeKernel. ``grid`` is workgroup count "
    "(NOT total threads — global size = grid * local internally). "
    "``sync=False`` skips the clFinish; caller must call sync() "
    "before reading output buffers.");

    m.def("sync", &sync_internal,
          "Wait for the command queue to drain (clFinish).");
}
