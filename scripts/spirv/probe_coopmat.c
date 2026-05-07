// Vulkan capability probe for the SPIR-V backend.
//
// Used by PORTABILITY_PLAN §3.1 to characterise a target device before
// committing to a SpirVLowerer config. Run on each distinct target chip
// and append the captured output to ``probe_coopmat.md``.
//
//     cc -O2 -Wall -o probe_coopmat probe_coopmat.c -lvulkan
//     ./probe_coopmat
//
// Dumps every capability the SPIR-V lowerer / driver / mma_registry
// branches on:
//
//   * VkPhysicalDeviceProperties (deviceName, vendorID, apiVersion)
//   * VkPhysicalDeviceSubgroupProperties (subgroupSize, supportedStages,
//     supportedOperations)
//   * VkPhysicalDeviceCooperativeMatrixPropertiesKHR (M/N/K + dtype tuples)
//   * VkPhysicalDeviceShaderBfloat16FeaturesKHR (BFloat16Type,
//     BFloat16DotProduct, BFloat16CooperativeMatrix)
//   * VkPhysicalDeviceShaderAtomicFloat[2]FeaturesEXT (which atomic
//     adds are native vs need scalarisation legalisation)
//   * Selected limits (maxComputeWorkGroupInvocations,
//     maxPushConstantsSize, maxComputeSharedMemorySize)
//
// Cross-reference the printed output against ``quark.ir.mma_registry``,
// the legalize.py rewrites, and the SubgroupSize policy in
// PORTABILITY_PLAN §3.5. If a cap that the lowerer assumes is missing
// (e.g. atomic_float not exposed) — flag it before §3.2.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_ENABLE_BETA_EXTENSIONS 1
#include <vulkan/vulkan.h>

static const char *type_name(VkComponentTypeKHR t) {
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
        case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bf16";
        default: return "?";
    }
}
static const char *scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR:      return "device";
        case VK_SCOPE_WORKGROUP_KHR:   return "workgroup";
        case VK_SCOPE_SUBGROUP_KHR:    return "subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR:return "queueFamily";
        default: return "?";
    }
}

static void print_subgroup_stages(VkShaderStageFlags f) {
    if (f & VK_SHADER_STAGE_COMPUTE_BIT)  printf(" Compute");
    if (f & VK_SHADER_STAGE_VERTEX_BIT)   printf(" Vertex");
    if (f & VK_SHADER_STAGE_FRAGMENT_BIT) printf(" Fragment");
    if (f & VK_SHADER_STAGE_GEOMETRY_BIT) printf(" Geometry");
    if (f & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) printf(" TessCtrl");
    if (f & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) printf(" TessEval");
}

static void print_subgroup_ops(VkSubgroupFeatureFlags f) {
    if (f & VK_SUBGROUP_FEATURE_BASIC_BIT)            printf(" basic");
    if (f & VK_SUBGROUP_FEATURE_VOTE_BIT)             printf(" vote");
    if (f & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)       printf(" arithmetic");
    if (f & VK_SUBGROUP_FEATURE_BALLOT_BIT)           printf(" ballot");
    if (f & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)          printf(" shuffle");
    if (f & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) printf(" shuffleRel");
    if (f & VK_SUBGROUP_FEATURE_CLUSTERED_BIT)        printf(" clustered");
    if (f & VK_SUBGROUP_FEATURE_QUAD_BIT)             printf(" quad");
}

int main(void) {
    // Request Vulkan 1.3 — Properties2 / Features2 are core since 1.1
    // but some loaders / drivers' chain handling is broken if the
    // application doesn't advertise a high-enough API version. Asking
    // for 1.3 makes the loader dispatch every chained query through
    // the modern path.
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VkInstance instance;
    if (vkCreateInstance(&ici, NULL, &instance) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed\n"); return 1;
    }
    uint32_t n_phys = 0;
    vkEnumeratePhysicalDevices(instance, &n_phys, NULL);
    VkPhysicalDevice *phys = calloc(n_phys, sizeof(*phys));
    vkEnumeratePhysicalDevices(instance, &n_phys, phys);

    // Load Properties2 / Features2 via the loader explicitly — some
    // Mesa / proprietary driver combos don't honour chained sType
    // queries through the core static-link path.
    PFN_vkGetPhysicalDeviceProperties2 fn_props2 =
        (PFN_vkGetPhysicalDeviceProperties2)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
    PFN_vkGetPhysicalDeviceFeatures2 fn_feat2 =
        (PFN_vkGetPhysicalDeviceFeatures2)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2");
    if (!fn_props2 || !fn_feat2) {
        fprintf(stderr, "Failed to load vkGetPhysicalDevice{Properties,Features}2\n");
        return 3;
    }
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fn_coopmat =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

    for (uint32_t i = 0; i < n_phys; i++) {
        // ── 1. Identity ──
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys[i], &props);
        printf("=== Device %u: %s ===\n", i, props.deviceName);
        printf("  vendor=0x%04x  device=0x%04x  type=%u  apiVersion=%u.%u.%u\n",
            props.vendorID, props.deviceID, props.deviceType,
            VK_API_VERSION_MAJOR(props.apiVersion),
            VK_API_VERSION_MINOR(props.apiVersion),
            VK_API_VERSION_PATCH(props.apiVersion));

        // ── 2. Selected compute limits ──
        printf("  limits.maxComputeWorkGroupInvocations = %u\n",
            props.limits.maxComputeWorkGroupInvocations);
        printf("  limits.maxComputeSharedMemorySize     = %u  (%u KiB)\n",
            props.limits.maxComputeSharedMemorySize,
            props.limits.maxComputeSharedMemorySize >> 10);
        printf("  limits.maxPushConstantsSize           = %u\n",
            props.limits.maxPushConstantsSize);
        printf("  limits.maxComputeWorkGroupSize        = (%u, %u, %u)\n",
            props.limits.maxComputeWorkGroupSize[0],
            props.limits.maxComputeWorkGroupSize[1],
            props.limits.maxComputeWorkGroupSize[2]);

        // ── 3. Subgroup ──
        VkPhysicalDeviceSubgroupProperties subgroup = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 props2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &subgroup };
        fn_props2(phys[i], &props2);
        printf("  subgroup.subgroupSize       = %u\n", subgroup.subgroupSize);
        printf("  subgroup.supportedStages   ="); print_subgroup_stages(subgroup.supportedStages); printf("\n");
        printf("  subgroup.supportedOperations ="); print_subgroup_ops(subgroup.supportedOperations); printf("\n");

        // ── 4. BFloat16 features ──
        VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16_feat = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR };
        VkPhysicalDeviceFeatures2 feat2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &bf16_feat };
        fn_feat2(phys[i], &feat2);
        printf("  bf16.shaderBFloat16Type             = %s\n", bf16_feat.shaderBFloat16Type ? "yes" : "no");
        printf("  bf16.shaderBFloat16DotProduct       = %s\n", bf16_feat.shaderBFloat16DotProduct ? "yes" : "no");
        printf("  bf16.shaderBFloat16CooperativeMatrix= %s\n", bf16_feat.shaderBFloat16CooperativeMatrix ? "yes" : "no");

        // ── 5. Atomic float / float2 ──
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af1 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT };
        VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
            .pNext = &af1 };
        VkPhysicalDeviceFeatures2 feat2_atomic = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &af2 };
        fn_feat2(phys[i], &feat2_atomic);
        printf("  atomic.f32Add(buf)   = %s    f32Add(smem) = %s\n",
            af1.shaderBufferFloat32AtomicAdd ? "yes" : "no",
            af1.shaderSharedFloat32AtomicAdd ? "yes" : "no");
        printf("  atomic.f16Add(buf)   = %s    f16Add(smem) = %s\n",
            af2.shaderBufferFloat16AtomicAdd ? "yes" : "no",
            af2.shaderSharedFloat16AtomicAdd ? "yes" : "no");
        printf("  atomic.f16Min(buf)   = %s    f16Min(smem) = %s\n",
            af2.shaderBufferFloat16AtomicMinMax ? "yes" : "no",
            af2.shaderSharedFloat16AtomicMinMax ? "yes" : "no");

        // ── 6. Cooperative-matrix shapes (KHR) ──
        if (fn_coopmat) {
            uint32_t n_props = 0;
            VkResult r = fn_coopmat(phys[i], &n_props, NULL);
            if (r != VK_SUCCESS) {
                printf("  coopmat: vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR returned %d\n", r);
            } else {
                printf("  coopmat: %u shapes\n", n_props);
                VkCooperativeMatrixPropertiesKHR *p = calloc(n_props, sizeof(*p));
                for (uint32_t j = 0; j < n_props; j++)
                    p[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                fn_coopmat(phys[i], &n_props, p);
                printf("    %3s %3s %3s  %5s %5s %5s %5s  %-9s\n", "M", "N", "K", "A", "B", "C", "R", "scope");
                for (uint32_t j = 0; j < n_props; j++) {
                    printf("    %3u %3u %3u  %5s %5s %5s %5s  %-9s%s\n",
                        p[j].MSize, p[j].NSize, p[j].KSize,
                        type_name(p[j].AType), type_name(p[j].BType),
                        type_name(p[j].CType), type_name(p[j].ResultType),
                        scope_name(p[j].scope),
                        p[j].saturatingAccumulation ? "  sat" : "");
                }
                free(p);
            }
        } else {
            printf("  coopmat: VK_KHR_cooperative_matrix not loadable\n");
        }
        printf("\n");
    }
    free(phys);
    vkDestroyInstance(instance, NULL);
    return 0;
}
