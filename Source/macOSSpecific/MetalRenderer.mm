#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#import "MetalRendererFunctions.h"
#include <type_traits>
#include <utility>
#include "SceneMetaLoader.h"

#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <filesystem>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <fstream>

template <typename T, typename = void>
struct has_member_linearTextures : std::false_type {};

template <typename T>
struct has_member_linearTextures<T, std::void_t<decltype(std::declval<const T&>().linearTextures)>> : std::true_type {};

// Optional PBR indices on SceneMetaMaterial (newer LevelMeta versions may include them).
template <typename T, typename = void>
struct has_member_normalTexIndex : std::false_type {};
template <typename T>
struct has_member_normalTexIndex<T, std::void_t<decltype(std::declval<const T&>().normalTexIndex)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_ormTexIndex : std::false_type {};
template <typename T>
struct has_member_ormTexIndex<T, std::void_t<decltype(std::declval<const T&>().ormTexIndex)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_roughnessTexIndex : std::false_type {};
template <typename T>
struct has_member_roughnessTexIndex<T, std::void_t<decltype(std::declval<const T&>().roughnessTexIndex)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_metallicTexIndex : std::false_type {};
template <typename T>
struct has_member_metallicTexIndex<T, std::void_t<decltype(std::declval<const T&>().metallicTexIndex)>> : std::true_type {};

// Current SceneMetaMaterial in this project uses "occlusionTexIndex".
// (Older branches/exporters sometimes used "aoTexIndex"; we intentionally avoid
// referencing it directly here to keep compilation stable across versions.)
template <typename T, typename = void>
struct has_member_occlusionTexIndex : std::false_type {};
template <typename T>
struct has_member_occlusionTexIndex<T, std::void_t<decltype(std::declval<const T&>().occlusionTexIndex)>> : std::true_type {};

namespace
{
    id<MTLDevice>               g_device          = nil;
    id<MTLCommandQueue>         g_queue           = nil;
    id<MTLComputePipelineState> g_pipelineTexture      = nil;
    id<MTLComputePipelineState> g_pipelineBloomExtract = nil;
    id<MTLComputePipelineState> g_pipelineBloomBlurH   = nil;
    id<MTLComputePipelineState> g_pipelineBloomBlurV   = nil;
    id<MTLComputePipelineState> g_pipelinePostProcess  = nil;

    constexpr uint32_t          kMaxAllTextures = 124;
    constexpr uint32_t          kAccumulatedSampleCountMax = 0x00FFFFFFu;
    constexpr uint32_t          kMetalShaderUvDebugMode = 0u; // keep in sync with UV_DEBUG_MODE in RayTrace.metal
    constexpr std::uint64_t     kInvalidAccumulationHash = std::numeric_limits<std::uint64_t>::max();

    // Накопительная текстура + sample-based state для прогрессивного рендера
    id<MTLTexture>              g_accumTexture    = nil;
    uint32_t                    g_accumulatedSampleCount = 0;
    uint32_t                    g_accumulationDispatchCount = 0;
    std::uint64_t               g_accumulationStateHash = kInvalidAccumulationHash;
    id<MTLTexture>              g_hdrTexture      = nil;
    id<MTLTexture>              g_albedoTexture   = nil;
    id<MTLTexture>              g_normalTexture   = nil;
    id<MTLTexture>              g_bloomTextureA   = nil;
    id<MTLTexture>              g_bloomTextureB   = nil;
    id<MTLTexture>              g_outTexture      = nil;
    int                         g_postTextureWidth = 0;
    int                         g_postTextureHeight = 0;

    id<MTLBuffer>               g_materialBuffer      = nil;
    id<MTLBuffer>               g_materialCountBuffer = nil;

    std::vector<id<MTLTexture>> g_baseColorTextures;
    std::vector<std::string>   g_baseColorTexturePaths; // чтобы понимать, когда нужно перезагружать
    struct CpuSampleTexture
    {
        size_t width = 0;
        size_t height = 0;
        std::vector<uint8_t> bgra;

        bool valid() const
        {
            return width > 0 && height > 0 && bgra.size() == width * height * 4u;
        }
    };
    std::unordered_map<std::string, CpuSampleTexture> g_cpuSceneTextureCache;

    id<MTLBuffer>               g_materialPBRBuffer      = nil;   // подготовка под расширенный MaterialGPU (пока не используется шейдером)
    id<MTLBuffer>               g_materialPBRCountBuffer = nil;
    std::vector<uint8_t>        g_linearTextureValid; // 1 если текстура реально загрузилась
    const SceneMetaResources   *g_loadedMetaRes         = nullptr;

    id<MTLBuffer>               g_tlasBuffer                  = nil;
    id<MTLBuffer>               g_triBuffer                   = nil;
    id<MTLBuffer>               g_meshNodeBuffer              = nil;
    id<MTLBuffer>               g_instanceBuffer              = nil;
    id<MTLBuffer>               g_nodeCountBuffer             = nil;
    id<MTLBuffer>               g_rootIndexBuffer             = nil;
    id<MTLBuffer>               g_meshNodeCountBuffer         = nil;
    id<MTLBuffer>               g_instanceCountBuffer         = nil;
    id<MTLBuffer>               g_lightBuffer                 = nil;
    id<MTLBuffer>               g_lightCountBuffer            = nil;
    id<MTLBuffer>               g_emissiveTriangleBuffer      = nil;
    id<MTLBuffer>               g_emissiveTriangleCountBuffer = nil;
    id<MTLBuffer>               g_decalBuffer                 = nil;
    id<MTLBuffer>               g_decalCountBuffer            = nil;
    id<MTLBuffer>               g_airDustVolumeBuffer         = nil;
    id<MTLBuffer>               g_airDustVolumeCountBuffer    = nil;
    id<MTLBuffer>               g_camBuffer                   = nil;
    id<MTLBuffer>               g_imgSizeBuffer               = nil;
    id<MTLBuffer>               g_sampleCountBuffer           = nil;
    id<MTLBuffer>               g_postProcessBuffer           = nil;

    std::uint64_t               g_cachedSceneRevision         = 0;
    const SceneMetaResources   *g_cachedDecalMetaRes          = nullptr;
    const SceneMetaResources   *g_cachedAirDustMetaRes        = nullptr;

    using ProfileClock = std::chrono::steady_clock;

    struct TextureFrameProfile
    {
        uint64_t callIndex = 0;
        uint32_t accumulatedSamples = 0;
        uint32_t samplesThisDispatch = 0;
        MetalAccumulationMode accumulationMode = MetalAccumulationMode::PreviewProgressive;
        uint32_t prototypeTriangles = 0;
        uint32_t totalInstances = 0;
        uint32_t tlasNodeCount = 0;
        uint32_t blasNodeCount = 0;
        double initMetalMs = 0.0;
        double ensureMaterialsMs = 0.0;
        double accumTextureMs = 0.0;
        double intermediateTexturesMs = 0.0;
        double geometryBuffersMs = 0.0;
        double smallBuffersMs = 0.0;
        double lightUploadMs = 0.0;
        double emissiveDecalMs = 0.0;
        double postParamsMs = 0.0;
        double encodeMs = 0.0;
        double waitMs = 0.0;
        double readbackMs = 0.0;
        double gpuMs = 0.0;
        double primaryDepthGpuMs = 0.0;
        double pathTraceGpuMs = 0.0;
        double bloomExtractGpuMs = 0.0;
        double blurHGpuMs = 0.0;
        double blurVGpuMs = 0.0;
        double finalCompositeGpuMs = 0.0;
        double totalMs = 0.0;
    };

    struct TextureProfileTotals
    {
        uint64_t frameCount = 0;
        double initMetalMs = 0.0;
        double ensureMaterialsMs = 0.0;
        double accumTextureMs = 0.0;
        double intermediateTexturesMs = 0.0;
        double geometryBuffersMs = 0.0;
        double smallBuffersMs = 0.0;
        double lightUploadMs = 0.0;
        double emissiveDecalMs = 0.0;
        double postParamsMs = 0.0;
        double encodeMs = 0.0;
        double waitMs = 0.0;
        double readbackMs = 0.0;
        double gpuMs = 0.0;
        double primaryDepthGpuMs = 0.0;
        double pathTraceGpuMs = 0.0;
        double bloomExtractGpuMs = 0.0;
        double blurHGpuMs = 0.0;
        double blurVGpuMs = 0.0;
        double finalCompositeGpuMs = 0.0;
        double totalMs = 0.0;
    };

    uint64_t g_textureProfileCallIndex = 0;
    TextureProfileTotals g_textureProfileTotals;

    static double ToMilliseconds(ProfileClock::duration duration)
    {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    static double SafeGpuTimeMs(id<MTLCommandBuffer> cmd)
    {
        if (!cmd)
            return 0.0;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED)
        if (@available(macOS 10.15, *))
        {
            const CFTimeInterval start = cmd.GPUStartTime;
            const CFTimeInterval end   = cmd.GPUEndTime;
            if (start > 0.0 && end >= start)
                return (end - start) * 1000.0;
        }
#endif
        return 0.0;
    }

    static void AppendProfileLine(std::ostringstream &oss,
                                  const char *name,
                                  double valueMs,
                                  double totalMs,
                                  bool showPercent = true)
    {
        oss << "    "
            << std::left  << std::setw(18) << name
            << std::right << std::setw(12) << std::fixed << std::setprecision(3) << valueMs
            << " ms";

        if (showPercent && totalMs > 0.0)
        {
            const double percent = (valueMs * 100.0) / totalMs;
            oss << "  (" << std::setw(7) << std::fixed << std::setprecision(2) << percent << "%)";
        }

        oss << "\n";
    }

    static void AppendCountLine(std::ostringstream &oss,
                                const char *name,
                                uint32_t value)
    {
        oss << "    "
            << std::left  << std::setw(18) << name
            << std::right << std::setw(12) << value
            << "\n";
    }

    static void AppendProfileText(const std::string &textBlock)
    {
        std::cout << textBlock;
        try
        {
            const std::filesystem::path outDir = std::filesystem::path("Results");
            std::filesystem::create_directories(outDir);
            std::ofstream out(outDir / "ComplexFrameStats.txt", std::ios::out | std::ios::app);
            if (out)
                out << textBlock;
        }
        catch (...)
        {
        }
    }

    static bool EnsureSharedBufferLength(id<MTLDevice> dev,
                                         id<MTLBuffer> &buffer,
                                         std::size_t requiredBytes)
    {
        if (!dev)
            return false;

        const NSUInteger targetBytes =
            static_cast<NSUInteger>(std::max<std::size_t>(requiredBytes, 1u));
        if (!buffer || [buffer length] < targetBytes)
            buffer = [dev newBufferWithLength:targetBytes options:MTLResourceStorageModeShared];

        return buffer != nil;
    }

    template <typename T>
    static bool UploadSharedValue(id<MTLDevice> dev,
                                  id<MTLBuffer> &buffer,
                                  const T &value)
    {
        if (!EnsureSharedBufferLength(dev, buffer, sizeof(T)))
            return false;

        std::memcpy([buffer contents], &value, sizeof(T));
        return true;
    }

    template <typename T>
    static bool UploadSharedVector(id<MTLDevice> dev,
                                   id<MTLBuffer> &buffer,
                                   const std::vector<T> &values,
                                   const T &fallback)
    {
        const T *src = values.empty() ? &fallback : values.data();
        const std::size_t bytes = values.empty()
                                ? sizeof(T)
                                : values.size() * sizeof(T);
        if (!EnsureSharedBufferLength(dev, buffer, bytes))
            return false;

        std::memcpy([buffer contents], src, bytes);
        return true;
    }

    static const char *AccumulationModeName(MetalAccumulationMode mode)
    {
        switch (mode)
        {
            case MetalAccumulationMode::PreviewProgressive: return "preview_progressive";
            case MetalAccumulationMode::FinalStill:         return "final_still";
        }
        return "unknown";
    }

    struct HashBuilder64
    {
        std::uint64_t value = 1469598103934665603ull;

        void addBytes(const void *data, std::size_t size)
        {
            const auto *bytes = static_cast<const std::uint8_t *>(data);
            for (std::size_t i = 0; i < size; ++i)
            {
                value ^= static_cast<std::uint64_t>(bytes[i]);
                value *= 1099511628211ull;
            }
        }

        void addU8(std::uint8_t v) { addBytes(&v, sizeof(v)); }
        void addU32(std::uint32_t v) { addBytes(&v, sizeof(v)); }
        void addU64(std::uint64_t v) { addBytes(&v, sizeof(v)); }
        void addI32(std::int32_t v) { addBytes(&v, sizeof(v)); }

        void addBool(bool v)
        {
            addU8(v ? 1u : 0u);
        }

        void addFloat(float v)
        {
            std::uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(v));
            std::memcpy(&bits, &v, sizeof(bits));
            addU32(bits);
        }

        void addVec3(const Vec3 &v)
        {
            addFloat(v.x);
            addFloat(v.y);
            addFloat(v.z);
        }

        void addString(const std::string &s)
        {
            addU64(static_cast<std::uint64_t>(s.size()));
            if (!s.empty())
                addBytes(s.data(), s.size());
        }
    };

    static void HashCameraForAccumulation(HashBuilder64 &hash,
                                          const CameraDataCPU &cameraCPU)
    {
        hash.addVec3(cameraCPU.position);
        hash.addVec3(cameraCPU.forward);
        hash.addVec3(cameraCPU.up);
        hash.addVec3(cameraCPU.right);
        hash.addFloat(cameraCPU.fovY);
        hash.addFloat(cameraCPU.nearPlane);
        hash.addFloat(cameraCPU.farPlane);
        hash.addFloat(cameraCPU.focusDistance);
        hash.addFloat(cameraCPU.aspectRatio);
    }

    static void HashLight(HashBuilder64 &hash,
                          const Light &light)
    {
        hash.addI32(static_cast<std::int32_t>(light.type));
        hash.addVec3(light.position);
        hash.addVec3(light.direction);
        hash.addVec3(light.color);
        hash.addFloat(light.intensity);
        hash.addFloat(light.radius);
        hash.addFloat(light.sourceLength);
        hash.addFloat(light.softSourceRadius);
        hash.addFloat(light.attenuationRadius);
        hash.addFloat(light.spotSize);
        hash.addFloat(light.spotBlend);
        hash.addI32(static_cast<std::int32_t>(light.areaShape));
        hash.addFloat(light.areaSizeX);
        hash.addFloat(light.areaSizeY);
    }

    static void HashMaterial(HashBuilder64 &hash,
                             const SceneMetaMaterial &material)
    {
        hash.addVec3(material.baseColor);
        hash.addVec3(material.emissionColor);
        hash.addFloat(material.emissionStrength);
        hash.addFloat(material.metallic);
        hash.addFloat(material.roughness);
        hash.addFloat(material.opacity);
        hash.addI32(material.blendMode);
        hash.addBool(material.twoSided);
        hash.addBool(material.thinEmissiveSurface);
        hash.addBool(material.emissionUseAlphaMask);
        hash.addI32(material.specialModel);
        hash.addI32(material.specialTex0Index);
        hash.addI32(material.specialTex1Index);
        hash.addFloat(material.specialScalar0);
        hash.addFloat(material.specialScalar1);
        hash.addFloat(material.specialScalar2);
        hash.addFloat(material.specialScalar3);
        hash.addFloat(material.specialScalar4);
        hash.addFloat(material.specialScalar5);
        hash.addFloat(material.decalTilingU);
        hash.addFloat(material.decalTilingV);
        hash.addFloat(material.decalOpacityPower);
        hash.addFloat(material.decalNormalIntensity);
        hash.addFloat(material.decalRoughnessBias);
        hash.addI32(material.decalOpacityTexIndex);
        hash.addBool(material.decalOpacityTexIsLinear);
        hash.addI32(material.decalDetailTexIndex);
        hash.addBool(material.decalDetailTexIsLinear);
        hash.addI32(material.baseColorTexIndex);
        hash.addI32(material.emissionTexIndex);
        hash.addI32(material.baseColorUvSet);
        hash.addI32(material.emissionUvSet);
        hash.addI32(material.normalTexIndex);
        hash.addI32(material.ormTexIndex);
        hash.addI32(material.roughnessTexIndex);
        hash.addI32(material.metallicTexIndex);
        hash.addI32(material.occlusionTexIndex);
        hash.addI32(material.normalUvSet);
        hash.addI32(material.ormUvSet);
        hash.addI32(material.roughnessUvSet);
        hash.addI32(material.metallicUvSet);
        hash.addI32(material.occlusionUvSet);
        hash.addU8(material.ormChannels.occlusion);
        hash.addU8(material.ormChannels.roughness);
        hash.addU8(material.ormChannels.metallic);
    }

    static void HashDecal(HashBuilder64 &hash,
                          const SceneMetaDecal &decal)
    {
        hash.addVec3(decal.position);
        hash.addVec3(decal.axisX);
        hash.addVec3(decal.axisY);
        hash.addVec3(decal.axisZ);
        hash.addVec3(decal.size);
        hash.addI32(decal.materialIndex);
        hash.addI32(decal.sortOrder);
        hash.addFloat(decal.fadeScreenSize);
    }

    static std::uint64_t ComputeAccumulationStateHash(const CameraDataCPU &cameraCPU,
                                                      int width,
                                                      int height,
                                                      std::uint64_t sceneRevision,
                                                      const std::vector<Light> &lights,
                                                      const SceneMetaResources *metaRes,
                                                      MetalAccumulationMode accumulationMode)
    {
        HashBuilder64 hash;
        HashCameraForAccumulation(hash, cameraCPU);
        hash.addU64(sceneRevision);
        hash.addI32(width);
        hash.addI32(height);
        hash.addU32(kMetalShaderUvDebugMode);
        hash.addU32(static_cast<std::uint32_t>(accumulationMode));

        hash.addU64(static_cast<std::uint64_t>(lights.size()));
        for (const Light &light : lights)
            HashLight(hash, light);

        const bool hasMetaRes = (metaRes != nullptr);
        hash.addBool(hasMetaRes);
        if (!hasMetaRes)
            return hash.value;

        hash.addU64(static_cast<std::uint64_t>(metaRes->baseColorTextures.size()));
        for (const std::string &path : metaRes->baseColorTextures)
            hash.addString(path);

        hash.addU64(static_cast<std::uint64_t>(metaRes->linearTextures.size()));
        for (const std::string &path : metaRes->linearTextures)
            hash.addString(path);

        hash.addU64(static_cast<std::uint64_t>(metaRes->materials.size()));
        for (const SceneMetaMaterial &material : metaRes->materials)
            HashMaterial(hash, material);

        hash.addU64(static_cast<std::uint64_t>(metaRes->materialsPBR.size()));
        for (const SceneMetaMaterial &material : metaRes->materialsPBR)
            HashMaterial(hash, material);

        hash.addU64(static_cast<std::uint64_t>(metaRes->decals.size()));
        for (const SceneMetaDecal &decal : metaRes->decals)
            HashDecal(hash, decal);

        return hash.value;
    }

    static uint32_t EffectiveSamplesPerDispatch(const CameraDataCPU &cameraCPU)
    {
        if (kMetalShaderUvDebugMode != 0u)
            return 0u;

        return static_cast<uint32_t>(std::max(cameraCPU.samplesPerPixel, 1));
    }

    static void ResetMetalAccumulationState()
    {
        g_accumulatedSampleCount = 0;
        g_accumulationDispatchCount = 0;
        g_accumulationStateHash = kInvalidAccumulationHash;
        g_accumTexture = nil;
    }

    static void PrintTextureFrameProfile(const TextureFrameProfile &p)
    {
        auto &tot = g_textureProfileTotals;
        tot.frameCount += 1;
        tot.initMetalMs += p.initMetalMs;
        tot.ensureMaterialsMs += p.ensureMaterialsMs;
        tot.accumTextureMs += p.accumTextureMs;
        tot.intermediateTexturesMs += p.intermediateTexturesMs;
        tot.geometryBuffersMs += p.geometryBuffersMs;
        tot.smallBuffersMs += p.smallBuffersMs;
        tot.lightUploadMs += p.lightUploadMs;
        tot.emissiveDecalMs += p.emissiveDecalMs;
        tot.postParamsMs += p.postParamsMs;
        tot.encodeMs += p.encodeMs;
        tot.waitMs += p.waitMs;
        tot.readbackMs += p.readbackMs;
        tot.gpuMs += p.gpuMs;
        tot.primaryDepthGpuMs += p.primaryDepthGpuMs;
        tot.pathTraceGpuMs += p.pathTraceGpuMs;
        tot.bloomExtractGpuMs += p.bloomExtractGpuMs;
        tot.blurHGpuMs += p.blurHGpuMs;
        tot.blurVGpuMs += p.blurVGpuMs;
        tot.finalCompositeGpuMs += p.finalCompositeGpuMs;
        tot.totalMs += p.totalMs;

        const double inv = (tot.frameCount > 0) ? (1.0 / static_cast<double>(tot.frameCount)) : 0.0;
        const double avgTotal = tot.totalMs * inv;
        const double avgWait = tot.waitMs * inv;
        const double avgReadback = tot.readbackMs * inv;
        const double avgGPU = tot.gpuMs * inv;
        const double avgPrimaryDepth = tot.primaryDepthGpuMs * inv;
        const double avgPathTrace = tot.pathTraceGpuMs * inv;
        const double avgBloomExtract = tot.bloomExtractGpuMs * inv;
        const double avgBlurH = tot.blurHGpuMs * inv;
        const double avgBlurV = tot.blurVGpuMs * inv;
        const double avgFinalComposite = tot.finalCompositeGpuMs * inv;

        std::ostringstream oss;
        oss << "[MetalProfiler][Texture] call=" << p.callIndex
            << " mode=" << AccumulationModeName(p.accumulationMode)
            << " accumSamples=" << p.accumulatedSamples
            << " batchSamples=" << p.samplesThisDispatch << "\n"
            << "  scene stats:\n";
        AppendCountLine(oss, "proto tris", p.prototypeTriangles);
        AppendCountLine(oss, "instances", p.totalInstances);
        AppendCountLine(oss, "TLAS nodes", p.tlasNodeCount);
        AppendCountLine(oss, "BLAS nodes", p.blasNodeCount);

        oss << "  current:\n";
        AppendProfileLine(oss, "init", p.initMetalMs, p.totalMs);
        AppendProfileLine(oss, "materials", p.ensureMaterialsMs, p.totalMs);
        AppendProfileLine(oss, "accumTex", p.accumTextureMs, p.totalMs);
        AppendProfileLine(oss, "postTex", p.intermediateTexturesMs, p.totalMs);
        AppendProfileLine(oss, "geom", p.geometryBuffersMs, p.totalMs);
        AppendProfileLine(oss, "small", p.smallBuffersMs, p.totalMs);
        AppendProfileLine(oss, "lights", p.lightUploadMs, p.totalMs);
        AppendProfileLine(oss, "emissive+decal", p.emissiveDecalMs, p.totalMs);
        AppendProfileLine(oss, "postParams", p.postParamsMs, p.totalMs);
        AppendProfileLine(oss, "encode", p.encodeMs, p.totalMs);
        AppendProfileLine(oss, "wait", p.waitMs, p.totalMs);
        AppendProfileLine(oss, "readback", p.readbackMs, p.totalMs);
        AppendProfileLine(oss, "gpu", p.gpuMs, p.totalMs);
        AppendProfileLine(oss, "total", p.totalMs, p.totalMs);

        oss << "  gpu passes (current):\n";
        AppendProfileLine(oss, "primary-depth", p.primaryDepthGpuMs, p.gpuMs);
        AppendProfileLine(oss, "path trace", p.pathTraceGpuMs, p.gpuMs);
        AppendProfileLine(oss, "bloom extract", p.bloomExtractGpuMs, p.gpuMs);
        AppendProfileLine(oss, "blur H", p.blurHGpuMs, p.gpuMs);
        AppendProfileLine(oss, "blur V", p.blurVGpuMs, p.gpuMs);
        AppendProfileLine(oss, "final composite", p.finalCompositeGpuMs, p.gpuMs);

        oss << "  average:\n";
        AppendProfileLine(oss, "avgTotal", avgTotal, avgTotal);
        AppendProfileLine(oss, "avgWait", avgWait, avgTotal);
        AppendProfileLine(oss, "avgReadback", avgReadback, avgTotal);
        AppendProfileLine(oss, "avgGPU", avgGPU, avgTotal);

        oss << "  gpu passes (average):\n";
        AppendProfileLine(oss, "primary-depth", avgPrimaryDepth, avgGPU);
        AppendProfileLine(oss, "path trace", avgPathTrace, avgGPU);
        AppendProfileLine(oss, "bloom extract", avgBloomExtract, avgGPU);
        AppendProfileLine(oss, "blur H", avgBlurH, avgGPU);
        AppendProfileLine(oss, "blur V", avgBlurV, avgGPU);
        AppendProfileLine(oss, "final composite", avgFinalComposite, avgGPU);
        oss << std::endl;

        AppendProfileText(oss.str());
    }

}

struct LightGPU
{
    int   type;
    std::uint32_t flags;

    Vec3  position;
    Vec3  direction;
    Vec3  color;

    float intensity;
    float radius;            // геометрический размер источника
    float sourceLength;      // длина трубчатого/капсульного источника вдоль direction
    float softSourceRadius;  // эффективный мягкий радиус для spec/penumbra
    float spotSize;          // радианы
    float spotBlend;         // 0..1

    float attenuationRadius; // UE AttenuationRadius, 0 = бесконечный (старое поведение)
    std::uint32_t ownerId;
};
static_assert(sizeof(LightGPU) == 76, "LightGPU size must be 76 bytes");

// Материал на GPU: треугольник хранит materialIndex, а материал хранит ссылки на текстуры.
// (Это масштабируемый “движковый” способ, как в UE/рендер-пайплайнах.)
// Материал на GPU (минимальный layout, должен совпадать с RayTrace.metal)
struct MaterialGPU
{
    int32_t baseColorTexIndex; // индекс в массиве baseColorTextures, -1 если нет
    int32_t emissionTexIndex;
    int32_t baseColorUvSet;
    int32_t emissionUvSet;
};

// Расширенный материал (layout должен совпадать с RayTrace.metal).
struct MaterialGPU_PBR
{
    int32_t baseColorTexIndex;
    int32_t emissionTexIndex;
    int32_t normalTexIndex;
    int32_t ormTexIndex;
    int32_t roughnessTexIndex;
    int32_t metallicTexIndex;
    int32_t occlusionTexIndex;
    int32_t flags;

    int32_t baseColorUvSet;
    int32_t emissionUvSet;
    int32_t normalUvSet;
    int32_t ormUvSet;
    int32_t roughnessUvSet;
    int32_t metallicUvSet;
    int32_t occlusionUvSet;
    int32_t specialModel;

    int32_t specialTex0Index;
    int32_t specialTex1Index;
    int32_t _pad0;
    int32_t _pad1;

    float   specialScalar0;
    float   specialScalar1;
    float   specialScalar2;
    float   specialScalar3;
    float   specialScalar4;
    float   specialScalar5;
    float   _pad2;
    float   _pad3;
};
static_assert(sizeof(MaterialGPU_PBR) == 112, "MaterialGPU_PBR size must be 112 bytes");
static_assert(sizeof(MaterialGPU) == 16, "MaterialGPU size must be 16 bytes");

struct DecalGPU
{
    float posX,   posY,   posZ,   sizeX;   // sizeX = half depth along projection axis (UE local X)
    float axisXx, axisXy, axisXz, sizeY;   // sizeY = half width  along UE local Y
    float axisYx, axisYy, axisYz, sizeZ;   // sizeZ = half height along UE local Z
    float axisZx, axisZy, axisZz, opacity;

    int32_t baseColorTexIndex;
    int32_t ormTexIndex;
    int32_t roughnessTexIndex;
    int32_t normalTexIndex;

    int32_t opacityTexIndex;
    int32_t detailTexIndex;
    int32_t flags;
    int32_t _pad0;

    float baseColorX, baseColorY, baseColorZ, roughnessBias;
    float tilingU, tilingV, opacityPower, normalIntensity;
};
static_assert(sizeof(DecalGPU) == 128, "DecalGPU size must be 128 bytes");

struct EmissiveTriangleGPUCPU
{
    uint32_t triIndex;
    uint32_t instanceIndex;
    float    area;
    float    selectionPdf;
    float    cdf;
};
static_assert(sizeof(EmissiveTriangleGPUCPU) == 20, "EmissiveTriangleGPUCPU size must be 20 bytes");

struct AirDustVolumeGPUCPU
{
    float centerX, centerY, centerZ, density;
    float extentX, extentY, extentZ, anisotropy;
    float lightPosX, lightPosY, lightPosZ, lightIntensity;
    float lightColorX, lightColorY, lightColorZ, lightRadius;
};
static_assert(sizeof(AirDustVolumeGPUCPU) == 64, "AirDustVolumeGPUCPU size must be 64 bytes");

struct PostProcessParamsCPU
{
    float exposure;
    float bloomIntensity;
    float bloomThreshold;
    float vignetteIntensity;

    float chromaticAberration;
    float filmGrainIntensity;
    float filmSlope;
    float filmToe;

    float filmShoulder;
    float filmBlackClip;
    float filmWhiteClip;
    float fogDensity;

    float fogHeightFalloff;
    float fogScatteringG;
    float fogColorX;
    float fogColorY;

    float fogColorZ;
    float fogExtinctionScale;
    float fogAlbedoX;
    float fogAlbedoY;

    float fogAlbedoZ;
    float volumetricFog;
    float nearPlane;
    float farPlane;

    float time;
    float width;
    float height;
    float _pad0;

    float colorSaturationX;
    float colorSaturationY;
    float colorSaturationZ;
    float shadowLift;

    float fogStartDistance;
    float fogMaxOpacity;
    float fogHeightZ;
    float worldUnitToMeters;
};
static_assert(sizeof(PostProcessParamsCPU) == 144, "PostProcessParamsCPU size must be 144 bytes");

// -------------------- helpers: texture loading --------------------

static id<MTLTexture> CreateSolidTextureBGRA8_sRGB(id<MTLDevice> dev,
                                                   uint8_t b, uint8_t g, uint8_t r, uint8_t a)
{
    if (!dev) return nil;

    uint8_t px[4] = { b, g, r, a };

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
    if (!tex) return nil;

    MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
    [tex replaceRegion:region mipmapLevel:0 withBytes:px bytesPerRow:4];
    return tex;
}

// Линейная версия (без sRGB), для data-текстур (normal/orm/rough/metal/ao).
static id<MTLTexture> CreateSolidTextureBGRA8_Unorm(id<MTLDevice> dev,
                                                    uint8_t b, uint8_t g, uint8_t r, uint8_t a)
{
    if (!dev) return nil;

    uint8_t px[4] = { b, g, r, a };

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
    if (!tex) return nil;

    MTLRegion region = MTLRegionMake2D(0, 0, 1, 1);
    [tex replaceRegion:region mipmapLevel:0 withBytes:px bytesPerRow:4];
    return tex;
}

// Декодим PNG/JPG/etc через ImageIO в BGRA8 и создаём MTLTexture.
// Не требует stb_image/MetalKit. Нужно линковать framework'и ImageIO/CoreGraphics.
static id<MTLTexture> LoadTextureBGRA8_sRGB(id<MTLDevice> dev, const std::string& pathUtf8)
{
    if (!dev) return nil;

    NSString *nsPath = [NSString stringWithUTF8String:pathUtf8.c_str()];
    if (!nsPath) return nil;

    NSURL *url = [NSURL fileURLWithPath:nsPath];
    if (!url) return nil;

    CGImageSourceRef src = CGImageSourceCreateWithURL((CFURLRef)url, nullptr);
    if (!src) return nil;

    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);

    if (!img) return nil;

    const size_t w = CGImageGetWidth(img);
    const size_t h = CGImageGetHeight(img);

    if (w == 0 || h == 0)
    {
        CGImageRelease(img);
        return nil;
    }

    std::vector<uint8_t> pixels;
    pixels.resize(w * h * 4);

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();

    // 32Little + PremultipliedFirst => BGRA
    CGContextRef ctx = CGBitmapContextCreate(pixels.data(),
                                             w, h,
                                             8,
                                             w * 4,
                                             cs,
                                             kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);

    CGColorSpaceRelease(cs);

    if (!ctx)
    {
        CGImageRelease(img);
        return nil;
    }

    // Рисуем без флипа: верхняя строка изображения станет первой строкой в буфере.
    // Если UV окажутся “вверх ногами” — проще и правильнее поправить uv.y в шейдере.
    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h), img);

    CGContextRelease(ctx);
    CGImageRelease(img);

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
                                                           width:(NSUInteger)w
                                                          height:(NSUInteger)h
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
    if (!tex) return nil;

    MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)w, (NSUInteger)h);
    [tex replaceRegion:region
           mipmapLevel:0
             withBytes:pixels.data()
           bytesPerRow:(NSUInteger)(w * 4)];

    return tex;
}

// Линейная версия (без sRGB), для data-текстур (normal/orm/rough/metal/ao).
static id<MTLTexture> LoadTextureBGRA8_Unorm(id<MTLDevice> dev, const std::string& pathUtf8)
{
    if (!dev) return nil;

    NSString *nsPath = [NSString stringWithUTF8String:pathUtf8.c_str()];
    if (!nsPath) return nil;

    NSURL *url = [NSURL fileURLWithPath:nsPath];
    if (!url) return nil;

    CGImageSourceRef src = CGImageSourceCreateWithURL((CFURLRef)url, nullptr);
    if (!src) return nil;

    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);

    if (!img) return nil;

    const size_t w = CGImageGetWidth(img);
    const size_t h = CGImageGetHeight(img);

    if (w == 0 || h == 0)
    {
        CGImageRelease(img);
        return nil;
    }

    const size_t bytesPerRow = w * 4;

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    if (!cs)
    {
        CGImageRelease(img);
        return nil;
    }

    CGContextRef ctx = CGBitmapContextCreate(nullptr, w, h, 8, bytesPerRow, cs,
                                             kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Little);
    CGColorSpaceRelease(cs);

    if (!ctx)
    {
        CGImageRelease(img);
        return nil;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h), img);
    CGImageRelease(img);

    uint8_t *data = (uint8_t*)CGBitmapContextGetData(ctx);
    if (!data)
    {
        CGContextRelease(ctx);
        return nil;
    }

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:(NSUInteger)w
                                                          height:(NSUInteger)h
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;

    id<MTLTexture> tex = [dev newTextureWithDescriptor:desc];
    if (!tex)
    {
        CGContextRelease(ctx);
        return nil;
    }

    MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)w, (NSUInteger)h);
    [tex replaceRegion:region mipmapLevel:0 withBytes:data bytesPerRow:bytesPerRow];

    CGContextRelease(ctx);
    return tex;
}

static bool EnsureMaterialAndTexturesLoaded(const SceneMetaResources* metaRes)
{
    if (!g_device)
        return false;

    if (!metaRes)
    {
        if (g_loadedMetaRes != nullptr)
        {
            ResetMetalAccumulationState();
            g_loadedMetaRes = nullptr;
            g_cachedDecalMetaRes = nullptr;
            g_baseColorTextures.clear();
            g_baseColorTexturePaths.clear();
            g_cpuSceneTextureCache.clear();
            g_linearTextureValid.clear();
            g_materialBuffer = nil;
            g_materialCountBuffer = nil;
            g_materialPBRBuffer = nil;
            g_materialPBRCountBuffer = nil;
            g_decalBuffer = nil;
            g_decalCountBuffer = nil;
            g_airDustVolumeBuffer = nil;
            g_airDustVolumeCountBuffer = nil;
            g_cachedAirDustMetaRes = nullptr;
        }
        return true;
    }

    // ------------------------------------------------------------
    // Важно:
    // В compute-пайплайне лимит texture-аргументов обычно 128.
    // Уже заняты texture(0)=accum, texture(1)=hdr, texture(2)=albedo guide, texture(3)=normal guide,
    // поэтому под сценовые текстуры остаётся 124 слота.
    // Чтобы не "резать" линейные PBR-текстуры (LevelMeta13+), пакуем их в ЕДИНЫЙ пул:
    //   [baseColor/emission (sRGB)] + [normal/orm/rough/metal/ao (linear)]
    // и биндится всё подряд начиная с texture(4).
    //
    // Сейчас шейдер использует только baseColor/emission индексы (0..baseCount-1).
    // PBR-индексы готовятся в g_materialPBRBuffer для следующего шага (шейдер будет читать их позже).
    // ------------------------------------------------------------

    const std::vector<std::string>& baseFull = metaRes->baseColorTextures;

    std::vector<std::string> linFull;
    if constexpr (has_member_linearTextures<SceneMetaResources>::value)
    {
        linFull = metaRes->linearTextures;
    }

    // Собираем комбинированный список путей, ограничивая общим лимитом texture slots.
    const size_t baseCount  = std::min(baseFull.size(), (size_t)kMaxAllTextures); // kMaxAllTextures = 124 (см. выше)
    const size_t remaining  = (size_t)kMaxAllTextures - baseCount;
    const size_t linCount   = std::min(linFull.size(), remaining);

    std::vector<std::string> combinedPaths;
    combinedPaths.reserve(baseCount + linCount);
    combinedPaths.insert(combinedPaths.end(), baseFull.begin(), baseFull.begin() + baseCount);
    combinedPaths.insert(combinedPaths.end(), linFull.begin(),  linFull.begin() + linCount);

    // Если набор путей не изменился — оставляем как есть
    const bool same = (metaRes == g_loadedMetaRes) &&
                      (combinedPaths == g_baseColorTexturePaths);
    if (same && g_materialBuffer != nil && g_materialCountBuffer != nil)
        return true;

    // ИНАЧЕ: перезагружаем текстуры/материалы и сбрасываем accumulation (иначе будет "смесь" старых/новых)
    ResetMetalAccumulationState();

    g_baseColorTextures.clear();
    g_baseColorTexturePaths.clear();
    g_cpuSceneTextureCache.clear();
    g_linearTextureValid.clear();

    g_materialBuffer          = nil;
    g_materialCountBuffer     = nil;
    g_materialPBRBuffer       = nil;
    g_materialPBRCountBuffer  = nil;
    g_cachedDecalMetaRes      = nullptr;
    g_decalBuffer             = nil;
    g_decalCountBuffer        = nil;
    g_cachedAirDustMetaRes    = nullptr;
    g_airDustVolumeBuffer     = nil;
    g_airDustVolumeCountBuffer = nil;

    g_baseColorTextures.reserve(baseCount + linCount);
    g_baseColorTexturePaths = combinedPaths;

    g_linearTextureValid.reserve(baseCount + linCount);

    id<MTLTexture> fallbackSRGB   = CreateSolidTextureBGRA8_sRGB(g_device,  /*B*/255, /*G*/0, /*R*/255, /*A*/255);
    id<MTLTexture> fallbackLinear = CreateSolidTextureBGRA8_Unorm(g_device, /*B*/255, /*G*/0, /*R*/255, /*A*/255);

    // --- загрузка baseColor/emission (sRGB) ---
    for (size_t i = 0; i < baseCount; ++i)
    {
        const std::string& p = combinedPaths[i];

        id<MTLTexture> tex = nil;
        if (!std::filesystem::exists(p))
        {
            std::cerr << "MetalRenderer: texture file not found: " << p << "\n";
            tex = fallbackSRGB;
        }
        else
        {
            tex = LoadTextureBGRA8_sRGB(g_device, p);
            if (!tex)
            {
                std::cerr << "MetalRenderer: failed to load texture: " << p << "\n";
                tex = fallbackSRGB;
            }
        }

        g_baseColorTextures.push_back(tex);
        g_linearTextureValid.push_back(false);
    }

    // --- загрузка linear textures (normal/ORM/roughness/metallic/ao) ---
    for (size_t i = 0; i < linCount; ++i)
    {
        const std::string& p = combinedPaths[baseCount + i];

        id<MTLTexture> tex = nil;
        if (!std::filesystem::exists(p))
        {
            std::cerr << "MetalRenderer: linear texture file not found: " << p << "\n";
            tex = fallbackLinear;
        }
        else
        {
            tex = LoadTextureBGRA8_Unorm(g_device, p);
            if (!tex)
            {
                std::cerr << "MetalRenderer: failed to load linear texture: " << p << "\n";
                tex = fallbackLinear;
            }
        }

        g_baseColorTextures.push_back(tex);
        g_linearTextureValid.push_back(true);
    }

    // Предупреждение выводим ОДИН раз на перезагрузку, а не каждый кадр.
    const size_t totalFull = baseFull.size() + linFull.size();
    if (totalFull > (size_t)kMaxAllTextures)
    {
        std::cerr << "MetalRenderer: WARNING total textures (base+linear) = " << totalFull
                  << ", but MAX slots = " << kMaxAllTextures
                  << ". Some textures will be ignored.\n";
    }

    // --- материалы (минимальный layout, который читает текущий шейдер) ---
    std::vector<MaterialGPU> mats;
    mats.resize(metaRes->materials.size());

    for (std::size_t i = 0; i < metaRes->materials.size(); ++i)
    {
        const SceneMetaMaterial& sm = metaRes->materials[i];

        MaterialGPU m{};
        int bc = sm.baseColorTexIndex;
        if (bc < 0 || (size_t)bc >= baseCount)
            bc = -1;

        int em = sm.emissionTexIndex;
        if (em < 0 || (size_t)em >= baseCount)
            em = -1;

        m.baseColorTexIndex = bc;
        m.emissionTexIndex  = em;
        m.baseColorUvSet    = std::clamp(sm.baseColorUvSet, 0, 2);
        m.emissionUvSet     = std::clamp(sm.emissionUvSet, 0, 2);

        mats[i] = m;
    }

    if (mats.empty())
        mats.resize(1);

    // --- PBR materials table (индексы в комбинированном пуле).
    // Предполагаем, что SceneMetaMaterial хранит normal/orm/rough/metal/ao как индексы ВНУТРИ linFull.
    auto mapLinear = [&](int idx) -> int32_t
    {
        if (idx < 0)
            return -1;
        if ((size_t)idx >= linCount)
            return -1;
        return (int32_t)(baseCount + (size_t)idx);
    };
    auto mapSceneTex = [&](int idx, bool isLinear) -> int32_t
    {
        return isLinear ? mapLinear(idx) : idx;
    };

    std::vector<MaterialGPU_PBR> matsPBR;
    matsPBR.resize(metaRes->materials.size());

    for (std::size_t i = 0; i < metaRes->materials.size(); ++i)
    {
        const SceneMetaMaterial& sm = metaRes->materials[i];

        MaterialGPU_PBR mp{};
        mp.baseColorTexIndex = (mats[i].baseColorTexIndex >= 0) ? mats[i].baseColorTexIndex : -1;
        mp.emissionTexIndex  = (mats[i].emissionTexIndex  >= 0) ? mats[i].emissionTexIndex  : -1;

        mp.normalTexIndex    = -1;
        mp.ormTexIndex       = -1;
        mp.roughnessTexIndex = -1;
        mp.metallicTexIndex  = -1;
        mp.occlusionTexIndex = -1;
        mp.flags             = 0;
        mp.baseColorUvSet    = std::clamp(sm.baseColorUvSet, 0, 2);
        mp.emissionUvSet     = std::clamp(sm.emissionUvSet, 0, 2);
        mp.normalUvSet       = std::clamp(sm.normalUvSet, 0, 2);
        mp.ormUvSet          = std::clamp(sm.ormUvSet, 0, 2);
        mp.roughnessUvSet    = std::clamp(sm.roughnessUvSet, 0, 2);
        mp.metallicUvSet     = std::clamp(sm.metallicUvSet, 0, 2);
        mp.occlusionUvSet    = std::clamp(sm.occlusionUvSet, 0, 2);
        mp.specialModel      = sm.specialModel;
        mp.specialTex0Index  = mapSceneTex(sm.specialTex0Index, false);
        mp.specialTex1Index  = mapSceneTex(sm.specialTex1Index, false);
        mp._pad0             = 0;
        mp._pad1             = 0;
        mp.specialScalar0    = sm.specialScalar0;
        mp.specialScalar1    = sm.specialScalar1;
        mp.specialScalar2    = sm.specialScalar2;
        mp.specialScalar3    = sm.specialScalar3;
        mp.specialScalar4    = sm.specialScalar4;
        mp.specialScalar5    = sm.specialScalar5;
        mp._pad2             = 0.0f;
        mp._pad3             = 0.0f;

        if constexpr (has_member_normalTexIndex<SceneMetaMaterial>::value)
            mp.normalTexIndex = mapLinear(sm.normalTexIndex);

        if constexpr (has_member_ormTexIndex<SceneMetaMaterial>::value)
            mp.ormTexIndex = mapLinear(sm.ormTexIndex);

        if constexpr (has_member_roughnessTexIndex<SceneMetaMaterial>::value)
            mp.roughnessTexIndex = mapLinear(sm.roughnessTexIndex);

        if constexpr (has_member_metallicTexIndex<SceneMetaMaterial>::value)
            mp.metallicTexIndex = mapLinear(sm.metallicTexIndex);

        if constexpr (has_member_occlusionTexIndex<SceneMetaMaterial>::value)
            mp.occlusionTexIndex = mapLinear(sm.occlusionTexIndex);
        else if constexpr (has_member_ormTexIndex<SceneMetaMaterial>::value)
            mp.occlusionTexIndex = mapLinear(sm.ormTexIndex); // fallback: AO channel is inside ORM

        if (sm.emissionUseAlphaMask)
            mp.flags |= 1;
        if (sm.thinEmissiveSurface)
            mp.flags |= 2;

        matsPBR[i] = mp;
    }

    if (matsPBR.empty())
        matsPBR.resize(1);

    // --- GPU buffers ---
    g_materialBuffer =
        [g_device newBufferWithBytes:mats.data()
                              length:(NSUInteger)(mats.size() * sizeof(MaterialGPU))
                             options:MTLResourceStorageModeShared];

    uint32_t materialCountU32 = (uint32_t)mats.size();
    g_materialCountBuffer =
        [g_device newBufferWithBytes:&materialCountU32
                              length:sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];

    uint32_t materialPBRCountU32 = (uint32_t)matsPBR.size();
    g_materialPBRBuffer =
        [g_device newBufferWithBytes:matsPBR.data()
                              length:(NSUInteger)(matsPBR.size() * sizeof(MaterialGPU_PBR))
                             options:MTLResourceStorageModeShared];

    g_materialPBRCountBuffer =
        [g_device newBufferWithBytes:&materialPBRCountU32
                              length:sizeof(uint32_t)
                             options:MTLResourceStorageModeShared];

    // Логируем только при перезагрузке
    std::cerr << "MetalRenderer: loaded scene textures = " << (uint32_t)(baseCount + linCount)
              << " (base=" << (uint32_t)baseCount << ", linear=" << (uint32_t)linCount << ")"
              << ", materials = " << materialCountU32
              << ", materialsPBR = " << materialPBRCountU32
              << "\n";

    g_loadedMetaRes = metaRes;
    return (g_materialBuffer != nil && g_materialCountBuffer != nil);
}


static inline float LuminanceCPU(const Vec3 &c)
{
    return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
}

static inline float Clamp01CPU(float v)
{
    return std::clamp(v, 0.0f, 1.0f);
}

static inline float FractCPU(float v)
{
    return v - std::floor(v);
}

static inline Vec2 FractCPU(const Vec2 &uv)
{
    return Vec2{FractCPU(uv.x), FractCPU(uv.y)};
}

static inline float SrgbToLinearCPU(float c)
{
    if (c <= 0.04045f)
        return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

static CpuSampleTexture LoadCpuSceneTexture(const std::string& pathUtf8)
{
    CpuSampleTexture out{};
    NSString *nsPath = [NSString stringWithUTF8String:pathUtf8.c_str()];
    if (!nsPath)
        return out;

    NSURL *url = [NSURL fileURLWithPath:nsPath];
    if (!url)
        return out;

    CGImageSourceRef src = CGImageSourceCreateWithURL((CFURLRef)url, nullptr);
    if (!src)
        return out;

    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
    CFRelease(src);
    if (!img)
        return out;

    const size_t w = CGImageGetWidth(img);
    const size_t h = CGImageGetHeight(img);
    if (w == 0 || h == 0)
    {
        CGImageRelease(img);
        return out;
    }

    out.width = w;
    out.height = h;
    out.bgra.resize(w * h * 4u);

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(out.bgra.data(),
                                             w, h,
                                             8,
                                             w * 4,
                                             cs,
                                             kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGColorSpaceRelease(cs);
    if (!ctx)
    {
        out = {};
        CGImageRelease(img);
        return out;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h), img);
    CGContextRelease(ctx);
    CGImageRelease(img);
    return out;
}

static const CpuSampleTexture* GetCpuSceneTexture(const SceneMetaResources *metaRes, int32_t texIndex)
{
    if (!metaRes || texIndex < 0 || (std::size_t)texIndex >= metaRes->baseColorTextures.size())
        return nullptr;

    const std::string &path = metaRes->baseColorTextures[(std::size_t)texIndex];
    auto it = g_cpuSceneTextureCache.find(path);
    if (it == g_cpuSceneTextureCache.end())
    {
        auto [ins, _] = g_cpuSceneTextureCache.emplace(path, LoadCpuSceneTexture(path));
        it = ins;
    }
    return it->second.valid() ? &it->second : nullptr;
}

static inline Vec3 DecodeCpuSceneTexelLinear(const CpuSampleTexture &tex, int x, int y)
{
    if (!tex.valid())
        return Vec3{0.0f, 0.0f, 0.0f};

    const int w = (int)tex.width;
    const int h = (int)tex.height;
    x = ((x % w) + w) % w;
    y = ((y % h) + h) % h;

    const std::size_t idx = ((std::size_t)y * tex.width + (std::size_t)x) * 4u;
    const float b = tex.bgra[idx + 0u] / 255.0f;
    const float g = tex.bgra[idx + 1u] / 255.0f;
    const float r = tex.bgra[idx + 2u] / 255.0f;
    return Vec3{SrgbToLinearCPU(r), SrgbToLinearCPU(g), SrgbToLinearCPU(b)};
}

static inline float DecodeCpuSceneAlpha(const CpuSampleTexture &tex, int x, int y)
{
    if (!tex.valid())
        return 1.0f;

    const int w = (int)tex.width;
    const int h = (int)tex.height;
    x = ((x % w) + w) % w;
    y = ((y % h) + h) % h;

    const std::size_t idx = ((std::size_t)y * tex.width + (std::size_t)x) * 4u;
    return tex.bgra[idx + 3u] / 255.0f;
}

static inline void SampleCpuSceneTextureLinear(const CpuSampleTexture &tex,
                                               const Vec2 &uvIn,
                                               Vec3 &rgbOut,
                                               float &alphaOut)
{
    if (!tex.valid())
    {
        rgbOut = Vec3{1.0f, 1.0f, 1.0f};
        alphaOut = 1.0f;
        return;
    }

    const Vec2 uv = FractCPU(uvIn);
    const float fx = uv.x * (float)tex.width - 0.5f;
    const float fy = uv.y * (float)tex.height - 0.5f;
    const int x0 = (int)std::floor(fx);
    const int y0 = (int)std::floor(fy);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = fx - (float)x0;
    const float ty = fy - (float)y0;

    const Vec3 c00 = DecodeCpuSceneTexelLinear(tex, x0, y0);
    const Vec3 c10 = DecodeCpuSceneTexelLinear(tex, x1, y0);
    const Vec3 c01 = DecodeCpuSceneTexelLinear(tex, x0, y1);
    const Vec3 c11 = DecodeCpuSceneTexelLinear(tex, x1, y1);

    const float a00 = DecodeCpuSceneAlpha(tex, x0, y0);
    const float a10 = DecodeCpuSceneAlpha(tex, x1, y0);
    const float a01 = DecodeCpuSceneAlpha(tex, x0, y1);
    const float a11 = DecodeCpuSceneAlpha(tex, x1, y1);

    const Vec3 cx0{
        c00.x + (c10.x - c00.x) * tx,
        c00.y + (c10.y - c00.y) * tx,
        c00.z + (c10.z - c00.z) * tx
    };
    const Vec3 cx1{
        c01.x + (c11.x - c01.x) * tx,
        c01.y + (c11.y - c01.y) * tx,
        c01.z + (c11.z - c01.z) * tx
    };
    rgbOut = Vec3{
        cx0.x + (cx1.x - cx0.x) * ty,
        cx0.y + (cx1.y - cx0.y) * ty,
        cx0.z + (cx1.z - cx0.z) * ty
    };

    const float ax0 = a00 + (a10 - a00) * tx;
    const float ax1 = a01 + (a11 - a01) * tx;
    alphaOut = ax0 + (ax1 - ax0) * ty;
}

static inline Vec2 TriangleUvAtBary(const Triangle &tri, int uvSet, float b0, float b1, float b2)
{
    const int clampedSet = std::clamp(uvSet, 0, 2);
    const Vec2 &uv0 = TriangleUV(tri, clampedSet, 0);
    const Vec2 &uv1 = TriangleUV(tri, clampedSet, 1);
    const Vec2 &uv2 = TriangleUV(tri, clampedSet, 2);
    return Vec2{
        uv0.x * b0 + uv1.x * b1 + uv2.x * b2,
        uv0.y * b0 + uv1.y * b1 + uv2.y * b2
    };
}

static inline float SampleEmissiveMaskCPU(const Vec3 &rgb, float alpha)
{
    const float alphaMask = Clamp01CPU(alpha);
    const float lumaMask = Clamp01CPU(LuminanceCPU(rgb));
    if (alphaMask > 1.0e-4f)
    {
        const float t = Clamp01CPU((alphaMask - 0.22f) / std::max(0.82f - 0.22f, 1.0e-6f));
        return t * t * (3.0f - 2.0f * t);
    }

    const float t = Clamp01CPU((lumaMask - 0.16f) / std::max(0.42f - 0.16f, 1.0e-6f));
    return t * t * (3.0f - 2.0f * t);
}

static Vec3 EvaluateTriangleEmissionCPU(const Triangle &tri,
                                        const SceneMetaMaterial &material,
                                        const SceneMetaResources *metaRes,
                                        const Vec2 &uvBase,
                                        const Vec2 &uvEmission,
                                        float ndv)
{
    Vec3 emissive{tri.emission.x, tri.emission.y, tri.emission.z};

    Vec3 baseColorTex{1.0f, 1.0f, 1.0f};
    float dummyAlpha = 1.0f;
    if (const CpuSampleTexture *baseTex = GetCpuSceneTexture(metaRes, material.baseColorTexIndex))
        SampleCpuSceneTextureLinear(*baseTex, uvBase, baseColorTex, dummyAlpha);

    Vec3 emissionTex{1.0f, 1.0f, 1.0f};
    float emissionAlpha = 1.0f;
    const CpuSampleTexture *emissionTexCpu = GetCpuSceneTexture(metaRes, material.emissionTexIndex);
    if (emissionTexCpu)
        SampleCpuSceneTextureLinear(*emissionTexCpu, uvEmission, emissionTex, emissionAlpha);

    if (material.specialModel == 2)
    {
        emissive.x *= emissionTex.x;
        emissive.y *= emissionTex.y;
        emissive.z *= emissionTex.z;
    }
    else if (emissionTexCpu)
    {
        if (material.emissionUseAlphaMask)
        {
            const float mask = SampleEmissiveMaskCPU(emissionTex, emissionAlpha);
            emissive.x *= mask;
            emissive.y *= mask;
            emissive.z *= mask;
        }
        else
        {
            emissive.x *= emissionTex.x;
            emissive.y *= emissionTex.y;
            emissive.z *= emissionTex.z;
        }
    }

    if (material.specialModel == 1)
    {
        const float baseLuma = Clamp01CPU(LuminanceCPU(baseColorTex));
        const float innerGlow = Clamp01CPU(material.specialScalar0);
        const float transparency = Clamp01CPU(material.specialScalar4);
        const float rim = std::pow(Clamp01CPU(1.0f - ndv), std::max(material.specialScalar1, 1.0f));
        const float dirtTransmission =
            std::clamp(material.specialScalar2 + baseLuma * std::max(material.specialScalar5, 0.5f) + material.specialScalar3,
                       0.20f, 1.40f);
        const float glowGain = (0.95f + innerGlow * 0.30f) * (0.75f + baseLuma * 0.35f);
        const float rimGain = 1.0f + rim * 0.10f * transparency;
        const float totalGain = std::max(dirtTransmission * glowGain * rimGain, 0.25f);
        emissive.x *= totalGain;
        emissive.y *= totalGain;
        emissive.z *= totalGain;
    }

    return emissive;
}

static float EstimateTriangleTexturedEmissiveLuminanceCPU(const Triangle &tri,
                                                          const SceneMetaMaterial &material,
                                                          const SceneMetaResources *metaRes)
{
    constexpr std::array<std::array<float, 3>, 4> kSampleBary = {{
        {0.33333334f, 0.33333334f, 0.33333334f},
        {0.60f, 0.20f, 0.20f},
        {0.20f, 0.60f, 0.20f},
        {0.20f, 0.20f, 0.60f},
    }};

    float accum = 0.0f;
    for (const auto &b : kSampleBary)
    {
        const Vec2 uvBase = FractCPU(TriangleUvAtBary(tri, material.baseColorUvSet, b[0], b[1], b[2]));
        const Vec2 uvEmission = FractCPU(TriangleUvAtBary(tri, material.emissionUvSet, b[0], b[1], b[2]));
        const Vec3 emissive = EvaluateTriangleEmissionCPU(tri, material, metaRes, uvBase, uvEmission, 1.0f);
        accum += LuminanceCPU(emissive);
    }
    return accum / (float)kSampleBary.size();
}

static inline float TriangleAreaCPU(const Triangle &t)
{
    const float e1x = t.v1.x - t.v0.x;
    const float e1y = t.v1.y - t.v0.y;
    const float e1z = t.v1.z - t.v0.z;

    const float e2x = t.v2.x - t.v0.x;
    const float e2y = t.v2.y - t.v0.y;
    const float e2z = t.v2.z - t.v0.z;

    const float crx = e1y * e2z - e1z * e2y;
    const float cry = e1z * e2x - e1x * e2z;
    const float crz = e1x * e2y - e1y * e2x;

    const float crLenSq = crx * crx + cry * cry + crz * crz;
    return 0.5f * std::sqrt(std::max(0.0f, crLenSq));
}
static inline Vec3 TransformPointCPU(const float m[12], const Vec3 &p)
{
    return Vec3{
        m[0] * p.x + m[1] * p.y + m[2]  * p.z + m[3],
        m[4] * p.x + m[5] * p.y + m[6]  * p.z + m[7],
        m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
    };
}


static std::vector<EmissiveTriangleGPUCPU> BuildEmissiveTriangleTable(const std::vector<Triangle> &tris,
                                                                  const std::vector<SceneInstanceGPU> &instances,
                                                                  const SceneMetaResources *metaRes)
{
    std::vector<EmissiveTriangleGPUCPU> out;
    if (tris.empty() || instances.empty())
        return out;

    out.reserve(tris.size());
    double totalWeight = 0.0;
    std::vector<double> weights;
    weights.reserve(tris.size());

    std::vector<float> triangleEmissiveLuma(tris.size(), 0.0f);
    for (std::size_t triIdx = 0; triIdx < tris.size(); ++triIdx)
    {
        const Triangle &tri = tris[triIdx];
        const float rawLum = std::max(0.0f, LuminanceCPU(tri.emission));
        if (rawLum <= 1.0e-8f)
            continue;

        float texturedLum = rawLum;
        const int matId = tri.materialIndex;
        if (metaRes &&
            matId >= 0 &&
            (std::size_t)matId < metaRes->materialsPBR.size())
        {
            const SceneMetaMaterial &material = metaRes->materialsPBR[(std::size_t)matId];
            if (material.emissionTexIndex >= 0 || material.specialModel != 0)
            {
                texturedLum = EstimateTriangleTexturedEmissiveLuminanceCPU(tri, material, metaRes);
                texturedLum = std::max(texturedLum, rawLum * 0.05f);
            }
        }

        triangleEmissiveLuma[triIdx] = texturedLum;
    }

    for (std::size_t instIdx = 0; instIdx < instances.size(); ++instIdx)
    {
        const SceneInstanceGPU &inst = instances[instIdx];
        for (std::size_t triIdx = 0; triIdx < tris.size(); ++triIdx)
        {
            const Triangle &t = tris[triIdx];
            const float lum  = triangleEmissiveLuma[triIdx];
            if (lum <= 1.0e-8f)
                continue;

            const Vec3 w0 = TransformPointCPU(inst.objectToWorld, t.v0);
            const Vec3 w1 = TransformPointCPU(inst.objectToWorld, t.v1);
            const Vec3 w2 = TransformPointCPU(inst.objectToWorld, t.v2);

            Triangle worldTri = t;
            worldTri.v0 = w0;
            worldTri.v1 = w1;
            worldTri.v2 = w2;

            const float area = TriangleAreaCPU(worldTri);
            const double weight = static_cast<double>(area) * static_cast<double>(lum);
            if (weight <= 1.0e-12)
                continue;

            EmissiveTriangleGPUCPU e{};
            e.triIndex = static_cast<uint32_t>(triIdx);
            e.instanceIndex = static_cast<uint32_t>(instIdx);
            e.area = std::max(area, 1.0e-8f);
            out.push_back(e);
            weights.push_back(weight);
            totalWeight += weight;
        }
    }

    if (out.empty() || totalWeight <= 0.0)
        return out;

    double cdf = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const float pdf = static_cast<float>(weights[i] / totalWeight);
        cdf += weights[i] / totalWeight;
        out[i].selectionPdf = std::max(pdf, 1.0e-8f);
        out[i].cdf = static_cast<float>((i + 1 == out.size()) ? 1.0 : std::min(cdf, 1.0));
    }

    return out;
}

static std::vector<AirDustVolumeGPUCPU> BuildAirDustVolumeTable(const SceneMetaResources *metaRes)
{
    std::vector<AirDustVolumeGPUCPU> out;
    if (!metaRes)
        return out;

    out.reserve(metaRes->airDustVolumes.size());
    for (const SceneMetaAirDustVolume &src : metaRes->airDustVolumes)
    {
        if (src.linkedLightIntensity <= 0.0f)
            continue;

        AirDustVolumeGPUCPU v{};
        v.centerX = src.position.x;
        v.centerY = src.position.y;
        v.centerZ = src.position.z;
        v.extentX = std::max(src.extent.x, 1.0f);
        v.extentY = std::max(src.extent.y, 1.0f);
        v.extentZ = std::max(src.extent.z, 1.0f);
        v.lightPosX = src.linkedLightPosition.x;
        v.lightPosY = src.linkedLightPosition.y;
        v.lightPosZ = src.linkedLightPosition.z;
        v.lightColorX = src.linkedLightColor.x;
        v.lightColorY = src.linkedLightColor.y;
        v.lightColorZ = src.linkedLightColor.z;
        v.lightIntensity = src.linkedLightIntensity;
        v.lightRadius = std::max(src.linkedLightRadius,
                                 std::max(v.extentX, std::max(v.extentY, v.extentZ)) * 1.5f);

        const float normalizedIntensity = std::sqrt(std::max(src.linkedLightIntensity, 0.0f) / 2500.0f);
        v.density = std::clamp(0.08f + normalizedIntensity * 0.10f, 0.06f, 0.24f);
        v.anisotropy = 0.45f;
        out.push_back(v);
    }

    return out;
}

static id<MTLTexture> CreateRGBA32FloatTexture(id<MTLDevice> dev,
                                               int width,
                                               int height,
                                               MTLTextureUsage usage)
{
    if (!dev || width <= 0 || height <= 0)
        return nil;

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                           width:(NSUInteger)width
                                                          height:(NSUInteger)height
                                                       mipmapped:NO];
    desc.usage = usage;
    return [dev newTextureWithDescriptor:desc];
}

static bool EnsurePostProcessTextures(int width, int height)
{
    if (!g_device || width <= 0 || height <= 0)
        return false;

    const bool sizeMatches =
        g_hdrTexture &&
        g_albedoTexture &&
        g_normalTexture &&
        g_bloomTextureA &&
        g_bloomTextureB &&
        g_outTexture &&
        g_postTextureWidth == width &&
        g_postTextureHeight == height;
    if (sizeMatches)
        return true;

    g_hdrTexture = CreateRGBA32FloatTexture(g_device,
                                            width,
                                            height,
                                            MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    g_albedoTexture = CreateRGBA32FloatTexture(g_device,
                                               width,
                                               height,
                                               MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    g_normalTexture = CreateRGBA32FloatTexture(g_device,
                                               width,
                                               height,
                                               MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    g_bloomTextureA = CreateRGBA32FloatTexture(g_device,
                                               width,
                                               height,
                                               MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    g_bloomTextureB = CreateRGBA32FloatTexture(g_device,
                                               width,
                                               height,
                                               MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    g_outTexture = CreateRGBA32FloatTexture(g_device,
                                            width,
                                            height,
                                            MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    if (!g_hdrTexture || !g_albedoTexture || !g_normalTexture ||
        !g_bloomTextureA || !g_bloomTextureB || !g_outTexture)
        return false;

    g_postTextureWidth = width;
    g_postTextureHeight = height;
    return true;
}


static std::vector<DecalGPU> BuildDecalTable(const SceneMetaResources *metaRes)
{
    std::vector<DecalGPU> out;
    if (!metaRes || metaRes->decals.empty() || metaRes->materials.empty())
        return out;

    const size_t baseCount = std::min(metaRes->baseColorTextures.size(), (size_t)kMaxAllTextures);
    const size_t remaining = (size_t)kMaxAllTextures - baseCount;
    const size_t linCount = std::min(metaRes->linearTextures.size(), remaining);

    auto mapSRGB = [&](int idx) -> int32_t
    {
        if (idx < 0 || (size_t)idx >= baseCount)
            return -1;
        return idx;
    };

    auto mapLinear = [&](int idx) -> int32_t
    {
        if (idx < 0 || (size_t)idx >= linCount)
            return -1;
        return (int32_t)(baseCount + (size_t)idx);
    };

    auto mapAny = [&](int idx, bool isLinear) -> int32_t
    {
        return isLinear ? mapLinear(idx) : mapSRGB(idx);
    };

    out.reserve(metaRes->decals.size());
    for (const auto &sd : metaRes->decals)
    {
        if (sd.materialIndex < 0 || (size_t)sd.materialIndex >= metaRes->materials.size())
            continue;

        const SceneMetaMaterial &sm = metaRes->materials[(size_t)sd.materialIndex];
        const int32_t baseTex = mapSRGB(sm.baseColorTexIndex);
        if (baseTex < 0)
            continue;

        DecalGPU d{};
        d.posX = sd.position.x; d.posY = sd.position.y; d.posZ = sd.position.z;
        d.sizeX = std::max(0.5f * sd.size.x, 1.0f); // UE DecalSize.X = depth
        d.axisXx = sd.axisX.x; d.axisXy = sd.axisX.y; d.axisXz = sd.axisX.z;
        d.sizeY = std::max(0.5f * sd.size.y, 1.0f); // UE DecalSize.Y = width
        d.axisYx = sd.axisY.x; d.axisYy = sd.axisY.y; d.axisYz = sd.axisY.z;
        d.sizeZ = std::max(0.5f * sd.size.z, 1.0f); // UE DecalSize.Z = height
        d.axisZx = sd.axisZ.x; d.axisZy = sd.axisZ.y; d.axisZz = sd.axisZ.z;
        d.opacity = std::clamp(sm.opacity * 1.30f, 0.0f, 2.0f);
        d.baseColorTexIndex = baseTex;
        d.ormTexIndex = mapLinear(sm.ormTexIndex);
        d.roughnessTexIndex = mapLinear(sm.roughnessTexIndex);
        d.normalTexIndex = mapLinear(sm.normalTexIndex);
        d.opacityTexIndex = mapAny(sm.decalOpacityTexIndex, sm.decalOpacityTexIsLinear);
        d.detailTexIndex = mapAny(sm.decalDetailTexIndex, sm.decalDetailTexIsLinear);
        d.flags = 0;
        d._pad0 = 0;
        d.baseColorX = sm.baseColor.x;
        d.baseColorY = sm.baseColor.y;
        d.baseColorZ = sm.baseColor.z;
        d.roughnessBias = sm.decalRoughnessBias;
        d.tilingU = sm.decalTilingU;
        d.tilingV = sm.decalTilingV;
        d.opacityPower = std::max(sm.decalOpacityPower, 0.25f);
        d.normalIntensity = std::clamp(sm.decalNormalIntensity, 0.0f, 8.0f);
        out.push_back(d);
    }

    return out;
}

static bool EnsureSceneBuffersUploaded(const std::vector<BVHNode> &tlasNodes,
                                       const std::vector<BVHNode> &meshNodes,
                                       const std::vector<Triangle> &tris,
                                       const std::vector<SceneInstanceGPU> &instances,
                                       const std::vector<Light> &lights,
                                       std::uint64_t sceneRevision,
                                       int rootIndex,
                                       TextureFrameProfile &profile)
{
    const bool sceneChanged =
        (g_cachedSceneRevision != sceneRevision) ||
        !g_tlasBuffer ||
        !g_triBuffer ||
        !g_meshNodeBuffer ||
        !g_instanceBuffer ||
        !g_nodeCountBuffer ||
        !g_rootIndexBuffer ||
        !g_meshNodeCountBuffer ||
        !g_instanceCountBuffer ||
        !g_lightBuffer ||
        !g_lightCountBuffer ||
        !g_emissiveTriangleBuffer ||
        !g_emissiveTriangleCountBuffer;

    if (!sceneChanged)
        return true;

    const auto geomStart = ProfileClock::now();
    const BVHNode dummyNode{};
    const Triangle dummyTriangle{};
    const SceneInstanceGPU dummyInstance{};

    if (!UploadSharedVector(g_device, g_tlasBuffer, tlasNodes, dummyNode) ||
        !UploadSharedVector(g_device, g_triBuffer, tris, dummyTriangle) ||
        !UploadSharedVector(g_device, g_meshNodeBuffer, meshNodes, dummyNode) ||
        !UploadSharedVector(g_device, g_instanceBuffer, instances, dummyInstance))
    {
        std::cerr << "Metal: failed to create persistent geometry buffers (texture)\n";
        return false;
    }

    const uint32_t nodeCount = static_cast<uint32_t>(tlasNodes.size());
    const uint32_t meshNodeCount = static_cast<uint32_t>(meshNodes.size());
    const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
    if (!UploadSharedValue(g_device, g_nodeCountBuffer, nodeCount) ||
        !UploadSharedValue(g_device, g_rootIndexBuffer, rootIndex) ||
        !UploadSharedValue(g_device, g_meshNodeCountBuffer, meshNodeCount) ||
        !UploadSharedValue(g_device, g_instanceCountBuffer, instanceCount))
    {
        std::cerr << "Metal: failed to create persistent small geometry buffers\n";
        return false;
    }
    profile.geometryBuffersMs = ToMilliseconds(ProfileClock::now() - geomStart);

    const auto lightStart = ProfileClock::now();
    std::vector<LightGPU> gpuLights;
    gpuLights.reserve(lights.size());
    for (const Light &src : lights)
    {
        LightGPU dst{};
        dst.type      = static_cast<int>(src.type);
        dst.flags     = src.castShadows ? 1u : 0u;
        dst.position  = src.position;
        dst.direction = src.direction;
        dst.color     = src.color;
        dst.intensity = src.intensity;
        dst.radius    = src.radius;
        dst.sourceLength = src.sourceLength;
        dst.softSourceRadius = src.softSourceRadius;
        dst.spotSize  = src.spotSize;
        dst.spotBlend = src.spotBlend;
        dst.attenuationRadius = src.attenuationRadius;
        dst.ownerId = src.ownerId;
        gpuLights.push_back(dst);
    }

    const LightGPU dummyLight{};
    const uint32_t lightCount = static_cast<uint32_t>(gpuLights.size());
    if (!UploadSharedVector(g_device, g_lightBuffer, gpuLights, dummyLight) ||
        !UploadSharedValue(g_device, g_lightCountBuffer, lightCount))
    {
        std::cerr << "Metal: failed to create persistent light buffers\n";
        return false;
    }
    profile.lightUploadMs = ToMilliseconds(ProfileClock::now() - lightStart);

    const auto emissiveStart = ProfileClock::now();
    const std::vector<EmissiveTriangleGPUCPU> emissiveTrianglesCPU =
        BuildEmissiveTriangleTable(tris, instances, g_loadedMetaRes);
    const EmissiveTriangleGPUCPU dummyEmissive{};
    const uint32_t emissiveTriangleCount =
        static_cast<uint32_t>(emissiveTrianglesCPU.size());
    if (!UploadSharedVector(g_device, g_emissiveTriangleBuffer, emissiveTrianglesCPU, dummyEmissive) ||
        !UploadSharedValue(g_device, g_emissiveTriangleCountBuffer, emissiveTriangleCount))
    {
        std::cerr << "Metal: failed to create persistent emissive buffers\n";
        return false;
    }
    profile.emissiveDecalMs = ToMilliseconds(ProfileClock::now() - emissiveStart);

    g_cachedSceneRevision = sceneRevision;
    return true;
}

static bool EnsureDecalBufferUploaded(const SceneMetaResources *metaRes,
                                      TextureFrameProfile &profile)
{
    if (g_cachedDecalMetaRes == metaRes && g_decalBuffer && g_decalCountBuffer)
        return true;

    const auto decalStart = ProfileClock::now();
    const std::vector<DecalGPU> decalsCPU = BuildDecalTable(metaRes);
    const DecalGPU dummyDecal{};
    const uint32_t decalCount = static_cast<uint32_t>(decalsCPU.size());
    if (!UploadSharedVector(g_device, g_decalBuffer, decalsCPU, dummyDecal) ||
        !UploadSharedValue(g_device, g_decalCountBuffer, decalCount))
    {
        std::cerr << "Metal: failed to create persistent decal buffers\n";
        return false;
    }

    profile.emissiveDecalMs += ToMilliseconds(ProfileClock::now() - decalStart);
    g_cachedDecalMetaRes = metaRes;
    return true;
}

static bool EnsureAirDustBufferUploaded(const SceneMetaResources *metaRes,
                                        TextureFrameProfile &profile)
{
    if (g_cachedAirDustMetaRes == metaRes && g_airDustVolumeBuffer && g_airDustVolumeCountBuffer)
        return true;

    const auto airDustStart = ProfileClock::now();
    const std::vector<AirDustVolumeGPUCPU> airDustCPU = BuildAirDustVolumeTable(metaRes);
    const AirDustVolumeGPUCPU dummyAirDust{};
    const uint32_t airDustCount = static_cast<uint32_t>(airDustCPU.size());
    if (!UploadSharedVector(g_device, g_airDustVolumeBuffer, airDustCPU, dummyAirDust) ||
        !UploadSharedValue(g_device, g_airDustVolumeCountBuffer, airDustCount))
    {
        std::cerr << "Metal: failed to create persistent air-dust buffers\n";
        return false;
    }

    profile.postParamsMs += ToMilliseconds(ProfileClock::now() - airDustStart);
    g_cachedAirDustMetaRes = metaRes;
    return true;
}

static PostProcessParamsCPU BuildPostProcessParams(const SceneMetaResources *metaRes,
                                                   const CameraDataCPU &cameraCPU,
                                                   int width,
                                                   int height,
                                                   uint32_t previewDispatchCount,
                                                   MetalAccumulationMode accumulationMode)
{
    PostProcessParamsCPU pp{};

    // Calibrated fallback for scenes without exported post-process.
    pp.exposure = 1.05f;
    pp.bloomIntensity = 0.0f;
    pp.bloomThreshold = 0.25f;
    pp.vignetteIntensity = 0.0f;
    pp.chromaticAberration = 0.0f;
    pp.filmGrainIntensity = 0.0f;
    pp.filmSlope = 0.95f;
    pp.filmToe = 0.40f;
    pp.filmShoulder = 0.35f;
    pp.filmBlackClip = 0.0f;
    pp.filmWhiteClip = 0.12f;

    pp.fogDensity = 0.0f;
    pp.fogHeightFalloff = 0.0f;
    pp.fogScatteringG = 0.0f;
    pp.fogColorX = 1.0f;
    pp.fogColorY = 1.0f;
    pp.fogColorZ = 1.0f;
    pp.fogExtinctionScale = 1.0f;
    pp.fogAlbedoX = 1.0f;
    pp.fogAlbedoY = 1.0f;
    pp.fogAlbedoZ = 1.0f;
    pp.volumetricFog = 0.0f;

    pp.colorSaturationX = 1.0f;
    pp.colorSaturationY = 1.0f;
    pp.colorSaturationZ = 1.0f;
    pp.shadowLift = 0.010f;
    pp.fogStartDistance = 0.0f;
    pp.fogMaxOpacity = 1.0f;
    pp.fogHeightZ = 0.0f;
    pp.worldUnitToMeters = 1.0f;

    if (metaRes)
    {
        if (metaRes->hasPostProcess)
        {
            const SceneMetaPostProcess &src = metaRes->postProcess;
            const float exposureBias = std::clamp(src.autoExposureBias, -4.0f, 4.0f);
            pp.exposure = std::exp2(exposureBias);

            pp.bloomIntensity = std::max(0.0f, src.bloomIntensity);
            pp.bloomThreshold = src.bloomThreshold;

            pp.vignetteIntensity = std::clamp(src.vignetteIntensity, 0.0f, 1.0f);
            pp.chromaticAberration = std::max(0.0f, src.chromaticAberration);
            pp.filmGrainIntensity = std::max(0.0f, src.filmGrainIntensity);
            pp.filmSlope = std::max(0.05f, src.filmSlope);
            pp.filmToe = std::max(0.0f, src.filmToe);
            pp.filmShoulder = std::max(0.0f, src.filmShoulder);
            pp.filmBlackClip = std::max(0.0f, src.filmBlackClip);
            pp.filmWhiteClip = std::max(0.0f, src.filmWhiteClip);

            pp.colorSaturationX = std::clamp(src.colorSaturation.x, 0.0f, 2.0f);
            pp.colorSaturationY = std::clamp(src.colorSaturation.y, 0.0f, 2.0f);
            pp.colorSaturationZ = std::clamp(src.colorSaturation.z, 0.0f, 2.0f);
        }

        if (metaRes->hasFog)
        {
            const SceneMetaFog &src = metaRes->fog;
            pp.fogDensity = std::max(0.0f, src.fogDensity);
            pp.fogHeightFalloff = std::max(0.0f, src.heightFalloff);
            pp.fogScatteringG = std::clamp(src.scatteringG, -0.95f, 0.95f);
            pp.fogColorX = src.inscatteringColor.x;
            pp.fogColorY = src.inscatteringColor.y;
            pp.fogColorZ = src.inscatteringColor.z;
            pp.fogExtinctionScale = std::max(0.0f, src.extinctionScale);
            pp.fogAlbedoX = src.volumetricAlbedo.x;
            pp.fogAlbedoY = src.volumetricAlbedo.y;
            pp.fogAlbedoZ = src.volumetricAlbedo.z;
            pp.volumetricFog = src.volumetricFog ? 1.0f : 0.0f;
            pp.fogStartDistance = std::max(0.0f, src.startDistance);
            pp.fogMaxOpacity = std::clamp(src.maxOpacity, 0.0f, 1.0f);
            pp.fogHeightZ = src.heightReferenceZ;
        }

        pp.worldUnitToMeters = std::max(metaRes->worldUnitToMeters, 1.0e-4f);
    }

    pp.nearPlane = std::max(1.0e-4f, cameraCPU.nearPlane);
    pp.farPlane  = std::max(pp.nearPlane + 1.0e-3f, cameraCPU.farPlane);
    pp.time = (accumulationMode == MetalAccumulationMode::FinalStill)
            ? 0.0f
            : static_cast<float>(previewDispatchCount) * (1.0f / 60.0f);
    pp.width = static_cast<float>(width);
    pp.height = static_cast<float>(height);
    pp._pad0 = 0.0f;
    return pp;
}

void ResetMetalAccumulation()
{
    ResetMetalAccumulationState();
    g_textureProfileTotals = {};
    g_textureProfileCallIndex = 0;
}

// Инициализация Metal: устройство, очередь, пайплайны
bool InitMetalRenderer()
{
    @autoreleasepool
    {
        if (g_device && g_queue && g_pipelineTexture && g_pipelineBloomExtract &&
            g_pipelineBloomBlurH && g_pipelineBloomBlurV && g_pipelinePostProcess)
        {
            return true;
        }

        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device)
        {
            std::cerr << "Metal: failed to create device\n";
            return false;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue)
        {
            std::cerr << "Metal: failed to create command queue\n";
            return false;
        }

        NSError *error = nil;
        NSString *libPath = @"build/RayTrace.metallib";
        NSURL    *libURL  = [NSURL fileURLWithPath:libPath];

        id<MTLLibrary> lib = [g_device newLibraryWithURL:libURL error:&error];
        if (!lib || error)
        {
            std::cerr << "Metal: failed to load RayTrace.metallib from path: "
                      << [libPath UTF8String] << "\n";
            if (error)
                std::cerr << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> funcTex = [lib newFunctionWithName:@"RayTraceTextureKernel"];
        if (!funcTex)
        {
            std::cerr << "Metal: function not found in metallib: RayTraceTextureKernel" << std::endl;
            return false;
        }
        error = nil;
        g_pipelineTexture = [g_device newComputePipelineStateWithFunction:funcTex error:&error];

        if (!g_pipelineTexture || error)
        {
            std::cerr << "Metal: failed to create compute pipeline state (RayTraceTextureKernel)\n";
            if (error)
                std::cerr << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        auto makePipeline = [&](NSString *name) -> id<MTLComputePipelineState>
        {
            NSError *localError = nil;
            id<MTLFunction> func = [lib newFunctionWithName:name];
            if (!func)
            {
                std::cerr << "Metal: function not found in metallib: " << [name UTF8String] << std::endl;
                return (id<MTLComputePipelineState>)nil;
            }
            id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:func error:&localError];
            if (!pipeline || localError)
            {
                std::cerr << "Metal: failed to create compute pipeline state: " << [name UTF8String] << std::endl;
                if (localError)
                    std::cerr << [[localError localizedDescription] UTF8String] << std::endl;
                return (id<MTLComputePipelineState>)nil;
            }
            return pipeline;
        };

        g_pipelineBloomExtract = makePipeline(@"BloomExtractKernel");
        g_pipelineBloomBlurH   = makePipeline(@"BloomBlurKernelH");
        g_pipelineBloomBlurV   = makePipeline(@"BloomBlurKernelV");
        g_pipelinePostProcess  = makePipeline(@"PostProcessKernel");

        if (!g_pipelineBloomExtract || !g_pipelineBloomBlurH || !g_pipelineBloomBlurV || !g_pipelinePostProcess)
            return false;

        return true;
    }
}

bool PreloadMetalSceneResources(const SceneMetaResources *metaRes)
{
    if (!InitMetalRenderer())
        return false;

    if (!EnsureMaterialAndTexturesLoaded(metaRes))
        return false;

    TextureFrameProfile preloadProfile{};
    if (!EnsureDecalBufferUploaded(metaRes, preloadProfile))
        return false;
    if (!EnsureAirDustBufferUploaded(metaRes, preloadProfile))
        return false;

    return true;
}

// ВАЖНО: добавили параметр metaRes, чтобы передать список материалов/текстур из SceneMetaLoader.
bool RenderFrameMetalTexture(const std::vector<BVHNode>   &tlasNodes,
                             const std::vector<BVHNode>   &meshNodes,
                             const std::vector<Triangle> &tris,
                             const std::vector<SceneInstanceGPU> &instances,
                             const std::vector<Light>    &lights,
                             std::uint64_t                sceneRevision,
                             MetalAccumulationMode        accumulationMode,
                             int                          rootIndex,
                             const CameraDataCPU         &cameraCPU,
                             const SceneMetaResources    *metaRes,
                             std::vector<Vec3>           &framebuffer)
{
    TextureFrameProfile profile{};
    profile.callIndex = ++g_textureProfileCallIndex;
    profile.accumulatedSamples = g_accumulatedSampleCount;
    profile.samplesThisDispatch = EffectiveSamplesPerDispatch(cameraCPU);
    profile.accumulationMode = accumulationMode;
    profile.prototypeTriangles = static_cast<uint32_t>(std::min<std::size_t>(tris.size(), UINT32_MAX));
    profile.totalInstances = static_cast<uint32_t>(std::min<std::size_t>(instances.size(), UINT32_MAX));
    profile.tlasNodeCount = static_cast<uint32_t>(std::min<std::size_t>(tlasNodes.size(), UINT32_MAX));
    profile.blasNodeCount = static_cast<uint32_t>(std::min<std::size_t>(meshNodes.size(), UINT32_MAX));
    const auto totalStart = ProfileClock::now();

    if (!g_device || !g_queue || !g_pipelineTexture || !g_pipelineBloomExtract || !g_pipelineBloomBlurH ||
        !g_pipelineBloomBlurV || !g_pipelinePostProcess)
    {
        const auto initStart = ProfileClock::now();
        if (!InitMetalRenderer())
            return false;
        profile.initMetalMs = ToMilliseconds(ProfileClock::now() - initStart);
    }

    {
        const auto materialsStart = ProfileClock::now();
        if (!EnsureMaterialAndTexturesLoaded(metaRes))
            return false;
        profile.ensureMaterialsMs = ToMilliseconds(ProfileClock::now() - materialsStart);
    }

    const int width  = cameraCPU.width;
    const int height = cameraCPU.height;
    if (width <= 0 || height <= 0)
        return false;

    framebuffer.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    @autoreleasepool
    {
        const std::uint64_t accumulationStateHash = ComputeAccumulationStateHash(cameraCPU,
                                                                                 width,
                                                                                 height,
                                                                                 sceneRevision,
                                                                                 lights,
                                                                                 metaRes,
                                                                                 accumulationMode);
        if (g_accumulationStateHash != accumulationStateHash)
        {
            ResetMetalAccumulationState();
            g_accumulationStateHash = accumulationStateHash;
            profile.accumulatedSamples = 0;
        }

        const auto accumStart = ProfileClock::now();
        if (!g_accumTexture ||
            g_accumTexture.width  != static_cast<NSUInteger>(width) ||
            g_accumTexture.height != static_cast<NSUInteger>(height))
        {
            g_accumTexture = CreateRGBA32FloatTexture(g_device,
                                                      width,
                                                      height,
                                                      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
            g_accumulatedSampleCount = 0;
            g_accumulationDispatchCount = 0;
            profile.accumulatedSamples = 0;
        }
        profile.accumTextureMs = ToMilliseconds(ProfileClock::now() - accumStart);

        const auto interTexStart = ProfileClock::now();
        if (!EnsurePostProcessTextures(width, height))
        {
            std::cerr << "Metal: failed to create post-process textures\n";
            return false;
        }
        profile.intermediateTexturesMs = ToMilliseconds(ProfileClock::now() - interTexStart);

        if (!EnsureSceneBuffersUploaded(tlasNodes,
                                        meshNodes,
                                        tris,
                                        instances,
                                        lights,
                                        sceneRevision,
                                        rootIndex,
                                        profile))
        {
            return false;
        }

        const uint32_t sampleBaseIndex = g_accumulatedSampleCount;

        if (!g_materialPBRBuffer || !g_materialPBRCountBuffer)
        {
            const uint32_t zero = 0u;
            MaterialGPU_PBR dummy{};
            g_materialPBRBuffer = [g_device newBufferWithBytes:&dummy
                                                        length:sizeof(MaterialGPU_PBR)
                                                       options:MTLResourceStorageModeShared];
            g_materialPBRCountBuffer = [g_device newBufferWithBytes:&zero
                                                             length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        }

        const auto postParamsStart = ProfileClock::now();
        if (!EnsureDecalBufferUploaded(metaRes, profile))
            return false;
        if (!EnsureAirDustBufferUploaded(metaRes, profile))
            return false;

        PostProcessParamsCPU pp = BuildPostProcessParams(metaRes,
                                                        cameraCPU,
                                                        width,
                                                        height,
                                                        g_accumulationDispatchCount,
                                                        accumulationMode);
        const std::array<uint32_t, 2> imageSize = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        const auto smallStart = ProfileClock::now();
        if (!UploadSharedValue(g_device, g_camBuffer, cameraCPU) ||
            !UploadSharedValue(g_device, g_imgSizeBuffer, imageSize) ||
            !UploadSharedValue(g_device, g_sampleCountBuffer, sampleBaseIndex) ||
            !UploadSharedValue(g_device, g_postProcessBuffer, pp))
        {
            std::cerr << "Metal: failed to update persistent small/frame buffers\n";
            return false;
        }
        profile.smallBuffersMs = ToMilliseconds(ProfileClock::now() - smallStart);
        profile.postParamsMs = ToMilliseconds(ProfileClock::now() - postParamsStart);

        auto dispatchPass = [&](id<MTLComputePipelineState> pipeline, id<MTLComputeCommandEncoder> enc)
        {
            [enc setComputePipelineState:pipeline];
            MTLSize gridSize = MTLSizeMake(width, height, 1);
            NSUInteger tgW = 8;
            NSUInteger tgH = 8;
            NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
            if (tgW * tgH > maxThreads)
            {
                tgW = maxThreads;
                tgH = 1;
            }
            MTLSize threadsPerGroup = MTLSizeMake(tgW, tgH, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
        };

        const auto encodeStart = ProfileClock::now();
        id<MTLCommandBuffer> pathTraceCmd = [g_queue commandBuffer];
        id<MTLCommandBuffer> bloomExtractCmd = [g_queue commandBuffer];
        id<MTLCommandBuffer> blurHCmd = [g_queue commandBuffer];
        id<MTLCommandBuffer> blurVCmd = [g_queue commandBuffer];
        id<MTLCommandBuffer> finalCompositeCmd = [g_queue commandBuffer];

        {
            id<MTLComputeCommandEncoder> enc = [pathTraceCmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipelineTexture];
            [enc setBuffer:g_tlasBuffer                offset:0 atIndex:0];
            [enc setBuffer:g_triBuffer                 offset:0 atIndex:1];
            [enc setBuffer:g_nodeCountBuffer           offset:0 atIndex:2];
            [enc setBuffer:g_rootIndexBuffer           offset:0 atIndex:3];
            [enc setBuffer:g_camBuffer                 offset:0 atIndex:4];
            [enc setBuffer:g_imgSizeBuffer             offset:0 atIndex:5];
            [enc setBuffer:g_lightBuffer               offset:0 atIndex:7];
            [enc setBuffer:g_lightCountBuffer          offset:0 atIndex:8];
            [enc setBuffer:g_sampleCountBuffer         offset:0 atIndex:9];
            if (g_materialBuffer)
                [enc setBuffer:g_materialBuffer offset:0 atIndex:10];
            if (g_materialCountBuffer)
                [enc setBuffer:g_materialCountBuffer offset:0 atIndex:11];
            [enc setBuffer:g_meshNodeBuffer            offset:0 atIndex:12];
            [enc setBuffer:g_instanceBuffer            offset:0 atIndex:13];
            [enc setBuffer:g_materialPBRBuffer         offset:0 atIndex:14];
            [enc setBuffer:g_materialPBRCountBuffer    offset:0 atIndex:15];
            [enc setBuffer:g_emissiveTriangleBuffer      offset:0 atIndex:16];
            [enc setBuffer:g_emissiveTriangleCountBuffer offset:0 atIndex:17];
            [enc setBuffer:g_decalBuffer                 offset:0 atIndex:18];
            [enc setBuffer:g_decalCountBuffer            offset:0 atIndex:19];
            [enc setBuffer:g_meshNodeCountBuffer         offset:0 atIndex:20];
            [enc setBuffer:g_instanceCountBuffer         offset:0 atIndex:21];
            [enc setBuffer:g_postProcessBuffer           offset:0 atIndex:22];
            [enc setBuffer:g_airDustVolumeBuffer         offset:0 atIndex:23];
            [enc setBuffer:g_airDustVolumeCountBuffer    offset:0 atIndex:24];

            [enc setTexture:g_accumTexture atIndex:0];
            [enc setTexture:g_hdrTexture   atIndex:1];
            [enc setTexture:g_albedoTexture atIndex:2];
            [enc setTexture:g_normalTexture atIndex:3];

            const uint32_t texCount = static_cast<uint32_t>(g_baseColorTextures.size());
            for (uint32_t i = 0; i < texCount; ++i)
                [enc setTexture:g_baseColorTextures[i] atIndex:(NSUInteger)(4 + i)];

            dispatchPass(g_pipelineTexture, enc);
            [enc endEncoding];
        }
        [pathTraceCmd commit];

        {
            id<MTLComputeCommandEncoder> enc = [bloomExtractCmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipelineBloomExtract];
            [enc setBuffer:g_postProcessBuffer offset:0 atIndex:0];
            [enc setTexture:g_hdrTexture    atIndex:0];
            [enc setTexture:g_bloomTextureA atIndex:1];
            dispatchPass(g_pipelineBloomExtract, enc);
            [enc endEncoding];
        }
        [bloomExtractCmd commit];

        {
            id<MTLComputeCommandEncoder> enc = [blurHCmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipelineBloomBlurH];
            [enc setBuffer:g_postProcessBuffer offset:0 atIndex:0];
            [enc setTexture:g_bloomTextureA atIndex:0];
            [enc setTexture:g_bloomTextureB atIndex:1];
            dispatchPass(g_pipelineBloomBlurH, enc);
            [enc endEncoding];
        }
        [blurHCmd commit];

        {
            id<MTLComputeCommandEncoder> enc = [blurVCmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipelineBloomBlurV];
            [enc setBuffer:g_postProcessBuffer offset:0 atIndex:0];
            [enc setTexture:g_bloomTextureB atIndex:0];
            [enc setTexture:g_bloomTextureA atIndex:1];
            dispatchPass(g_pipelineBloomBlurV, enc);
            [enc endEncoding];
        }
        [blurVCmd commit];

        {
            id<MTLComputeCommandEncoder> enc = [finalCompositeCmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipelinePostProcess];
            [enc setBuffer:g_postProcessBuffer offset:0 atIndex:0];
            [enc setBuffer:g_airDustVolumeBuffer offset:0 atIndex:1];
            [enc setBuffer:g_airDustVolumeCountBuffer offset:0 atIndex:2];
            [enc setBuffer:g_camBuffer offset:0 atIndex:3];
            [enc setTexture:g_hdrTexture    atIndex:0];
            [enc setTexture:g_bloomTextureA atIndex:1];
            [enc setTexture:g_outTexture    atIndex:2];
            dispatchPass(g_pipelinePostProcess, enc);
            [enc endEncoding];
        }
        [finalCompositeCmd commit];
        profile.encodeMs = ToMilliseconds(ProfileClock::now() - encodeStart);

        const auto waitStart = ProfileClock::now();
        [finalCompositeCmd waitUntilCompleted];
        profile.waitMs = ToMilliseconds(ProfileClock::now() - waitStart);

        profile.primaryDepthGpuMs = 0.0;
        profile.pathTraceGpuMs = SafeGpuTimeMs(pathTraceCmd);
        profile.bloomExtractGpuMs = SafeGpuTimeMs(bloomExtractCmd);
        profile.blurHGpuMs = SafeGpuTimeMs(blurHCmd);
        profile.blurVGpuMs = SafeGpuTimeMs(blurVCmd);
        profile.finalCompositeGpuMs = SafeGpuTimeMs(finalCompositeCmd);
        profile.gpuMs = profile.primaryDepthGpuMs +
                        profile.pathTraceGpuMs +
                        profile.bloomExtractGpuMs +
                        profile.blurHGpuMs +
                        profile.blurVGpuMs +
                        profile.finalCompositeGpuMs;

        if (profile.samplesThisDispatch > 0u)
        {
            const uint32_t remainingSamples = kAccumulatedSampleCountMax - g_accumulatedSampleCount;
            g_accumulatedSampleCount += std::min(profile.samplesThisDispatch, remainingSamples);
        }
        if (accumulationMode == MetalAccumulationMode::PreviewProgressive &&
            g_accumulationDispatchCount < std::numeric_limits<uint32_t>::max())
        {
            ++g_accumulationDispatchCount;
        }

        const auto readbackStart = ProfileClock::now();
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        std::vector<float> tmp;
        tmp.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);

        [g_outTexture getBytes:tmp.data()
                    bytesPerRow:width * 4 * sizeof(float)
                     fromRegion:region
                    mipmapLevel:0];

        const std::size_t pixelCount = framebuffer.size();
        for (std::size_t i = 0; i < pixelCount; ++i)
        {
            framebuffer[i].x = tmp[i * 4 + 0];
            framebuffer[i].y = tmp[i * 4 + 1];
            framebuffer[i].z = tmp[i * 4 + 2];
        }
        profile.readbackMs = ToMilliseconds(ProfileClock::now() - readbackStart);
        profile.totalMs = ToMilliseconds(ProfileClock::now() - totalStart);

        PrintTextureFrameProfile(profile);
        return true;
    }
}
