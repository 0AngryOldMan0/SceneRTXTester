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
    id<MTLComputePipelineState> g_pipelineBuffer       = nil;
    id<MTLComputePipelineState> g_pipelineTexture      = nil;
    id<MTLComputePipelineState> g_pipelineBloomExtract = nil;
    id<MTLComputePipelineState> g_pipelineBloomBlurH   = nil;
    id<MTLComputePipelineState> g_pipelineBloomBlurV   = nil;
    id<MTLComputePipelineState> g_pipelinePostProcess  = nil;

    // Накопительная текстура + счётчик кадров для прогрессивного рендера
    id<MTLTexture>              g_accumTexture    = nil;
    uint32_t                    g_frameIndex      = 0;
    id<MTLTexture>              g_hdrTexture      = nil;
    id<MTLTexture>              g_albedoTexture   = nil;
    id<MTLTexture>              g_normalTexture   = nil;
    id<MTLTexture>              g_bloomTextureA   = nil;
    id<MTLTexture>              g_bloomTextureB   = nil;
    id<MTLTexture>              g_outTexture      = nil;
    int                         g_postTextureWidth = 0;
    int                         g_postTextureHeight = 0;

    // --- НОВОЕ: ресурсы материалов/текстур ---
    constexpr uint32_t          kMaxAllTextures = 124;

    id<MTLBuffer>               g_materialBuffer      = nil;
    id<MTLBuffer>               g_materialCountBuffer = nil;

    std::vector<id<MTLTexture>> g_baseColorTextures;
    std::vector<std::string>   g_baseColorTexturePaths; // чтобы понимать, когда нужно перезагружать

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
    id<MTLBuffer>               g_triCountBuffer              = nil;
    id<MTLBuffer>               g_meshNodeCountBuffer         = nil;
    id<MTLBuffer>               g_instanceCountBuffer         = nil;
    id<MTLBuffer>               g_lightBuffer                 = nil;
    id<MTLBuffer>               g_lightCountBuffer            = nil;
    id<MTLBuffer>               g_emissiveTriangleBuffer      = nil;
    id<MTLBuffer>               g_emissiveTriangleCountBuffer = nil;
    id<MTLBuffer>               g_decalBuffer                 = nil;
    id<MTLBuffer>               g_decalCountBuffer            = nil;
    id<MTLBuffer>               g_camBuffer                   = nil;
    id<MTLBuffer>               g_imgSizeBuffer               = nil;
    id<MTLBuffer>               g_frameIndexBuffer            = nil;
    id<MTLBuffer>               g_postProcessBuffer           = nil;

    std::uint64_t               g_cachedSceneRevision         = 0;
    const SceneMetaResources   *g_cachedDecalMetaRes          = nullptr;

    using ProfileClock = std::chrono::steady_clock;

    struct TextureFrameProfile
    {
        uint64_t callIndex = 0;
        uint32_t accumulationFrame = 0;
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

    struct BufferFrameProfile
    {
        uint64_t callIndex = 0;
        double geometryBuffersMs = 0.0;
        double framebufferBufferMs = 0.0;
        double lightUploadMs = 0.0;
        double encodeMs = 0.0;
        double waitMs = 0.0;
        double readbackMs = 0.0;
        double gpuMs = 0.0;
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

    struct BufferProfileTotals
    {
        uint64_t frameCount = 0;
        double geometryBuffersMs = 0.0;
        double framebufferBufferMs = 0.0;
        double lightUploadMs = 0.0;
        double encodeMs = 0.0;
        double waitMs = 0.0;
        double readbackMs = 0.0;
        double gpuMs = 0.0;
        double totalMs = 0.0;
    };

    uint64_t g_textureProfileCallIndex = 0;
    uint64_t g_bufferProfileCallIndex = 0;
    TextureProfileTotals g_textureProfileTotals;
    BufferProfileTotals g_bufferProfileTotals;

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
            << " accum=" << p.accumulationFrame << "\n"
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
        if (p.primaryDepthGpuMs <= 0.0)
            oss << "    note: primary-depth is not a separate pass in the current texture path; it is folded into path trace.\n";

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

    static void PrintBufferFrameProfile(const BufferFrameProfile &p)
    {
        auto &tot = g_bufferProfileTotals;
        tot.frameCount += 1;
        tot.geometryBuffersMs += p.geometryBuffersMs;
        tot.framebufferBufferMs += p.framebufferBufferMs;
        tot.lightUploadMs += p.lightUploadMs;
        tot.encodeMs += p.encodeMs;
        tot.waitMs += p.waitMs;
        tot.readbackMs += p.readbackMs;
        tot.gpuMs += p.gpuMs;
        tot.totalMs += p.totalMs;

        const double inv = (tot.frameCount > 0) ? (1.0 / static_cast<double>(tot.frameCount)) : 0.0;
        const double avgTotal = tot.totalMs * inv;
        const double avgWait = tot.waitMs * inv;
        const double avgReadback = tot.readbackMs * inv;
        const double avgGPU = tot.gpuMs * inv;

        std::ostringstream oss;
        oss << "[MetalProfiler][Buffer] call=" << p.callIndex << "\n"
            << "  current:\n";

        AppendProfileLine(oss, "geom", p.geometryBuffersMs, p.totalMs);
        AppendProfileLine(oss, "fb", p.framebufferBufferMs, p.totalMs);
        AppendProfileLine(oss, "lights", p.lightUploadMs, p.totalMs);
        AppendProfileLine(oss, "encode", p.encodeMs, p.totalMs);
        AppendProfileLine(oss, "wait", p.waitMs, p.totalMs);
        AppendProfileLine(oss, "readback", p.readbackMs, p.totalMs);
        AppendProfileLine(oss, "gpu", p.gpuMs, p.totalMs);
        AppendProfileLine(oss, "total", p.totalMs, p.totalMs);

        oss << "  average:\n";
        AppendProfileLine(oss, "avgTotal", avgTotal, avgTotal);
        AppendProfileLine(oss, "avgWait", avgWait, avgTotal);
        AppendProfileLine(oss, "avgReadback", avgReadback, avgTotal);
        AppendProfileLine(oss, "avgGPU", avgGPU, avgTotal);
        oss << std::endl;

        AppendProfileText(oss.str());
    }
}

struct LightGPU
{
    int   type;
    int   _pad0;

    Vec3  position;
    Vec3  direction;
    Vec3  color;

    float intensity;
    float radius;            // геометрический размер источника
    float spotSize;          // радианы
    float spotBlend;         // 0..1

    float attenuationRadius; // UE AttenuationRadius, 0 = бесконечный (старое поведение)
};
static_assert(sizeof(LightGPU) == 64, "LightGPU size must be 64 bytes");

// Материал на GPU: треугольник хранит materialIndex, а материал хранит ссылки на текстуры.
// (Это масштабируемый “движковый” способ, как в UE/рендер-пайплайнах.)
// Материал на GPU (минимальный layout, должен совпадать с RayTrace.metal)
struct MaterialGPU
{
    int32_t baseColorTexIndex; // индекс в массиве baseColorTextures, -1 если нет
    int32_t emissionTexIndex;  // на будущее
    int32_t _pad0;
    int32_t _pad1;
};

// Расширенный материал (под PBR карты). Пока НЕ используется шейдером.
// Чтобы не сломать текущий рендер, старый MaterialGPU (16B) остаётся в buffer(10).
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
};
static_assert(sizeof(MaterialGPU_PBR) == 32, "MaterialGPU_PBR size must be 32 bytes");
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
};
static_assert(sizeof(PostProcessParamsCPU) == 128, "PostProcessParamsCPU size must be 128 bytes");

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
            ResetMetalAccumulation();
            g_loadedMetaRes = nullptr;
            g_cachedDecalMetaRes = nullptr;
            g_baseColorTextures.clear();
            g_baseColorTexturePaths.clear();
            g_linearTextureValid.clear();
            g_materialBuffer = nil;
            g_materialCountBuffer = nil;
            g_materialPBRBuffer = nil;
            g_materialPBRCountBuffer = nil;
            g_decalBuffer = nil;
            g_decalCountBuffer = nil;
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
    ResetMetalAccumulation();

    g_baseColorTextures.clear();
    g_baseColorTexturePaths.clear();
    g_linearTextureValid.clear();

    g_materialBuffer          = nil;
    g_materialCountBuffer     = nil;
    g_materialPBRBuffer       = nil;
    g_materialPBRCountBuffer  = nil;
    g_cachedDecalMetaRes      = nullptr;
    g_decalBuffer             = nil;
    g_decalCountBuffer        = nil;

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
        m._pad0 = 0;
        m._pad1 = 0;

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

        if (sm.blendMode >= 2 && mp.emissionTexIndex >= 0 && mp.emissionTexIndex == mp.baseColorTexIndex)
            mp.flags |= 1;

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
                                                                  const std::vector<SceneInstanceGPU> &instances)
{
    std::vector<EmissiveTriangleGPUCPU> out;
    if (tris.empty() || instances.empty())
        return out;

    out.reserve(tris.size());
    double totalWeight = 0.0;
    std::vector<double> weights;
    weights.reserve(tris.size());

    for (std::size_t instIdx = 0; instIdx < instances.size(); ++instIdx)
    {
        const SceneInstanceGPU &inst = instances[instIdx];
        for (std::size_t triIdx = 0; triIdx < tris.size(); ++triIdx)
        {
            const Triangle &t = tris[triIdx];
            const float lum  = std::max(0.0f, LuminanceCPU(t.emission));
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
        !g_triCountBuffer ||
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
        std::cerr << "Metal: не удалось создать persistent буферы геометрии (texture)\n";
        return false;
    }

    const uint32_t nodeCount = static_cast<uint32_t>(tlasNodes.size());
    const uint32_t triCount = static_cast<uint32_t>(tris.size());
    const uint32_t meshNodeCount = static_cast<uint32_t>(meshNodes.size());
    const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
    if (!UploadSharedValue(g_device, g_nodeCountBuffer, nodeCount) ||
        !UploadSharedValue(g_device, g_rootIndexBuffer, rootIndex) ||
        !UploadSharedValue(g_device, g_triCountBuffer, triCount) ||
        !UploadSharedValue(g_device, g_meshNodeCountBuffer, meshNodeCount) ||
        !UploadSharedValue(g_device, g_instanceCountBuffer, instanceCount))
    {
        std::cerr << "Metal: не удалось создать persistent small geometry buffers\n";
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
        dst._pad0     = 0;
        dst.position  = src.position;
        dst.direction = src.direction;
        dst.color     = src.color;
        dst.intensity = src.intensity;
        dst.radius    = src.radius;
        dst.spotSize  = src.spotSize;
        dst.spotBlend = src.spotBlend;
        dst.attenuationRadius = src.attenuationRadius;
        gpuLights.push_back(dst);
    }

    const LightGPU dummyLight{};
    const uint32_t lightCount = static_cast<uint32_t>(gpuLights.size());
    if (!UploadSharedVector(g_device, g_lightBuffer, gpuLights, dummyLight) ||
        !UploadSharedValue(g_device, g_lightCountBuffer, lightCount))
    {
        std::cerr << "Metal: не удалось создать persistent light buffers\n";
        return false;
    }
    profile.lightUploadMs = ToMilliseconds(ProfileClock::now() - lightStart);

    const auto emissiveStart = ProfileClock::now();
    const std::vector<EmissiveTriangleGPUCPU> emissiveTrianglesCPU =
        BuildEmissiveTriangleTable(tris, instances);
    const EmissiveTriangleGPUCPU dummyEmissive{};
    const uint32_t emissiveTriangleCount =
        static_cast<uint32_t>(emissiveTrianglesCPU.size());
    if (!UploadSharedVector(g_device, g_emissiveTriangleBuffer, emissiveTrianglesCPU, dummyEmissive) ||
        !UploadSharedValue(g_device, g_emissiveTriangleCountBuffer, emissiveTriangleCount))
    {
        std::cerr << "Metal: не удалось создать persistent emissive buffers\n";
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
        std::cerr << "Metal: не удалось создать persistent decal buffers\n";
        return false;
    }

    profile.emissiveDecalMs += ToMilliseconds(ProfileClock::now() - decalStart);
    g_cachedDecalMetaRes = metaRes;
    return true;
}

static PostProcessParamsCPU BuildPostProcessParams(const SceneMetaResources *metaRes,
                                                   const CameraDataCPU &cameraCPU,
                                                   int width,
                                                   int height,
                                                   uint32_t frameIndex)
{
    PostProcessParamsCPU pp{};

    pp.exposure = 0.72f;
    pp.bloomIntensity = 0.09f;
    pp.bloomThreshold = -0.04f;
    pp.vignetteIntensity = 0.0f;
    pp.chromaticAberration = 0.0f;
    pp.filmGrainIntensity = 0.0f;
    pp.filmSlope = 1.0f;
    pp.filmToe = 0.52f;
    pp.filmShoulder = 1.0f;
    pp.filmBlackClip = 0.0f;
    pp.filmWhiteClip = 0.78f;

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

    pp.colorSaturationX = 1.02f;
    pp.colorSaturationY = 0.99f;
    pp.colorSaturationZ = 0.97f;
    pp.shadowLift = 0.006f;

    if (metaRes)
    {
        if (metaRes->hasPostProcess)
        {
            const SceneMetaPostProcess &src = metaRes->postProcess;
            const float exposureBias = std::clamp(src.autoExposureBias, -4.0f, 4.0f);
            pp.exposure = 0.72f * std::pow(2.0f, 0.18f * exposureBias);

            pp.bloomIntensity = std::max(0.0f, src.bloomIntensity) * 0.34f;
            pp.bloomThreshold = std::clamp(src.bloomThreshold * 0.40f + 0.02f, -1.0f, 1.25f);

            pp.vignetteIntensity = std::clamp(src.vignetteIntensity * 0.82f, 0.0f, 1.0f);
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
            pp.fogDensity = std::max(0.0f, src.fogDensity) * 0.11f;
            pp.fogHeightFalloff = std::max(0.0f, src.heightFalloff) * 0.30f;
            pp.fogScatteringG = src.scatteringG;
            pp.fogColorX = src.inscatteringColor.x;
            pp.fogColorY = src.inscatteringColor.y;
            pp.fogColorZ = src.inscatteringColor.z;
            pp.fogExtinctionScale = std::max(0.0f, src.extinctionScale) * 0.60f;
            pp.fogAlbedoX = src.volumetricAlbedo.x;
            pp.fogAlbedoY = src.volumetricAlbedo.y;
            pp.fogAlbedoZ = src.volumetricAlbedo.z;
            pp.volumetricFog = src.volumetricFog ? 1.0f : 0.0f;
        }
    }

    pp.nearPlane = std::max(1.0e-4f, cameraCPU.nearPlane);
    pp.farPlane  = std::max(pp.nearPlane + 1.0e-3f, cameraCPU.farPlane);
    pp.time = static_cast<float>(frameIndex) * (1.0f / 60.0f);
    pp.width = static_cast<float>(width);
    pp.height = static_cast<float>(height);
    pp._pad0 = 0.0f;
    return pp;
}

void ResetMetalAccumulation()
{
    g_frameIndex   = 0;
    g_accumTexture = nil;
    g_textureProfileTotals = {};
    g_textureProfileCallIndex = 0;
}

// Инициализация Metal: устройство, очередь, пайплайны
bool InitMetalRenderer()
{
    @autoreleasepool
    {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device)
        {
            std::cerr << "Metal: не удалось создать устройство\n";
            return false;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue)
        {
            std::cerr << "Metal: не удалось создать командную очередь\n";
            return false;
        }

        NSError *error = nil;
        NSString *libPath = @"build/RayTrace.metallib";
        NSURL    *libURL  = [NSURL fileURLWithPath:libPath];

        id<MTLLibrary> lib = [g_device newLibraryWithURL:libURL error:&error];
        if (!lib || error)
        {
            std::cerr << "Metal: не удалось загрузить RayTrace.metallib по пути: "
                      << [libPath UTF8String] << "\n";
            if (error)
                std::cerr << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> funcBuf = [lib newFunctionWithName:@"RayTraceKernel"];
        if (!funcBuf)
        {
            std::cerr << "Metal: function not found in metallib: RayTraceKernel" << std::endl;
            return false;
        }
        error = nil;
        g_pipelineBuffer = [g_device newComputePipelineStateWithFunction:funcBuf error:&error];

        if (!g_pipelineBuffer || error)
        {
            std::cerr << "Metal: не удалось создать compute pipeline state (RayTraceKernel)\n";
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
            std::cerr << "Metal: не удалось создать compute pipeline state (RayTraceTextureKernel)\n";
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

bool RenderFrameMetal(const std::vector<BVHNode>  &nodes,
                      const std::vector<Triangle> &tris,
                      const std::vector<Light>    &lights,
                      int                          rootIndex,
                      const CameraDataCPU         &cameraCPU,
                      std::vector<Vec3>           &framebuffer)
{
    if (!g_device || !g_queue || !g_pipelineBuffer)
    {
        std::cerr << "MetalRenderer: не инициализирован (вызови InitMetalRenderer())\n";
        return false;
    }

    const int width  = cameraCPU.width;
    const int height = cameraCPU.height;
    if (width <= 0 || height <= 0)
        return false;

    framebuffer.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    BufferFrameProfile profile{};
    profile.callIndex = ++g_bufferProfileCallIndex;
    const auto totalStart = ProfileClock::now();

    @autoreleasepool
    {
        const auto geometryStart = ProfileClock::now();

        // ----------------- буферы геометрии -----------------
        const std::size_t bvhSize = nodes.size() * sizeof(BVHNode);
        id<MTLBuffer> bvhBuffer =
            [g_device newBufferWithBytes:nodes.data()
                                  length:bvhSize
                                 options:MTLResourceStorageModeShared];
        if (!bvhBuffer)
        {
            std::cerr << "Metal: не удалось создать буфер для BVH\n";
            return false;
        }

        const std::size_t triSize = tris.size() * sizeof(Triangle);
        id<MTLBuffer> triBuffer =
            [g_device newBufferWithBytes:tris.data()
                                  length:triSize
                                 options:MTLResourceStorageModeShared];
        if (!triBuffer)
        {
            std::cerr << "Metal: не удалось создать буфер для треугольников\n";
            return false;
        }
        profile.geometryBuffersMs = ToMilliseconds(ProfileClock::now() - geometryStart);

        // ----------------- буфер кадра -----------------
        const auto fbBufferStart = ProfileClock::now();
        const std::size_t fbSize = framebuffer.size() * sizeof(Vec3);
        id<MTLBuffer> fbBuffer =
            [g_device newBufferWithLength:fbSize
                                  options:MTLResourceStorageModeShared];
        if (!fbBuffer)
        {
            std::cerr << "Metal: не удалось создать буфер кадра\n";
            return false;
        }
        profile.framebufferBufferMs = ToMilliseconds(ProfileClock::now() - fbBufferStart);

        // ----------------- мелкие константы -----------------
        uint32_t triCount  = static_cast<uint32_t>(tris.size());
        uint32_t nodeCount = static_cast<uint32_t>(nodes.size());
        int      rootIdx   = rootIndex;
        CameraDataCPU camCopy = cameraCPU;
        uint32_t imageSize[2] = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        // ----------------- буфер источников света -----------------
        const auto lightStart = ProfileClock::now();
        std::vector<LightGPU> gpuLights;
        gpuLights.reserve(lights.size());

        for (const Light &src : lights)
        {
            LightGPU dst{};
            dst.type      = static_cast<int>(src.type);
            dst._pad0     = 0;
            dst.position  = src.position;
            dst.direction = src.direction;
            dst.color     = src.color;
            dst.intensity = src.intensity;
            dst.radius    = src.radius;
            dst.spotSize  = src.spotSize;
            dst.spotBlend = src.spotBlend;
            dst.attenuationRadius = src.attenuationRadius;

            gpuLights.push_back(dst);
        }

        id<MTLBuffer> lightBuffer = nil;
        uint32_t      lightCount  = static_cast<uint32_t>(gpuLights.size());

        if (!gpuLights.empty())
        {
            lightBuffer =
                [g_device newBufferWithBytes:gpuLights.data()
                                      length:gpuLights.size() * sizeof(LightGPU)
                                     options:MTLResourceStorageModeShared];
        }
        profile.lightUploadMs = ToMilliseconds(ProfileClock::now() - lightStart);

        // ----------------- командный буфер + encoder -----------------
        const auto encodeStart = ProfileClock::now();
        id<MTLCommandBuffer>         cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pipelineBuffer];
        [enc setBuffer:bvhBuffer offset:0 atIndex:0];
        [enc setBuffer:triBuffer offset:0 atIndex:1];
        [enc setBytes:&nodeCount length:sizeof(uint32_t)      atIndex:2];
        [enc setBytes:&rootIdx   length:sizeof(int)           atIndex:3];
        [enc setBytes:&camCopy   length:sizeof(CameraDataCPU) atIndex:4];
        [enc setBytes:&imageSize length:sizeof(imageSize)     atIndex:5];
        [enc setBytes:&triCount  length:sizeof(uint32_t)      atIndex:6];
        [enc setBuffer:fbBuffer  offset:0                     atIndex:7];

        if (lightBuffer)
            [enc setBuffer:lightBuffer offset:0 atIndex:8];
        [enc setBytes:&lightCount length:sizeof(uint32_t)     atIndex:9];

        MTLSize gridSize = MTLSizeMake(width, height, 1);

        NSUInteger tgWidth  = 8;
        NSUInteger tgHeight = 8;
        NSUInteger maxThreads = g_pipelineBuffer.maxTotalThreadsPerThreadgroup;
        if (tgWidth * tgHeight > maxThreads)
        {
            tgWidth  = maxThreads;
            tgHeight = 1;
        }

        MTLSize threadsPerGroup = MTLSizeMake(tgWidth, tgHeight, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cmd commit];
        profile.encodeMs = ToMilliseconds(ProfileClock::now() - encodeStart);

        const auto waitStart = ProfileClock::now();
        [cmd waitUntilCompleted];
        profile.waitMs = ToMilliseconds(ProfileClock::now() - waitStart);
        profile.gpuMs = SafeGpuTimeMs(cmd);

        // ----------------- копируем результат обратно -----------------
        const auto readbackStart = ProfileClock::now();
        float *fbPtr = reinterpret_cast<float *>([fbBuffer contents]);
        const std::size_t pixelCount = framebuffer.size();
        for (std::size_t i = 0; i < pixelCount; ++i)
        {
            framebuffer[i].x = fbPtr[i * 3 + 0];
            framebuffer[i].y = fbPtr[i * 3 + 1];
            framebuffer[i].z = fbPtr[i * 3 + 2];
        }
        profile.readbackMs = ToMilliseconds(ProfileClock::now() - readbackStart);
        profile.totalMs = ToMilliseconds(ProfileClock::now() - totalStart);

        PrintBufferFrameProfile(profile);
        return true;
    }
}

// ВАЖНО: добавили параметр metaRes, чтобы передать список материалов/текстур из SceneMetaLoader.
bool RenderFrameMetalTexture(const std::vector<BVHNode>   &tlasNodes,
                             const std::vector<BVHNode>   &meshNodes,
                             const std::vector<Triangle> &tris,
                             const std::vector<SceneInstanceGPU> &instances,
                             const std::vector<Light>    &lights,
                             std::uint64_t                sceneRevision,
                             int                          rootIndex,
                             const CameraDataCPU         &cameraCPU,
                             const SceneMetaResources    *metaRes,
                             std::vector<Vec3>           &framebuffer)
{
    TextureFrameProfile profile{};
    profile.callIndex = ++g_textureProfileCallIndex;
    profile.accumulationFrame = g_frameIndex;
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
        const auto accumStart = ProfileClock::now();
        if (!g_accumTexture ||
            g_accumTexture.width  != static_cast<NSUInteger>(width) ||
            g_accumTexture.height != static_cast<NSUInteger>(height))
        {
            g_accumTexture = CreateRGBA32FloatTexture(g_device,
                                                      width,
                                                      height,
                                                      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
            g_frameIndex = 0;
            profile.accumulationFrame = 0;
        }
        profile.accumTextureMs = ToMilliseconds(ProfileClock::now() - accumStart);

        const auto interTexStart = ProfileClock::now();
        if (!EnsurePostProcessTextures(width, height))
        {
            std::cerr << "Metal: не удалось создать post-process textures\n";
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

        const uint32_t frameIndex = g_frameIndex;

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

        PostProcessParamsCPU pp = BuildPostProcessParams(metaRes, cameraCPU, width, height, frameIndex);
        const std::array<uint32_t, 2> imageSize = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        const auto smallStart = ProfileClock::now();
        if (!UploadSharedValue(g_device, g_camBuffer, cameraCPU) ||
            !UploadSharedValue(g_device, g_imgSizeBuffer, imageSize) ||
            !UploadSharedValue(g_device, g_frameIndexBuffer, frameIndex) ||
            !UploadSharedValue(g_device, g_postProcessBuffer, pp))
        {
            std::cerr << "Metal: не удалось обновить persistent small/frame buffers\n";
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
            [enc setBuffer:g_triCountBuffer            offset:0 atIndex:6];
            [enc setBuffer:g_lightBuffer               offset:0 atIndex:7];
            [enc setBuffer:g_lightCountBuffer          offset:0 atIndex:8];
            [enc setBuffer:g_frameIndexBuffer          offset:0 atIndex:9];
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

        g_frameIndex++;

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
