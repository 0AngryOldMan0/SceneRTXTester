#include "HIPRendererFunctions.h"
#include "HIPRendererInternals.h"
#include "SceneMetaLoader.h"

#include "stb_image.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr std::uint64_t kInvalidAccumulationHash = std::numeric_limits<std::uint64_t>::max();

    struct CpuSampleTexture
    {
        std::size_t width = 0;
        std::size_t height = 0;
        std::vector<std::uint8_t> rgba;

        bool valid() const
        {
            return width > 0 && height > 0 && rgba.size() == width * height * 4u;
        }
    };

    struct CachedMetaResources
    {
        const SceneMetaResources *metaRes = nullptr;
        std::uint64_t signature = 0u;
        std::uint64_t generation = 0u;

        std::vector<std::string> combinedTexturePaths;
        std::vector<std::uint32_t> combinedTextureFlags;
        std::vector<HIPTextureDescGPU> textureDescs;
        std::vector<std::uint8_t> textureTexels;
        std::vector<HIPMaterialGPU> materials;
        std::vector<HIPMaterialPBRGPU> materialsPBR;
        std::vector<HIPDecalGPU> decals;
        std::vector<HIPAirDustVolumeGPU> airDustVolumes;
        std::unordered_map<std::string, CpuSampleTexture> cpuTextureCache;

        void clear()
        {
            metaRes = nullptr;
            signature = 0u;
            combinedTexturePaths.clear();
            combinedTextureFlags.clear();
            textureDescs.clear();
            textureTexels.clear();
            materials.clear();
            materialsPBR.clear();
            decals.clear();
            airDustVolumes.clear();
            cpuTextureCache.clear();
        }
    };

    CachedMetaResources g_metaCache;
    std::uint64_t g_accumulationStateHashHost = kInvalidAccumulationHash;
    std::uint32_t g_accumulatedSampleCountHost = 0u;
    std::uint32_t g_previewDispatchCountHost = 0u;
    std::uint64_t g_textureProfileCallIndexHost = 0u;
    std::uint64_t g_materialDebugPrintedGenerationHost = 0u;

    using ProfileClock = std::chrono::steady_clock;

    struct HIPTextureFrameProfile
    {
        std::uint64_t callIndex = 0u;
        std::uint32_t accumulatedSamples = 0u;
        std::uint32_t samplesThisDispatch = 0u;
        HIPAccumulationMode accumulationMode = HIPAccumulationMode::PreviewProgressive;
        HIPDebugView debugView = HIPDebugView::Disabled;
        std::uint32_t prototypeTriangles = 0u;
        std::uint32_t totalInstances = 0u;
        std::uint32_t tlasNodeCount = 0u;
        std::uint32_t blasNodeCount = 0u;
        std::uint32_t lightCount = 0u;
        std::uint32_t materialCount = 0u;
        std::uint32_t materialPBRCount = 0u;
        std::uint32_t textureCount = 0u;
        std::uint32_t emissiveTriangleCount = 0u;
        std::uint32_t decalCount = 0u;
        std::uint32_t airDustVolumeCount = 0u;
        double ensureMetaMs = 0.0;
        double accumulationStateMs = 0.0;
        double framebufferResizeMs = 0.0;
        double instanceConvertMs = 0.0;
        double lightConvertMs = 0.0;
        double emissiveBuildMs = 0.0;
        double postParamsMs = 0.0;
        double hipCoreMs = 0.0;
        double totalMs = 0.0;
        HIPTextureExecutionProfile lowLevel{};
    };

    struct HIPTextureProfileTotals
    {
        std::uint64_t frameCount = 0u;
        double ensureMetaMs = 0.0;
        double accumulationStateMs = 0.0;
        double framebufferResizeMs = 0.0;
        double instanceConvertMs = 0.0;
        double lightConvertMs = 0.0;
        double emissiveBuildMs = 0.0;
        double postParamsMs = 0.0;
        double hipCoreMs = 0.0;
        double totalMs = 0.0;
        double ensureOutputMs = 0.0;
        double uploadSceneMs = 0.0;
        double kernelMs = 0.0;
        double readbackMs = 0.0;
        double accumulationMs = 0.0;
        double postProcessMs = 0.0;
        double lowLevelTotalMs = 0.0;
    };

    HIPTextureProfileTotals g_textureProfileTotalsHost;

    struct HIPTextureIndexMapping
    {
        std::vector<std::int32_t> srgbColorToCombined;
        std::vector<std::int32_t> linearDataToCombined;
    };

    static double ToMilliseconds(ProfileClock::duration duration)
    {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    static std::uint32_t ToU32Count(std::size_t value)
    {
        return static_cast<std::uint32_t>(std::min<std::size_t>(value, UINT32_MAX));
    }

    static std::int32_t MapCombinedTextureIndex(int idx,
                                                const std::vector<std::int32_t> &semanticMap)
    {
        if (idx < 0 || static_cast<std::size_t>(idx) >= semanticMap.size())
            return -1;
        return semanticMap[static_cast<std::size_t>(idx)];
    }

    static HIPTextureIndexMapping BuildCombinedTextureCache(const SceneMetaResources *metaRes)
    {
        HIPTextureIndexMapping mapping;
        if (!metaRes)
            return mapping;

        const std::size_t srgbColorCount = std::min(metaRes->baseColorTextures.size(),
                                                    static_cast<std::size_t>(HIP_MAX_SCENE_TEXTURES));
        const std::size_t remaining = static_cast<std::size_t>(HIP_MAX_SCENE_TEXTURES) - srgbColorCount;
        const std::size_t linearDataCount = std::min(metaRes->linearTextures.size(), remaining);

        mapping.srgbColorToCombined.resize(srgbColorCount, -1);
        mapping.linearDataToCombined.resize(linearDataCount, -1);

        g_metaCache.combinedTexturePaths.reserve(srgbColorCount + linearDataCount);
        g_metaCache.combinedTextureFlags.reserve(srgbColorCount + linearDataCount);

        const auto appendTexture = [&](const std::string &path, std::uint32_t flags) -> std::int32_t
        {
            const std::int32_t combinedIndex = static_cast<std::int32_t>(g_metaCache.combinedTexturePaths.size());
            g_metaCache.combinedTexturePaths.push_back(path);
            g_metaCache.combinedTextureFlags.push_back(flags);
            return combinedIndex;
        };

        for (std::size_t i = 0; i < srgbColorCount; ++i)
        {
            mapping.srgbColorToCombined[i] =
                appendTexture(metaRes->baseColorTextures[i], HIP_TEXTURE_FLAG_SRGB);
        }

        for (std::size_t i = 0; i < linearDataCount; ++i)
        {
            mapping.linearDataToCombined[i] =
                appendTexture(metaRes->linearTextures[i], 0u);
        }

        return mapping;
    }

    static const char *AccumulationModeName(HIPAccumulationMode mode)
    {
        switch (mode)
        {
            case HIPAccumulationMode::PreviewProgressive: return "preview_progressive";
            case HIPAccumulationMode::FinalStill:         return "final_still";
        }
        return "unknown";
    }

    static const char *DebugViewName(HIPDebugView view)
    {
        switch (view)
        {
            case HIPDebugView::Disabled:      return "beauty";
            case HIPDebugView::Ns:            return "ns";
            case HIPDebugView::AO:            return "ao";
            case HIPDebugView::NsMinusNg:     return "ns_minus_ng";
            case HIPDebugView::BaseColor:     return "base_color";
            case HIPDebugView::Roughness:     return "roughness";
            case HIPDebugView::Metallic:      return "metallic";
            case HIPDebugView::Emissive:      return "emissive";
            case HIPDebugView::VertexColor:   return "vertex_color";
            case HIPDebugView::MaterialModel: return "material_model";
        }
        return "unknown";
    }

    static void AppendProfileLine(std::ostringstream &oss,
                                  const char *name,
                                  double valueMs,
                                  double totalMs,
                                  bool showPercent = true)
    {
        oss << "    "
            << std::left << std::setw(18) << name
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
                                std::uint32_t value)
    {
        oss << "    "
            << std::left << std::setw(18) << name
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

    static std::string ToLowerAscii(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return value;
    }

    static bool IsWallMaterialCandidate(const std::string &name)
    {
        const std::string lowered = ToLowerAscii(name);
        return lowered.find("concretewall") != std::string::npos ||
               lowered.find("concrete_wall") != std::string::npos ||
               lowered.find("wall") != std::string::npos ||
               lowered.find("tunnel") != std::string::npos ||
               lowered.find("subway") != std::string::npos;
    }

    static void PrintHIPMaterialDebugOnce(const SceneMetaResources *metaRes)
    {
        if (!metaRes || g_materialDebugPrintedGenerationHost == g_metaCache.generation)
            return;

        g_materialDebugPrintedGenerationHost = g_metaCache.generation;

        const std::size_t materialCount = std::min(metaRes->materialsPBR.size(), g_metaCache.materialsPBR.size());
        std::ostringstream oss;
        oss << "[HIPMaterialDebug] generation=" << g_metaCache.generation
            << " pbrSourceCount=" << metaRes->materialsPBR.size()
            << " pbrCachedCount=" << g_metaCache.materialsPBR.size()
            << " textureCount=" << g_metaCache.textureDescs.size()
            << "\n";

        if (materialCount == 0u)
        {
            oss << "[HIPMaterialDebug] no PBR materials available\n";
            AppendProfileText(oss.str());
            return;
        }

        std::vector<std::size_t> selected;
        selected.reserve(6);

        const auto pushUnique = [&](std::size_t index)
        {
            if (index >= materialCount)
                return;
            if (std::find(selected.begin(), selected.end(), index) == selected.end())
                selected.push_back(index);
        };

        for (std::size_t i = 0; i < materialCount && selected.size() < 6u; ++i)
        {
            if (IsWallMaterialCandidate(metaRes->materialsPBR[i].name))
                pushUnique(i);
        }

        for (std::size_t i = 0; i < materialCount && selected.size() < 6u; ++i)
        {
            const SceneMetaMaterial &material = metaRes->materialsPBR[i];
            if (material.normalTexIndex >= 0 || material.ormTexIndex >= 0 || material.occlusionTexIndex >= 0)
                pushUnique(i);
        }

        if (selected.empty())
            pushUnique(0u);

        for (std::size_t index : selected)
        {
            const SceneMetaMaterial &src = metaRes->materialsPBR[index];
            const HIPMaterialPBRGPU &gpu = g_metaCache.materialsPBR[index];
            oss << "[HIPMaterialDebug] matId=" << index
                << " name=\"" << src.name << "\""
                << " normalTex=" << src.normalTexIndex << "->" << gpu.normalTexIndex
                << " normalUvSet=" << src.normalUvSet << "->" << gpu.normalUvSet
                << " ormTex=" << src.ormTexIndex << "->" << gpu.ormTexIndex
                << " ormUvSet=" << src.ormUvSet << "->" << gpu.ormUvSet
                << " occlusionTex=" << src.occlusionTexIndex << "->" << gpu.occlusionTexIndex
                << " occlusionUvSet=" << src.occlusionUvSet << "->" << gpu.occlusionUvSet
                << " flags=0x" << std::hex << gpu.flags << std::dec
                << "\n";
        }

        AppendProfileText(oss.str());
    }

    static void PrintTextureFrameProfile(const HIPTextureFrameProfile &profile)
    {
        auto &totals = g_textureProfileTotalsHost;
        totals.frameCount += 1u;
        totals.ensureMetaMs += profile.ensureMetaMs;
        totals.accumulationStateMs += profile.accumulationStateMs;
        totals.framebufferResizeMs += profile.framebufferResizeMs;
        totals.instanceConvertMs += profile.instanceConvertMs;
        totals.lightConvertMs += profile.lightConvertMs;
        totals.emissiveBuildMs += profile.emissiveBuildMs;
        totals.postParamsMs += profile.postParamsMs;
        totals.hipCoreMs += profile.hipCoreMs;
        totals.totalMs += profile.totalMs;
        totals.ensureOutputMs += profile.lowLevel.ensureOutputMs;
        totals.uploadSceneMs += profile.lowLevel.uploadSceneMs;
        totals.kernelMs += profile.lowLevel.kernelMs;
        totals.readbackMs += profile.lowLevel.readbackMs;
        totals.accumulationMs += profile.lowLevel.accumulationMs;
        totals.postProcessMs += profile.lowLevel.postProcessMs;
        totals.lowLevelTotalMs += profile.lowLevel.totalMs;

        const double inv = (totals.frameCount > 0u)
                         ? (1.0 / static_cast<double>(totals.frameCount))
                         : 0.0;

        std::ostringstream oss;
        oss << "[HIPProfiler][Texture] call=" << profile.callIndex
            << " mode=" << AccumulationModeName(profile.accumulationMode)
            << " debug=" << DebugViewName(profile.debugView)
            << " accumSamples=" << profile.accumulatedSamples
            << " batchSamples=" << profile.samplesThisDispatch << "\n"
            << "  scene stats:\n";
        AppendCountLine(oss, "proto tris", profile.prototypeTriangles);
        AppendCountLine(oss, "instances", profile.totalInstances);
        AppendCountLine(oss, "TLAS nodes", profile.tlasNodeCount);
        AppendCountLine(oss, "BLAS nodes", profile.blasNodeCount);
        AppendCountLine(oss, "lights", profile.lightCount);
        AppendCountLine(oss, "materials", profile.materialCount);
        AppendCountLine(oss, "materialsPBR", profile.materialPBRCount);
        AppendCountLine(oss, "textures", profile.textureCount);
        AppendCountLine(oss, "emissive tris", profile.emissiveTriangleCount);
        AppendCountLine(oss, "decals", profile.decalCount);
        AppendCountLine(oss, "airDust vols", profile.airDustVolumeCount);

        oss << "  current:\n";
        AppendProfileLine(oss, "meta", profile.ensureMetaMs, profile.totalMs);
        AppendProfileLine(oss, "accumState", profile.accumulationStateMs, profile.totalMs);
        AppendProfileLine(oss, "resize", profile.framebufferResizeMs, profile.totalMs);
        AppendProfileLine(oss, "instances", profile.instanceConvertMs, profile.totalMs);
        AppendProfileLine(oss, "lights", profile.lightConvertMs, profile.totalMs);
        AppendProfileLine(oss, "emissive", profile.emissiveBuildMs, profile.totalMs);
        AppendProfileLine(oss, "postParams", profile.postParamsMs, profile.totalMs);
        AppendProfileLine(oss, "hipCore", profile.hipCoreMs, profile.totalMs);
        AppendProfileLine(oss, "total", profile.totalMs, profile.totalMs);

        oss << "  hip core (current):\n";
        AppendProfileLine(oss, "ensureOutput", profile.lowLevel.ensureOutputMs, profile.lowLevel.totalMs);
        AppendProfileLine(oss, "upload", profile.lowLevel.uploadSceneMs, profile.lowLevel.totalMs);
        AppendProfileLine(oss, "kernel", profile.lowLevel.kernelMs, profile.lowLevel.totalMs);
        AppendProfileLine(oss, "readback", profile.lowLevel.readbackMs, profile.lowLevel.totalMs);
        AppendProfileLine(oss, "accumulation", profile.lowLevel.accumulationMs, profile.lowLevel.totalMs);
        AppendProfileLine(oss, "postProcess", profile.lowLevel.postProcessMs, profile.lowLevel.totalMs);
        AppendProfileLine(oss, "coreTotal", profile.lowLevel.totalMs, profile.lowLevel.totalMs);

        oss << "  average:\n";
        AppendProfileLine(oss, "avgMeta", totals.ensureMetaMs * inv, totals.totalMs * inv);
        AppendProfileLine(oss, "avgAccumState", totals.accumulationStateMs * inv, totals.totalMs * inv);
        AppendProfileLine(oss, "avgInstances", totals.instanceConvertMs * inv, totals.totalMs * inv);
        AppendProfileLine(oss, "avgLights", totals.lightConvertMs * inv, totals.totalMs * inv);
        AppendProfileLine(oss, "avgEmissive", totals.emissiveBuildMs * inv, totals.totalMs * inv);
        AppendProfileLine(oss, "avgHipCore", totals.hipCoreMs * inv, totals.totalMs * inv);
        AppendProfileLine(oss, "avgTotal", totals.totalMs * inv, totals.totalMs * inv);

        oss << "  hip core (average):\n";
        AppendProfileLine(oss, "ensureOutput", totals.ensureOutputMs * inv, totals.lowLevelTotalMs * inv);
        AppendProfileLine(oss, "upload", totals.uploadSceneMs * inv, totals.lowLevelTotalMs * inv);
        AppendProfileLine(oss, "kernel", totals.kernelMs * inv, totals.lowLevelTotalMs * inv);
        AppendProfileLine(oss, "readback", totals.readbackMs * inv, totals.lowLevelTotalMs * inv);
        AppendProfileLine(oss, "accumulation", totals.accumulationMs * inv, totals.lowLevelTotalMs * inv);
        AppendProfileLine(oss, "postProcess", totals.postProcessMs * inv, totals.lowLevelTotalMs * inv);
        AppendProfileLine(oss, "coreTotal", totals.lowLevelTotalMs * inv, totals.lowLevelTotalMs * inv);
        oss << std::endl;

        AppendProfileText(oss.str());
    }

    static_assert(sizeof(SceneInstanceGPU) == sizeof(HIPSceneInstanceGPU),
                  "HIPSceneInstanceGPU must match SceneInstanceGPU");

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
            std::uint32_t bits = 0u;
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

    inline float Clamp01CPU(float v)
    {
        return std::clamp(v, 0.0f, 1.0f);
    }

    inline float LuminanceCPU(const Vec3 &c)
    {
        return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f;
    }

    inline float FractCPU(float v)
    {
        return v - std::floor(v);
    }

    inline Vec2 FractCPU(const Vec2 &uv)
    {
        return Vec2{FractCPU(uv.x), FractCPU(uv.y)};
    }

    inline float SrgbToLinearCPU(float c)
    {
        if (c <= 0.04045f)
            return c / 12.92f;
        return std::pow((c + 0.055f) / 1.055f, 2.4f);
    }

    const CpuSampleTexture *LoadCpuTextureCached(const std::string &path)
    {
        auto it = g_metaCache.cpuTextureCache.find(path);
        if (it != g_metaCache.cpuTextureCache.end())
            return it->second.valid() ? &it->second : nullptr;

        CpuSampleTexture image{};
        int width = 0;
        int height = 0;
        int channels = 0;
        stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
        if (pixels != nullptr && width > 0 && height > 0)
        {
            image.width = static_cast<std::size_t>(width);
            image.height = static_cast<std::size_t>(height);
            const std::size_t bytes = image.width * image.height * 4u;
            image.rgba.assign(pixels, pixels + bytes);
        }
        if (pixels != nullptr)
            stbi_image_free(pixels);

        auto [inserted, _] = g_metaCache.cpuTextureCache.emplace(path, std::move(image));
        return inserted->second.valid() ? &inserted->second : nullptr;
    }

    const CpuSampleTexture *GetCpuSceneTexture(const SceneMetaResources *metaRes, std::int32_t texIndex)
    {
        if (!metaRes || texIndex < 0 || static_cast<std::size_t>(texIndex) >= metaRes->baseColorTextures.size())
            return nullptr;

        return LoadCpuTextureCached(metaRes->baseColorTextures[static_cast<std::size_t>(texIndex)]);
    }

    inline Vec3 DecodeCpuSceneTexelLinear(const CpuSampleTexture &tex, int x, int y)
    {
        if (!tex.valid())
            return Vec3{0.0f, 0.0f, 0.0f};

        const int width = static_cast<int>(tex.width);
        const int height = static_cast<int>(tex.height);
        x = ((x % width) + width) % width;
        y = ((y % height) + height) % height;

        const std::size_t idx = (static_cast<std::size_t>(y) * tex.width + static_cast<std::size_t>(x)) * 4u;
        const float r = tex.rgba[idx + 0u] / 255.0f;
        const float g = tex.rgba[idx + 1u] / 255.0f;
        const float b = tex.rgba[idx + 2u] / 255.0f;
        return Vec3{SrgbToLinearCPU(r), SrgbToLinearCPU(g), SrgbToLinearCPU(b)};
    }

    inline float DecodeCpuSceneAlpha(const CpuSampleTexture &tex, int x, int y)
    {
        if (!tex.valid())
            return 1.0f;

        const int width = static_cast<int>(tex.width);
        const int height = static_cast<int>(tex.height);
        x = ((x % width) + width) % width;
        y = ((y % height) + height) % height;

        const std::size_t idx = (static_cast<std::size_t>(y) * tex.width + static_cast<std::size_t>(x)) * 4u;
        return tex.rgba[idx + 3u] / 255.0f;
    }

    void SampleCpuSceneTextureLinear(const CpuSampleTexture &tex,
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
        const float fx = uv.x * static_cast<float>(tex.width) - 0.5f;
        const float fy = uv.y * static_cast<float>(tex.height) - 0.5f;
        const int x0 = static_cast<int>(std::floor(fx));
        const int y0 = static_cast<int>(std::floor(fy));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float tx = fx - static_cast<float>(x0);
        const float ty = fy - static_cast<float>(y0);

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

    inline Vec2 TriangleUvAtBary(const Triangle &tri, int uvSet, float b0, float b1, float b2)
    {
        const int clampedSet = std::clamp(uvSet, 0, 2);
        const Vec2 &uv0 = tri.uv[clampedSet * 3 + 0];
        const Vec2 &uv1 = tri.uv[clampedSet * 3 + 1];
        const Vec2 &uv2 = tri.uv[clampedSet * 3 + 2];
        return Vec2{
            uv0.x * b0 + uv1.x * b1 + uv2.x * b2,
            uv0.y * b0 + uv1.y * b1 + uv2.y * b2
        };
    }

    inline float SampleEmissiveMaskCPU(const Vec3 &rgb, float alpha)
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

    Vec3 EvaluateTriangleEmissionCPU(const Triangle &tri,
                                     const SceneMetaMaterial &material,
                                     const SceneMetaResources *metaRes,
                                     const Vec2 &uvBase,
                                     const Vec2 &uvEmission,
                                     float ndv)
    {
        Vec3 emissive = tri.emission;

        Vec3 baseColorTex{1.0f, 1.0f, 1.0f};
        float dummyAlpha = 1.0f;
        if (const CpuSampleTexture *baseTex = GetCpuSceneTexture(metaRes, material.baseColorTexIndex))
            SampleCpuSceneTextureLinear(*baseTex, uvBase, baseColorTex, dummyAlpha);

        Vec3 emissionTex{1.0f, 1.0f, 1.0f};
        float emissionAlpha = 1.0f;
        const CpuSampleTexture *emissionTexCpu = GetCpuSceneTexture(metaRes, material.emissionTexIndex);
        if (emissionTexCpu)
            SampleCpuSceneTextureLinear(*emissionTexCpu, uvEmission, emissionTex, emissionAlpha);

        if (material.specialModel == HIP_SPECIAL_MATERIAL_UE_TRAFFIC_LIGHT)
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

        if (material.specialModel == HIP_SPECIAL_MATERIAL_UE_HEADLIGHT)
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

    float EstimateTriangleTexturedEmissiveLuminanceCPU(const Triangle &tri,
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
        return accum / static_cast<float>(kSampleBary.size());
    }

    inline float TriangleAreaCPU(const Triangle &tri)
    {
        const float e1x = tri.v1.x - tri.v0.x;
        const float e1y = tri.v1.y - tri.v0.y;
        const float e1z = tri.v1.z - tri.v0.z;
        const float e2x = tri.v2.x - tri.v0.x;
        const float e2y = tri.v2.y - tri.v0.y;
        const float e2z = tri.v2.z - tri.v0.z;

        const float crx = e1y * e2z - e1z * e2y;
        const float cry = e1z * e2x - e1x * e2z;
        const float crz = e1x * e2y - e1y * e2x;
        return 0.5f * std::sqrt(std::max(0.0f, crx * crx + cry * cry + crz * crz));
    }

    inline Vec3 TransformPointCPU(const float m[12], const Vec3 &p)
    {
        return Vec3{
            m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]
        };
    }

    std::vector<HIPEmissiveTriangleGPU> BuildEmissiveTriangleTable(const std::vector<Triangle> &tris,
                                                                   const std::vector<SceneInstanceGPU> &instances,
                                                                   const SceneMetaResources *metaRes)
    {
        std::vector<HIPEmissiveTriangleGPU> out;
        if (tris.empty() || instances.empty())
            return out;

        std::vector<float> triangleEmissiveLuma(tris.size(), 0.0f);
        for (std::size_t triIdx = 0; triIdx < tris.size(); ++triIdx)
        {
            const Triangle &tri = tris[triIdx];
            const float rawLum = std::max(0.0f, LuminanceCPU(tri.emission));
            if (rawLum <= 1.0e-8f)
                continue;

            float texturedLum = rawLum;
            const int matId = tri.materialIndex;
            if (metaRes && matId >= 0 && static_cast<std::size_t>(matId) < metaRes->materialsPBR.size())
            {
                const SceneMetaMaterial &material = metaRes->materialsPBR[static_cast<std::size_t>(matId)];
                if (material.emissionTexIndex >= 0 || material.specialModel != 0)
                {
                    texturedLum = EstimateTriangleTexturedEmissiveLuminanceCPU(tri, material, metaRes);
                    texturedLum = std::max(texturedLum, rawLum * 0.05f);
                }
            }
            triangleEmissiveLuma[triIdx] = texturedLum;
        }

        std::vector<double> weights;
        weights.reserve(tris.size() * instances.size());
        double totalWeight = 0.0;

        for (std::size_t instIdx = 0; instIdx < instances.size(); ++instIdx)
        {
            const SceneInstanceGPU &inst = instances[instIdx];
            for (std::size_t triIdx = 0; triIdx < tris.size(); ++triIdx)
            {
                const float lum = triangleEmissiveLuma[triIdx];
                if (lum <= 1.0e-8f)
                    continue;

                Triangle worldTri = tris[triIdx];
                worldTri.v0 = TransformPointCPU(inst.objectToWorld, worldTri.v0);
                worldTri.v1 = TransformPointCPU(inst.objectToWorld, worldTri.v1);
                worldTri.v2 = TransformPointCPU(inst.objectToWorld, worldTri.v2);

                const float area = TriangleAreaCPU(worldTri);
                const double weight = static_cast<double>(std::max(area, 1.0e-8f)) * static_cast<double>(lum);
                if (weight <= 1.0e-12)
                    continue;

                HIPEmissiveTriangleGPU entry{};
                entry.triIndex = static_cast<std::uint32_t>(triIdx);
                entry.instanceIndex = static_cast<std::uint32_t>(instIdx);
                entry.area = std::max(area, 1.0e-8f);
                out.push_back(entry);
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

    std::vector<HIPAirDustVolumeGPU> BuildAirDustVolumeTable(const SceneMetaResources *metaRes)
    {
        std::vector<HIPAirDustVolumeGPU> out;
        if (!metaRes)
            return out;

        out.reserve(metaRes->airDustVolumes.size());
        for (const SceneMetaAirDustVolume &src : metaRes->airDustVolumes)
        {
            if (src.linkedLightIntensity <= 0.0f)
                continue;

            HIPAirDustVolumeGPU volume{};
            volume.centerX = src.position.x;
            volume.centerY = src.position.y;
            volume.centerZ = src.position.z;
            volume.extentX = std::max(src.extent.x, 1.0f);
            volume.extentY = std::max(src.extent.y, 1.0f);
            volume.extentZ = std::max(src.extent.z, 1.0f);
            volume.lightPosX = src.linkedLightPosition.x;
            volume.lightPosY = src.linkedLightPosition.y;
            volume.lightPosZ = src.linkedLightPosition.z;
            volume.lightColorX = src.linkedLightColor.x;
            volume.lightColorY = src.linkedLightColor.y;
            volume.lightColorZ = src.linkedLightColor.z;
            volume.lightIntensity = src.linkedLightIntensity;
            volume.lightRadius = std::max(src.linkedLightRadius,
                                          std::max(volume.extentX, std::max(volume.extentY, volume.extentZ)) * 1.5f);

            const float normalizedIntensity = std::sqrt(std::max(src.linkedLightIntensity, 0.0f) / 2500.0f);
            volume.density = std::clamp(0.08f + normalizedIntensity * 0.10f, 0.06f, 0.24f);
            volume.anisotropy = 0.45f;
            out.push_back(volume);
        }

        return out;
    }

    std::vector<HIPDecalGPU> BuildDecalTable(const SceneMetaResources *metaRes,
                                             const HIPTextureIndexMapping &textureMapping)
    {
        std::vector<HIPDecalGPU> out;
        if (!metaRes || metaRes->decals.empty() || metaRes->materials.empty())
            return out;

        const auto mapBaseColorTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.srgbColorToCombined);
        };

        const auto mapLinearDataTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.linearDataToCombined);
        };

        const auto mapDecalAuxTextureIndex = [&](int idx, bool isLinear) -> std::int32_t
        {
            return isLinear ? mapLinearDataTextureIndex(idx) : mapBaseColorTextureIndex(idx);
        };

        out.reserve(metaRes->decals.size());
        for (const SceneMetaDecal &src : metaRes->decals)
        {
            if (src.materialIndex < 0 || static_cast<std::size_t>(src.materialIndex) >= metaRes->materials.size())
                continue;

            const SceneMetaMaterial &material = metaRes->materials[static_cast<std::size_t>(src.materialIndex)];
            const std::int32_t baseTex = mapBaseColorTextureIndex(material.baseColorTexIndex);
            if (baseTex < 0)
                continue;

            HIPDecalGPU decal{};
            decal.posX = src.position.x;
            decal.posY = src.position.y;
            decal.posZ = src.position.z;
            decal.sizeX = std::max(0.5f * src.size.x, 1.0f);
            decal.axisXx = src.axisX.x;
            decal.axisXy = src.axisX.y;
            decal.axisXz = src.axisX.z;
            decal.sizeY = std::max(0.5f * src.size.y, 1.0f);
            decal.axisYx = src.axisY.x;
            decal.axisYy = src.axisY.y;
            decal.axisYz = src.axisY.z;
            decal.sizeZ = std::max(0.5f * src.size.z, 1.0f);
            decal.axisZx = src.axisZ.x;
            decal.axisZy = src.axisZ.y;
            decal.axisZz = src.axisZ.z;
            decal.opacity = std::clamp(material.opacity * 1.30f, 0.0f, 2.0f);
            decal.baseColorTexIndex = baseTex;
            decal.ormTexIndex = mapLinearDataTextureIndex(material.ormTexIndex);
            decal.roughnessTexIndex = mapLinearDataTextureIndex(material.roughnessTexIndex);
            decal.normalTexIndex = mapLinearDataTextureIndex(material.normalTexIndex);
            decal.opacityTexIndex = mapDecalAuxTextureIndex(material.decalOpacityTexIndex, material.decalOpacityTexIsLinear);
            decal.detailTexIndex = mapDecalAuxTextureIndex(material.decalDetailTexIndex, material.decalDetailTexIsLinear);
            decal.baseColorX = material.baseColor.x;
            decal.baseColorY = material.baseColor.y;
            decal.baseColorZ = material.baseColor.z;
            decal.roughnessBias = material.decalRoughnessBias;
            decal.tilingU = material.decalTilingU;
            decal.tilingV = material.decalTilingV;
            decal.opacityPower = std::max(material.decalOpacityPower, 0.25f);
            decal.normalIntensity = std::clamp(material.decalNormalIntensity, 0.0f, 8.0f);
            out.push_back(decal);
        }

        return out;
    }

    HIPPostProcessParams BuildPostProcessParams(const SceneMetaResources *metaRes,
                                                const CameraDataCPU &cameraCPU,
                                                int width,
                                                int height,
                                                std::uint32_t previewDispatchCount,
                                                HIPAccumulationMode accumulationMode,
                                                HIPDebugView debugView)
    {
        HIPPostProcessParams pp{};
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
        pp.shadowLift = 0.010f;

        if (metaRes)
        {
            if (metaRes->hasPostProcess)
            {
                const SceneMetaPostProcess &src = metaRes->postProcess;
                pp.exposure = std::exp2(std::clamp(src.autoExposureBias, -4.0f, 4.0f));
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
        pp.farPlane = std::max(pp.nearPlane + 1.0e-3f, cameraCPU.farPlane);
        pp.time = (accumulationMode == HIPAccumulationMode::FinalStill)
                    ? 0.0f
                    : static_cast<float>(previewDispatchCount) * (1.0f / 60.0f);
        pp.width = static_cast<float>(width);
        pp.height = static_cast<float>(height);
        pp.debugView = static_cast<std::uint32_t>(debugView);
        return pp;
    }

    void HashCameraForAccumulation(HashBuilder64 &hash, const CameraDataCPU &cameraCPU)
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

    void HashLight(HashBuilder64 &hash, const Light &light)
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
        hash.addBool(light.castShadows);
        hash.addU32(light.ownerId);
    }

    void HashMaterial(HashBuilder64 &hash, const SceneMetaMaterial &material)
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

    void HashDecal(HashBuilder64 &hash, const SceneMetaDecal &decal)
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

    std::uint64_t ComputeMetaSignature(const SceneMetaResources *metaRes)
    {
        if (!metaRes)
            return 0u;

        HashBuilder64 hash;
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

        hash.addU64(static_cast<std::uint64_t>(metaRes->airDustVolumes.size()));
        for (const SceneMetaAirDustVolume &volume : metaRes->airDustVolumes)
        {
            hash.addVec3(volume.position);
            hash.addVec3(volume.extent);
            hash.addVec3(volume.linkedLightPosition);
            hash.addVec3(volume.linkedLightColor);
            hash.addFloat(volume.linkedLightIntensity);
            hash.addFloat(volume.linkedLightRadius);
        }

        hash.addBool(metaRes->hasPostProcess);
        hash.addBool(metaRes->hasFog);
        hash.addFloat(metaRes->worldUnitToMeters);
        return hash.value;
    }

    std::uint64_t ComputeAccumulationStateHash(const CameraDataCPU &cameraCPU,
                                               int width,
                                               int height,
                                               std::uint64_t sceneRevision,
                                               const std::vector<Light> &lights,
                                               const SceneMetaResources *metaRes,
                                               HIPAccumulationMode accumulationMode,
                                               HIPDebugView debugView)
    {
        HashBuilder64 hash;
        HashCameraForAccumulation(hash, cameraCPU);
        hash.addI32(width);
        hash.addI32(height);
        hash.addU64(sceneRevision);
        hash.addU32(static_cast<std::uint32_t>(accumulationMode));
        hash.addU32(static_cast<std::uint32_t>(debugView));

        hash.addU64(static_cast<std::uint64_t>(lights.size()));
        for (const Light &light : lights)
            HashLight(hash, light);

        hash.addU64(ComputeMetaSignature(metaRes));
        return hash.value;
    }

    std::vector<HIPLightGPU> ConvertLightsToGPU(const std::vector<Light> &lights)
    {
        std::vector<HIPLightGPU> gpuLights;
        gpuLights.reserve(lights.size());

        for (const Light &src : lights)
        {
            HIPLightGPU dst{};
            dst.type = static_cast<int>(src.type);
            dst.flags = src.castShadows ? HIP_LIGHT_FLAG_CASTS_SHADOW : 0u;
            dst.position = src.position;
            dst.direction = src.direction;
            dst.color = src.color;
            dst.intensity = src.intensity;
            dst.radius = src.radius;
            dst.sourceLength = src.sourceLength;
            dst.softSourceRadius = src.softSourceRadius;
            dst.spotSize = src.spotSize;
            dst.spotBlend = src.spotBlend;
            dst.attenuationRadius = src.attenuationRadius;
            dst.ownerId = src.ownerId;
            gpuLights.push_back(dst);
        }

        return gpuLights;
    }

    std::vector<HIPSceneInstanceGPU> ConvertInstancesToGPU(const std::vector<SceneInstanceGPU> &instances)
    {
        std::vector<HIPSceneInstanceGPU> gpuInstances(instances.size());
        if (!instances.empty())
        {
            std::memcpy(gpuInstances.data(),
                        instances.data(),
                        instances.size() * sizeof(HIPSceneInstanceGPU));
        }
        return gpuInstances;
    }

    bool EnsureMetaResourcesCached(const SceneMetaResources *metaRes)
    {
        const std::uint64_t signature = ComputeMetaSignature(metaRes);
        if (g_metaCache.metaRes == metaRes && g_metaCache.signature == signature)
            return true;

        g_metaCache.clear();
        g_metaCache.metaRes = metaRes;
        g_metaCache.signature = signature;
        ++g_metaCache.generation;

        if (!metaRes)
            return true;

        const HIPTextureIndexMapping textureMapping = BuildCombinedTextureCache(metaRes);

        g_metaCache.textureDescs.reserve(g_metaCache.combinedTexturePaths.size());

        for (std::size_t i = 0; i < g_metaCache.combinedTexturePaths.size(); ++i)
        {
            HIPTextureDescGPU desc{};
            desc.flags = (i < g_metaCache.combinedTextureFlags.size()) ? g_metaCache.combinedTextureFlags[i] : 0u;
            desc.texelOffset = static_cast<std::uint64_t>(g_metaCache.textureTexels.size());

            const std::string &path = g_metaCache.combinedTexturePaths[i];
            int width = 0;
            int height = 0;
            int channels = 0;
            stbi_uc *pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
            if (pixels != nullptr && width > 0 && height > 0)
            {
                desc.width = static_cast<std::uint32_t>(width);
                desc.height = static_cast<std::uint32_t>(height);
                const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
                g_metaCache.textureTexels.insert(g_metaCache.textureTexels.end(), pixels, pixels + bytes);
            }
            else
            {
                desc.width = 1u;
                desc.height = 1u;
                g_metaCache.textureTexels.push_back(255u);
                g_metaCache.textureTexels.push_back(0u);
                g_metaCache.textureTexels.push_back(255u);
                g_metaCache.textureTexels.push_back(255u);
                std::cerr << "HIPRenderer: failed to load texture, using fallback: " << path << "\n";
            }
            if (pixels != nullptr)
                stbi_image_free(pixels);

            g_metaCache.textureDescs.push_back(desc);
        }

        const auto mapBaseColorTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.srgbColorToCombined);
        };

        const auto mapEmissionTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.srgbColorToCombined);
        };

        const auto mapNormalTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.linearDataToCombined);
        };

        const auto mapOrmTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.linearDataToCombined);
        };

        const auto mapRoughnessTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.linearDataToCombined);
        };

        const auto mapMetallicTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.linearDataToCombined);
        };

        const auto mapOcclusionTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.linearDataToCombined);
        };

        const auto mapSpecialColorTextureIndex = [&](int idx) -> std::int32_t
        {
            return MapCombinedTextureIndex(idx, textureMapping.srgbColorToCombined);
        };

        const std::size_t materialCount = metaRes->materials.size();
        const std::size_t materialPBRSourceCount =
            metaRes->materialsPBR.empty() ? metaRes->materials.size() : metaRes->materialsPBR.size();
        const std::size_t materialPBRCount = std::max(materialCount, materialPBRSourceCount);

        if (!metaRes->materialsPBR.empty() && metaRes->materialsPBR.size() != metaRes->materials.size())
        {
            std::cerr << "HIPRenderer: materialsPBR count (" << metaRes->materialsPBR.size()
                      << ") differs from materials count (" << metaRes->materials.size()
                      << "); missing PBR entries will fall back to the base material table\n";
        }

        g_metaCache.materials.resize(std::max(materialCount, std::size_t{1}));
        g_metaCache.materialsPBR.resize(std::max(materialPBRCount, std::size_t{1}));

        for (std::size_t i = 0; i < materialCount; ++i)
        {
            const SceneMetaMaterial &material = metaRes->materials[i];

            HIPMaterialGPU simpleMat{};
            simpleMat.baseColorTexIndex = mapBaseColorTextureIndex(material.baseColorTexIndex);
            simpleMat.emissionTexIndex = mapEmissionTextureIndex(material.emissionTexIndex);
            simpleMat.baseColorUvSet = std::clamp(material.baseColorUvSet, 0, 2);
            simpleMat.emissionUvSet = std::clamp(material.emissionUvSet, 0, 2);
            g_metaCache.materials[i] = simpleMat;
        }

        for (std::size_t i = 0; i < materialPBRCount; ++i)
        {
            const SceneMetaMaterial *materialPtr = nullptr;
            SceneMetaMaterial defaultMaterial{};
            if (i < metaRes->materialsPBR.size())
                materialPtr = &metaRes->materialsPBR[i];
            else if (i < metaRes->materials.size())
                materialPtr = &metaRes->materials[i];
            else
                materialPtr = &defaultMaterial;
            const SceneMetaMaterial &material = *materialPtr;

            HIPMaterialPBRGPU pbr{};
            pbr.baseColorTexIndex = mapBaseColorTextureIndex(material.baseColorTexIndex);
            pbr.emissionTexIndex = mapEmissionTextureIndex(material.emissionTexIndex);
            pbr.normalTexIndex = mapNormalTextureIndex(material.normalTexIndex);
            pbr.ormTexIndex = mapOrmTextureIndex(material.ormTexIndex);
            pbr.roughnessTexIndex = mapRoughnessTextureIndex(material.roughnessTexIndex);
            pbr.metallicTexIndex = mapMetallicTextureIndex(material.metallicTexIndex);
            pbr.occlusionTexIndex = mapOcclusionTextureIndex(material.occlusionTexIndex);
            pbr.baseColorUvSet = std::clamp(material.baseColorUvSet, 0, 2);
            pbr.emissionUvSet = std::clamp(material.emissionUvSet, 0, 2);
            pbr.normalUvSet = std::clamp(material.normalUvSet, 0, 2);
            pbr.ormUvSet = std::clamp(material.ormUvSet, 0, 2);
            pbr.roughnessUvSet = std::clamp(material.roughnessUvSet, 0, 2);
            pbr.metallicUvSet = std::clamp(material.metallicUvSet, 0, 2);
            pbr.occlusionUvSet = std::clamp(material.occlusionUvSet, 0, 2);
            pbr.specialModel = material.specialModel;
            pbr.specialTex0Index = mapSpecialColorTextureIndex(material.specialTex0Index);
            pbr.specialTex1Index = mapSpecialColorTextureIndex(material.specialTex1Index);
            pbr.specialScalar0 = material.specialScalar0;
            pbr.specialScalar1 = material.specialScalar1;
            pbr.specialScalar2 = material.specialScalar2;
            pbr.specialScalar3 = material.specialScalar3;
            pbr.specialScalar4 = material.specialScalar4;
            pbr.specialScalar5 = material.specialScalar5;
            if (material.emissionUseAlphaMask)
                pbr.flags |= HIP_MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK;
            if (material.thinEmissiveSurface)
                pbr.flags |= HIP_MATERIAL_FLAG_THIN_EMISSIVE_SURFACE;
            g_metaCache.materialsPBR[i] = pbr;
        }

        g_metaCache.decals = BuildDecalTable(metaRes, textureMapping);
        g_metaCache.airDustVolumes = BuildAirDustVolumeTable(metaRes);

        return true;
    }
}

bool InitHIPRenderer()
{
    return true;
}

bool PreloadHIPSceneResources(const SceneMetaResources *metaRes)
{
    return EnsureMetaResourcesCached(metaRes);
}

bool RenderFrameHIPTexture(const std::vector<BVHNode> &tlasNodes,
                           const std::vector<BVHNode> &meshNodes,
                           const std::vector<Triangle> &tris,
                           const std::vector<SceneInstanceGPU> &instances,
                           const std::vector<Light> &lights,
                           std::uint64_t sceneRevision,
                           HIPAccumulationMode accumulationMode,
                           HIPDebugView debugView,
                           int rootIndex,
                           const CameraDataCPU &cameraCPU,
                           const SceneMetaResources *metaRes,
                           std::vector<Vec3> &framebuffer)
{
#ifndef USE_HIP_RENDERER
    (void)tlasNodes;
    (void)meshNodes;
    (void)tris;
    (void)instances;
    (void)lights;
    (void)sceneRevision;
    (void)accumulationMode;
    (void)debugView;
    (void)rootIndex;
    (void)cameraCPU;
    (void)metaRes;
    (void)framebuffer;
    std::cerr << "RenderFrameHIPTexture: project was built without USE_HIP_RENDERER\n";
    return false;
#else
    const auto totalStart = ProfileClock::now();

    if (cameraCPU.width <= 0 || cameraCPU.height <= 0)
    {
        std::cerr << "RenderFrameHIPTexture: invalid frame size\n";
        return false;
    }

    if (tlasNodes.empty() || meshNodes.empty() || tris.empty() || instances.empty())
    {
        std::cerr << "RenderFrameHIPTexture: scene BVH/geometry is incomplete\n";
        return false;
    }

    HIPTextureFrameProfile profile{};
    profile.callIndex = ++g_textureProfileCallIndexHost;
    profile.accumulatedSamples = g_accumulatedSampleCountHost;
    profile.samplesThisDispatch =
        static_cast<std::uint32_t>(std::max(cameraCPU.samplesPerPixel, 1));
    profile.accumulationMode = accumulationMode;
    profile.debugView = debugView;
    profile.prototypeTriangles = ToU32Count(tris.size());
    profile.totalInstances = ToU32Count(instances.size());
    profile.tlasNodeCount = ToU32Count(tlasNodes.size());
    profile.blasNodeCount = ToU32Count(meshNodes.size());
    profile.lightCount = ToU32Count(lights.size());

    const auto metaStart = ProfileClock::now();
    if (!EnsureMetaResourcesCached(metaRes))
    {
        profile.ensureMetaMs = ToMilliseconds(ProfileClock::now() - metaStart);
        std::cerr << "RenderFrameHIPTexture: failed to prepare meta resources\n";
        return false;
    }
    profile.ensureMetaMs = ToMilliseconds(ProfileClock::now() - metaStart);
    profile.materialCount = ToU32Count(g_metaCache.materials.size());
    profile.materialPBRCount = ToU32Count(g_metaCache.materialsPBR.size());
    profile.textureCount = ToU32Count(g_metaCache.textureDescs.size());
    profile.decalCount = ToU32Count(g_metaCache.decals.size());
    profile.airDustVolumeCount = ToU32Count(g_metaCache.airDustVolumes.size());
    PrintHIPMaterialDebugOnce(metaRes);

    const auto accumStateStart = ProfileClock::now();
    const std::uint64_t accumulationHash =
        ComputeAccumulationStateHash(cameraCPU,
                                     cameraCPU.width,
                                     cameraCPU.height,
                                     sceneRevision,
                                     lights,
                                     metaRes,
                                     accumulationMode,
                                     debugView);

    if (accumulationHash != g_accumulationStateHashHost)
    {
        ResetHIPAccumulation();
        g_accumulationStateHashHost = accumulationHash;
        profile.accumulatedSamples = 0u;
        profile.callIndex = 1u;
        g_textureProfileCallIndexHost = profile.callIndex;
    }
    profile.accumulationStateMs = ToMilliseconds(ProfileClock::now() - accumStateStart);

    const auto resizeStart = ProfileClock::now();
    framebuffer.resize(static_cast<std::size_t>(cameraCPU.width) * static_cast<std::size_t>(cameraCPU.height));
    profile.framebufferResizeMs = ToMilliseconds(ProfileClock::now() - resizeStart);

    const auto instanceStart = ProfileClock::now();
    const std::vector<HIPSceneInstanceGPU> gpuInstances = ConvertInstancesToGPU(instances);
    profile.instanceConvertMs = ToMilliseconds(ProfileClock::now() - instanceStart);

    const auto lightStart = ProfileClock::now();
    const std::vector<HIPLightGPU> gpuLights = ConvertLightsToGPU(lights);
    profile.lightConvertMs = ToMilliseconds(ProfileClock::now() - lightStart);

    const auto emissiveStart = ProfileClock::now();
    const std::vector<HIPEmissiveTriangleGPU> emissiveTriangles =
        BuildEmissiveTriangleTable(tris, instances, metaRes);
    profile.emissiveBuildMs = ToMilliseconds(ProfileClock::now() - emissiveStart);
    profile.emissiveTriangleCount = ToU32Count(emissiveTriangles.size());

    const auto postParamsStart = ProfileClock::now();
    const HIPPostProcessParams postParams =
        BuildPostProcessParams(metaRes,
                               cameraCPU,
                               cameraCPU.width,
                               cameraCPU.height,
                               g_previewDispatchCountHost,
                               accumulationMode,
                               debugView);
    profile.postParamsMs = ToMilliseconds(ProfileClock::now() - postParamsStart);

    const auto hipCoreStart = ProfileClock::now();
    const bool ok = HIP_RenderFrameTexture_C(tlasNodes.data(),
                                             static_cast<std::uint32_t>(tlasNodes.size()),
                                             meshNodes.data(),
                                             static_cast<std::uint32_t>(meshNodes.size()),
                                             tris.data(),
                                             static_cast<std::uint32_t>(tris.size()),
                                             gpuInstances.empty() ? nullptr : gpuInstances.data(),
                                             static_cast<std::uint32_t>(gpuInstances.size()),
                                             gpuLights.empty() ? nullptr : gpuLights.data(),
                                             static_cast<std::uint32_t>(gpuLights.size()),
                                             rootIndex,
                                             &cameraCPU,
                                             g_metaCache.materials.empty() ? nullptr : g_metaCache.materials.data(),
                                             static_cast<std::uint32_t>(g_metaCache.materials.size()),
                                             g_metaCache.materialsPBR.empty() ? nullptr : g_metaCache.materialsPBR.data(),
                                             static_cast<std::uint32_t>(g_metaCache.materialsPBR.size()),
                                             emissiveTriangles.empty() ? nullptr : emissiveTriangles.data(),
                                             static_cast<std::uint32_t>(emissiveTriangles.size()),
                                             g_metaCache.decals.empty() ? nullptr : g_metaCache.decals.data(),
                                             static_cast<std::uint32_t>(g_metaCache.decals.size()),
                                             g_metaCache.airDustVolumes.empty() ? nullptr : g_metaCache.airDustVolumes.data(),
                                             static_cast<std::uint32_t>(g_metaCache.airDustVolumes.size()),
                                             &postParams,
                                             g_metaCache.textureDescs.empty() ? nullptr : g_metaCache.textureDescs.data(),
                                             static_cast<std::uint32_t>(g_metaCache.textureDescs.size()),
                                             g_metaCache.textureTexels.empty() ? nullptr : g_metaCache.textureTexels.data(),
                                             g_metaCache.textureTexels.size(),
                                             sceneRevision,
                                             g_metaCache.generation,
                                             g_accumulatedSampleCountHost,
                                             framebuffer.data());
    profile.hipCoreMs = ToMilliseconds(ProfileClock::now() - hipCoreStart);
    (void)HIP_GetLastTextureExecutionProfile_C(&profile.lowLevel);

    if (!ok)
    {
        std::cerr << "RenderFrameHIPTexture: HIP_RenderFrameTexture_C returned false\n";
        return false;
    }

    g_accumulatedSampleCountHost += static_cast<std::uint32_t>(std::max(cameraCPU.samplesPerPixel, 1));
    if (accumulationMode == HIPAccumulationMode::PreviewProgressive)
        ++g_previewDispatchCountHost;
    else
        g_previewDispatchCountHost = 0u;

    profile.totalMs = ToMilliseconds(ProfileClock::now() - totalStart);
    PrintTextureFrameProfile(profile);
    return true;
#endif
}

void ResetHIPAccumulation()
{
    g_accumulationStateHashHost = kInvalidAccumulationHash;
    g_accumulatedSampleCountHost = 0u;
    g_previewDispatchCountHost = 0u;
    g_textureProfileTotalsHost = {};
    g_textureProfileCallIndexHost = 0u;
#ifdef USE_HIP_RENDERER
    HIP_ResetAccumulation_C();
#endif
}
