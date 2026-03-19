#include "SceneMetaLoader.h"
#include "Scene.h"
#include "Camera.h"
#include "SceneObject.h"
#include "Light.h"

#include <../ExternalLibs/json-develop/single_include/nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <string_view>
#include <algorithm>
#include <cctype>
#include <array>
#include <cmath>

namespace
{
    using json = nlohmann::json;

    constexpr float kPI = 3.14159265358979323846f;

    static Vec3 ReadVec3(const json& j, const Vec3& def)
    {
        if (!j.is_array() || j.size() < 3)
            return def;

        Vec3 v;
        v.x = (float)j[0].get<double>();
        v.y = (float)j[1].get<double>();
        v.z = (float)j[2].get<double>();
        return v;
    }

    static Vec3 ReadVec3Field(const json& obj, const char* key, const Vec3& def)
    {
        auto it = obj.find(key);
        if (it == obj.end())
            return def;
        return ReadVec3(*it, def);
    }

    static float ReadFloatField(const json& obj, const char* key, float def)
    {
        auto it = obj.find(key);
        if (it == obj.end() || !it->is_number())
            return def;
        return (float)it->get<double>();
    }

    static std::string ReadStringField(const json& obj, const char* key, const std::string& def = {})
    {
        auto it = obj.find(key);
        if (it == obj.end() || !it->is_string())
            return def;
        return it->get<std::string>();
    }


    static float ReadNamedScalarParam(const json& obj, const char* wantedName, float def)
    {
        auto it = obj.find("scalar_params");
        if (it == obj.end() || !it->is_array())
            return def;

        for (const json& p : *it)
        {
            if (!p.is_object())
                continue;
            const std::string name = ReadStringField(p, "name", std::string{});
            if (name == wantedName)
                return ReadFloatField(p, "value", def);
        }
        return def;
    }

    static Vec3 ReadNamedVectorParam(const json& obj, const char* wantedName, const Vec3& def)
    {
        auto it = obj.find("vector_params");
        if (it == obj.end() || !it->is_array())
            return def;

        for (const json& p : *it)
        {
            if (!p.is_object())
                continue;
            const std::string name = ReadStringField(p, "name", std::string{});
            if (name == wantedName)
            {
                auto itV = p.find("value");
                if (itV != p.end())
                    return ReadVec3(*itV, def);
                break;
            }
        }
        return def;
    }

    static const json* FindNamedTextureParam(const json& obj, std::initializer_list<const char*> wantedNames)
    {
        auto it = obj.find("texture_params");
        if (it == obj.end() || !it->is_array())
            return nullptr;

        for (const char* wanted : wantedNames)
        {
            for (const json& p : *it)
            {
                if (!p.is_object())
                    continue;
                if (ReadStringField(p, "name", std::string{}) == wanted)
                    return &p;
            }
        }
        return nullptr;
    }

    static std::string ReadTextureParamExportedPath(const json& obj,
                                                    std::initializer_list<const char*> wantedNames,
                                                    bool* outSRGB = nullptr)
    {
        if (outSRGB)
            *outSRGB = false;

        const json* p = FindNamedTextureParam(obj, wantedNames);
        if (!p)
            return std::string{};

        if (outSRGB)
        {
            auto it = p->find("srgb");
            if (it != p->end() && it->is_boolean())
                *outSRGB = it->get<bool>();
        }

        return ReadStringField(*p, "exported_path", std::string{});
    }

    static std::string ResolveTexturePath(const std::filesystem::path& metaDir,
                                          const std::string& relOrAbs)
    {
        if (relOrAbs.empty())
            return std::string{};

        std::filesystem::path p(relOrAbs);
        if (p.is_absolute())
            return p.lexically_normal().string();

        return (metaDir / p).lexically_normal().string();
    }

    static uint8_t ChannelFromString(const json& obj, const char* key, uint8_t def)
    {
        auto it = obj.find(key);
        if (it == obj.end() || !it->is_string())
            return def;

        const std::string s = it->get<std::string>();
        if (s.empty()) return def;

        const char c = (char)std::tolower((unsigned char)s[0]);
        switch (c)
        {
            case 'r': return 0;
            case 'g': return 1;
            case 'b': return 2;
            case 'a': return 3;
            default:  return def;
        }
    }

    static int32_t ReadIntField(const json& obj, const char* key, int32_t def)
    {
        auto it = obj.find(key);
        if (it == obj.end())
            return def;
        if (it->is_number_integer())
            return (int32_t)it->get<long long>();
        if (it->is_number())
            return (int32_t)it->get<double>();
        return def;
    }

    static bool ParsePostProcessFromJson(const json& obj, SceneMetaPostProcess& outPP)
    {
        if (!obj.is_object())
            return false;

        outPP.autoExposureMethod  = ReadIntField(obj, "auto_exposure_method", 0);
        outPP.autoExposureBias    = ReadFloatField(obj, "auto_exposure_bias", 0.0f);
        outPP.autoExposureMin     = ReadFloatField(obj, "auto_exposure_min_brightness", 1.0f);
        outPP.autoExposureMax     = ReadFloatField(obj, "auto_exposure_max_brightness", 1.0f);

        outPP.bloomIntensity      = ReadFloatField(obj, "bloom_intensity", 0.0f);
        outPP.bloomThreshold      = ReadFloatField(obj, "bloom_threshold", 0.0f);
        outPP.vignetteIntensity   = ReadFloatField(obj, "vignette_intensity", 0.0f);
        outPP.chromaticAberration = ReadFloatField(obj, "chromatic_aberration", 0.0f);
        outPP.filmGrainIntensity  = ReadFloatField(obj, "film_grain_intensity", 0.0f);

        outPP.filmSlope           = ReadFloatField(obj, "film_slope", outPP.filmSlope);
        outPP.filmToe             = ReadFloatField(obj, "film_toe", outPP.filmToe);
        outPP.filmShoulder        = ReadFloatField(obj, "film_shoulder", outPP.filmShoulder);
        outPP.filmBlackClip       = ReadFloatField(obj, "film_black_clip", outPP.filmBlackClip);
        outPP.filmWhiteClip       = ReadFloatField(obj, "film_white_clip", outPP.filmWhiteClip);

        if (auto itSat = obj.find("color_saturation"); itSat != obj.end() && itSat->is_array() && itSat->size() >= 3)
        {
            outPP.colorSaturation = ReadVec3(*itSat, Vec3{1.0f, 1.0f, 1.0f});
        }
        return true;
    }

    static bool ParseFogFromJson(const json& obj, SceneMetaFog& outFog)
    {
        if (!obj.is_object())
            return false;

        outFog.fogDensity        = ReadFloatField(obj, "fog_density", 0.0f);
        outFog.heightFalloff     = ReadFloatField(obj, "fog_height_falloff", 0.0f);
        outFog.inscatteringColor = ReadVec3Field(obj, "fog_inscattering_color", Vec3{1,1,1});

        if (auto itV = obj.find("volumetric_fog"); itV != obj.end())
            outFog.volumetricFog = itV->get<bool>();

        outFog.scatteringG      = ReadFloatField(obj, "volumetric_scattering_distribution", 0.0f);
        outFog.volumetricAlbedo = ReadVec3Field(obj, "volumetric_albedo", Vec3{1,1,1});
        outFog.extinctionScale  = ReadFloatField(obj, "volumetric_extinction_scale", 1.0f);
        return true;
    }

    static SceneMetaCameraInfo ParseCameraInfoFromJson(const json& jc, float unitScale)
    {
        SceneMetaCameraInfo c{};
        c.name     = ReadStringField(jc, "name", std::string{});

        c.position = ReadVec3Field(jc, "position", Vec3{0,0,0});
        c.forward  = ReadVec3Field(jc, "forward",  Vec3{0,-1,0});
        c.up       = ReadVec3Field(jc, "up",       Vec3{0,0,1});
        c.right    = ReadVec3Field(jc, "right",    Vec3{1,0,0});

        c.position.x *= unitScale; c.position.y *= unitScale; c.position.z *= unitScale;

        c.fovY      = ReadFloatField(jc, "fov_y", 60.0f * kPI / 180.0f);
        c.clipStart = ReadFloatField(jc, "clip_start", 0.1f) * unitScale;
        c.clipEnd   = ReadFloatField(jc, "clip_end",   1000.0f) * unitScale;
        c.focusDistance = ReadFloatField(jc, "focus_distance", 10.0f) * unitScale;

        if (auto itPP = jc.find("post_process"); itPP != jc.end() && itPP->is_object())
            c.hasPostProcess = ParsePostProcessFromJson(*itPP, c.postProcess);

        if (auto itFog = jc.find("fog"); itFog != jc.end() && itFog->is_object())
            c.hasFog = ParseFogFromJson(*itFog, c.fog);

        return c;
    }


    static inline std::string_view trimView(std::string_view s)
    {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n'))
            s.remove_prefix(1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n'))
            s.remove_suffix(1);
        return s;
    }

    static inline bool allDigits(std::string_view s)
    {
        if (s.empty()) return false;
        for (char c : s)
        {
            if (c < '0' || c > '9')
                return false;
        }
        return true;
    }

    // Normalize material/object keys: lower-case, basename, strip common UE suffixes, strip ".001"/"_001".
    static std::string NormalizeKey(std::string_view name)
    {
        name = trimView(name);

        // basename if path-like
        const std::size_t slash = name.find_last_of("/\\");
        if (slash != std::string_view::npos)
            name = name.substr(slash + 1);

        auto stripSuffix = [&](std::string_view suf)
        {
            if (name.size() >= suf.size() && name.substr(name.size() - suf.size()) == suf)
                name = name.substr(0, name.size() - suf.size());
        };

        stripSuffix("_INST");
        stripSuffix("_Inst");
        stripSuffix("_instance");
        stripSuffix("_Instance");
        stripSuffix("_C");

        // Strip numeric suffix ".001" (Blender-like) / "_001"
        {
            const std::size_t dot = name.find_last_of('.');
            if (dot != std::string_view::npos)
            {
                const std::string_view tail = name.substr(dot + 1);
                if (tail.size() <= 4 && allDigits(tail))
                    name = name.substr(0, dot);
            }
        }
        {
            const std::size_t us = name.find_last_of('_');
            if (us != std::string_view::npos)
            {
                const std::string_view tail = name.substr(us + 1);
                if (tail.size() <= 4 && allDigits(tail))
                    name = name.substr(0, us);
            }
        }

        std::string out;
        out.reserve(name.size());
        for (char c : name)
            out.push_back((char)std::tolower((unsigned char)c));
        return out;
    }

    static bool LooksEmissive(std::string_view nameLower)
    {
        return (nameLower.find("emissive") != std::string_view::npos) ||
               (nameLower.find("emission") != std::string_view::npos) ||
               (nameLower.find("glow") != std::string_view::npos);
    }
static std::string ToLowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static bool ContainsAny(std::string_view s, std::initializer_list<const char*> needles)
{
    for (const char* n : needles)
    {
        if (s.find(n) != std::string_view::npos)
            return true;
    }
    return false;
}

static bool IsUtilityTexturePathOrName(std::string_view s)
{
    return ContainsAny(s, {
        "_rma", "rma_", "/rma",
        "_orm", "orm_", "/orm",
        "_arm", "arm_", "/arm",
        "rough", "roughness", "metal", "metallic",
        "occlusion", "ambientocclusion", "_ao", "ao_",
        "normal", "_nrm", "nrm_", "_nrm_"
    });
}

static bool IsExplicitEmissiveMaterialName(std::string_view s)
{
    return ContainsAny(s, {
        "emissive", "light_on", "lighton", "neon", "headlight", "bulb_lit"
    });
}

static bool IsLikelyLightHousingMaterialName(std::string_view s)
{
    return ContainsAny(s, {
        "tunnel_light", "overhead_light", "ceiling_light", "light_fixture", "light_housing"
    });
}

static bool IsStrongNonEmissiveMaterialName(std::string_view s)
{
    return ContainsAny(s, {
        "pipe", "pipes", "wire", "wires", "cable", "wall", "floor",
        "rail", "track", "concrete", "vent", "metal_01a"
    });
}

static bool IsStrongEmissiveTexturePathOrName(std::string_view s)
{
    return ContainsAny(s, {
        "emissive", "_emm", "emm_", "glow", "signal", "traffic", "lamp", "bulb"
    });
}

static bool IsSpecialVisibleEmissiveSurface(std::string_view materialNameLower,
                                            std::string_view baseMaterialPathLower)
{
    return IsExplicitEmissiveMaterialName(materialNameLower) ||
           ContainsAny(baseMaterialPathLower, {"/mm_emissive_", "mm_emissive_", "headlight"});
}

    static bool LooksFogLike(std::string_view nameLower)
    {
        return (nameLower.find("fog") != std::string_view::npos) ||
               (nameLower.find("godray") != std::string_view::npos) ||
               (nameLower.find("god_ray") != std::string_view::npos);
    }

    static bool IsUnsupportedBlendModeForCurrentTracer(int32_t blendMode)
    {
        // UE: 0=Opaque, 1=Masked. 2+ are translucent/additive/modulate-like and
        // should not be traced as solid opaque geometry in the current renderer.
        return blendMode >= 2;
    }

    // Access mutable triangles without changing SceneObject.h
    static inline std::vector<Triangle>& MutableTriangles(SceneObject* obj)
    {
        return const_cast<std::vector<Triangle>&>(obj->getTriangles());
    }


    static inline bool IsSceneExportFormat(const json& j)
    {
        auto it = j.find("format");
        return (it != j.end() && it->is_string() && it->get<std::string>() == "SceneRTXSceneExport");
    }

    static float GetSceneExportUnitScale(const json& j)
    {
        auto itW = j.find("world");
        if (itW != j.end() && itW->is_object())
        {
            const float s = ReadFloatField(*itW, "unit_scale", 1.0f);
            return (s > 0.0f) ? s : 1.0f;
        }
        return 1.0f;
    }

    static bool ReadBoolFieldLoose(const json& obj, const char* key, bool def)
    {
        auto it = obj.find(key);
        if (it == obj.end())
            return def;
        if (it->is_boolean())
            return it->get<bool>();
        if (it->is_number_integer())
            return it->get<long long>() != 0;
        if (it->is_number())
            return it->get<double>() != 0.0;
        return def;
    }

    static float ReadSceneScalarParam(const json& obj,
                                      std::initializer_list<const char*> wantedNames,
                                      float def)
    {
        auto it = obj.find("scalar_parameters");
        if (it == obj.end() || !it->is_array())
            return def;

        for (const char* wanted : wantedNames)
        {
            for (const json& p : *it)
            {
                if (!p.is_object())
                    continue;
                const std::string name = ReadStringField(p, "name", std::string{});
                if (name == wanted)
                    return ReadFloatField(p, "value", def);
            }
        }
        return def;
    }

    static bool ReadSceneVectorParam(const json& obj,
                                     std::initializer_list<const char*> wantedNames,
                                     Vec3& outValue)
    {
        auto it = obj.find("vector_parameters");
        if (it == obj.end() || !it->is_array())
            return false;

        for (const char* wanted : wantedNames)
        {
            for (const json& p : *it)
            {
                if (!p.is_object())
                    continue;
                const std::string name = ReadStringField(p, "name", std::string{});
                if (name != wanted)
                    continue;

                auto itV = p.find("value");
                if (itV == p.end())
                    return false;

                outValue = ReadVec3(*itV, Vec3{0.0f, 0.0f, 0.0f});
                return true;
            }
        }

        return false;
    }

    static std::string ReadSceneTextureParamId(const json& obj,
                                               std::initializer_list<const char*> wantedNames)
    {
        auto it = obj.find("texture_parameters");
        if (it == obj.end() || !it->is_array())
            return std::string{};

        for (const char* wanted : wantedNames)
        {
            for (const json& p : *it)
            {
                if (!p.is_object())
                    continue;
                const std::string name = ReadStringField(p, "name", std::string{});
                if (name != wanted)
                    continue;

                std::string texId = ReadStringField(p, "texture_id", std::string{});
                if (!texId.empty())
                    return texId;
            }
        }

        return std::string{};
    }

    static std::array<float, 16> ReadMatrix4Field(const json& obj, const char* key)
    {
        std::array<float, 16> m{
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1
        };

        auto it = obj.find(key);
        if (it == obj.end() || !it->is_array() || it->size() < 16)
            return m;

        for (std::size_t i = 0; i < 16; ++i)
            m[i] = (float)(*it)[i].get<double>();

        return m;
    }

    static Vec3 MatrixPosition(const std::array<float,16>& m)
    {
        return Vec3{m[3], m[7], m[11]};
    }

    // Transform matrix is row-major affine with translation in indices 3/7/11.
    // Basis vectors are extracted from COLUMNS (UE local axes in world space).
    static Vec3 MatrixAxisX(const std::array<float,16>& m)
    {
        return Vec3{m[0], m[4], m[8]};
    }

    static Vec3 MatrixAxisY(const std::array<float,16>& m)
    {
        return Vec3{m[1], m[5], m[9]};
    }

    static Vec3 MatrixAxisZ(const std::array<float,16>& m)
    {
        return Vec3{m[2], m[6], m[10]};
    }

    static float Length3(const Vec3& v)
    {
        const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (!(lenSq > 0.0f) || !std::isfinite(lenSq))
            return 0.0f;
        return std::sqrt(lenSq);
    }

    static Vec3 NormalizeSafe(const Vec3& v, const Vec3& def)
    {
        const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (!(lenSq > 0.0f) || !std::isfinite(lenSq))
            return def;

        const float invLen = 1.0f / std::sqrt(lenSq);
        return Vec3{v.x * invLen, v.y * invLen, v.z * invLen};
    }
static float Dot3(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 Cross3(const Vec3& a, const Vec3& b)
{
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static Vec3 OrthoNormalizeUp(const Vec3& forward, const Vec3& upHint)
{
    Vec3 up = upHint;
    const float proj = Dot3(up, forward);
    up.x -= forward.x * proj;
    up.y -= forward.y * proj;
    up.z -= forward.z * proj;

    up = NormalizeSafe(up, Vec3{0, 0, 1});
    if (std::abs(Dot3(up, forward)) > 0.999f)
    {
        const Vec3 fallbackRight = NormalizeSafe(Cross3(Vec3{0, 0, 1}, forward), Vec3{1, 0, 0});
        up = NormalizeSafe(Cross3(forward, fallbackRight), Vec3{0, 1, 0});
    }
    return up;
}

static SceneMetaCameraInfo ParseSceneExportCameraInfo(const json& jc, float unitScale)
{
    SceneMetaCameraInfo c{};
    c.name = ReadStringField(jc, "name", std::string{});

    const std::array<float,16> m = ReadMatrix4Field(jc, "transform_matrix");

    c.position = ReadVec3Field(jc, "position", MatrixPosition(m));

    const Vec3 forwardExplicit = ReadVec3Field(jc, "forward", MatrixAxisX(m));
    const Vec3 upExplicit      = ReadVec3Field(jc, "up",      MatrixAxisZ(m));
    const Vec3 rightExplicit   = ReadVec3Field(jc, "right",   MatrixAxisY(m));

    c.forward = NormalizeSafe(forwardExplicit, Vec3{1, 0, 0});
    c.up      = OrthoNormalizeUp(c.forward, NormalizeSafe(upExplicit, Vec3{0, 0, 1}));
    c.right   = NormalizeSafe(Cross3(c.forward, c.up), NormalizeSafe(rightExplicit, Vec3{0, 1, 0}));
    c.up      = NormalizeSafe(Cross3(c.right, c.forward), c.up);

    c.position.x *= unitScale;
    c.position.y *= unitScale;
    c.position.z *= unitScale;

    const float aspectRatio = ReadFloatField(jc, "aspect_ratio", 1.0f);
    const float fovYRad     = ReadFloatField(jc, "fov_y", -1.0f);
    const float fovXRad     = ReadFloatField(jc, "fov_x", -1.0f);
    const float fovDeg      = ReadFloatField(jc, "fov", 60.0f);

    if (fovYRad > 0.0f && std::isfinite(fovYRad))
    {
        c.fovY = fovYRad;
    }
    else if (fovXRad > 0.0f && aspectRatio > 0.0f)
    {
        c.fovY = 2.0f * std::atan(std::tan(fovXRad * 0.5f) / aspectRatio);
    }
    else if (aspectRatio > 0.0f)
    {
        const float fovX = fovDeg * kPI / 180.0f;
        c.fovY = 2.0f * std::atan(std::tan(fovX * 0.5f) / aspectRatio);
    }
    else
    {
        c.fovY = fovDeg * kPI / 180.0f;
    }

    c.clipStart = ReadFloatField(jc, "clip_start", 0.1f) * unitScale;
    c.clipEnd   = ReadFloatField(jc, "clip_end", 100000.0f) * unitScale;
    c.focusDistance = ReadFloatField(jc, "focus_distance", 1000.0f) * unitScale;

    if (auto itPP = jc.find("post_process"); itPP != jc.end() && itPP->is_object())
        c.hasPostProcess = ParsePostProcessFromJson(*itPP, c.postProcess);

    if (auto itFog = jc.find("fog"); itFog != jc.end() && itFog->is_object())
        c.hasFog = ParseFogFromJson(*itFog, c.fog);

    return c;
}

static bool LoadCamerasFromSceneExportJson(const json& j,
                                               std::vector<SceneMetaCameraInfo>& outCameras)
    {
        outCameras.clear();

        const float unitScale = GetSceneExportUnitScale(j);

        auto it = j.find("cameras");
        if (it == j.end() || !it->is_array())
            return false;

        outCameras.reserve(it->size());
        for (const json& jc : *it)
        {
            if (!jc.is_object())
                continue;
            outCameras.push_back(ParseSceneExportCameraInfo(jc, unitScale));
        }

        return !outCameras.empty();
    }

    static bool LoadLightsAndMaterialsFromSceneExportJson(const json& j,
                                                          const std::string& metaPath,
                                                          Scene& scene,
                                                          SceneMetaResources* outRes)
    {
        const std::filesystem::path metaDir = std::filesystem::path(metaPath).parent_path();
        const float unitScale = GetSceneExportUnitScale(j);

        bool hasPostProcess = false;
        SceneMetaPostProcess postProcess{};
        if (auto itPP = j.find("postprocess"); itPP != j.end())
        {
            if (itPP->is_array())
            {
                for (const json& pp : *itPP)
                {
                    if (pp.is_object() && ParsePostProcessFromJson(pp, postProcess))
                    {
                        hasPostProcess = true;
                        break;
                    }
                }
            }
            else if (itPP->is_object())
            {
                hasPostProcess = ParsePostProcessFromJson(*itPP, postProcess);
            }
        }

        bool hasFog = false;
        SceneMetaFog fog{};
        if (auto itFog = j.find("fog"); itFog != j.end())
        {
            if (itFog->is_array())
            {
                for (const json& jf : *itFog)
                {
                    if (jf.is_object() && ParseFogFromJson(jf, fog))
                    {
                        hasFog = true;
                        break;
                    }
                }
            }
            else if (itFog->is_object())
            {
                hasFog = ParseFogFromJson(*itFog, fog);
            }
        }

        std::vector<SceneMetaCameraInfo> parsedCameras;
        LoadCamerasFromSceneExportJson(j, parsedCameras);

        std::unordered_map<std::string, std::string> texturePathById;
        std::unordered_map<std::string, std::string> textureNameById;
        std::unordered_map<std::string, bool> textureSrgbById;

        if (auto itT = j.find("textures"); itT != j.end() && itT->is_array())
        {
            texturePathById.reserve(itT->size());
            textureNameById.reserve(itT->size());
            textureSrgbById.reserve(itT->size());

            for (const json& jt : *itT)
            {
                if (!jt.is_object())
                    continue;

                const std::string id = ReadStringField(jt, "stable_id", std::string{});
                if (id.empty())
                    continue;

                const std::string relPath = ReadStringField(jt, "exported_path", std::string{});
                textureNameById[id] = ReadStringField(jt, "name", std::string{});
                texturePathById[id] = ResolveTexturePath(metaDir, relPath);
                textureSrgbById[id] = ReadBoolFieldLoose(jt, "srgb", false);
            }
        }

        std::vector<std::string> baseColorTextures;
        std::unordered_map<std::string, int32_t> baseColorTexIndexByPath;

        std::vector<std::string> linearTextures;
        std::unordered_map<std::string, int32_t> linearTexIndexByPath;

        auto addSrgbTexById = [&](const std::string& texId) -> int32_t
        {
            if (texId.empty())
                return -1;
            auto it = texturePathById.find(texId);
            if (it == texturePathById.end() || it->second.empty())
                return -1;

            auto [itIns, inserted] = baseColorTexIndexByPath.try_emplace(it->second, (int32_t)baseColorTextures.size());
            if (inserted)
                baseColorTextures.push_back(it->second);
            return itIns->second;
        };

        auto addLinearTexById = [&](const std::string& texId) -> int32_t
        {
            if (texId.empty())
                return -1;
            auto it = texturePathById.find(texId);
            if (it == texturePathById.end() || it->second.empty())
                return -1;

            auto [itIns, inserted] = linearTexIndexByPath.try_emplace(it->second, (int32_t)linearTextures.size());
            if (inserted)
                linearTextures.push_back(it->second);
            return itIns->second;
        };

        std::vector<SceneMetaMaterial> parsedMaterials;
        std::unordered_map<std::string, int32_t> materialIndexById;
        std::unordered_map<std::string, int32_t> materialIndexByName;
        std::unordered_map<std::string, int32_t> materialIndexByNorm;

        if (auto itM = j.find("materials"); itM != j.end() && itM->is_array())
        {
            parsedMaterials.reserve(itM->size());
            materialIndexById.reserve(itM->size());
            materialIndexByName.reserve(itM->size());
            materialIndexByNorm.reserve(itM->size());

            for (std::size_t i = 0; i < itM->size(); ++i)
            {
                const json& jm = (*itM)[i];
                if (!jm.is_object())
                    continue;

                SceneMetaMaterial m{};
                m.name = ReadStringField(jm, "name", std::string{});
                if (m.name.empty())
                    m.name = "Material_" + std::to_string(i);

                m.blendMode = ReadIntField(jm, "blend_mode", 0);
                m.twoSided  = ReadBoolFieldLoose(jm, "two_sided", false);

                Vec3 tint{1.0f, 1.0f, 1.0f};
                Vec3 tmp{};
                bool hasAnyTint = false;
                if (ReadSceneVectorParam(jm, {"Base Color"}, tmp))                { tint = Vec3{tint.x * tmp.x, tint.y * tmp.y, tint.z * tmp.z}; hasAnyTint = true; }
                if (ReadSceneVectorParam(jm, {"Albedo Tint", "Albedo TInt"}, tmp)){ tint = Vec3{tint.x * tmp.x, tint.y * tmp.y, tint.z * tmp.z}; hasAnyTint = true; }
                if (ReadSceneVectorParam(jm, {"Base Tint", "Base Tint - Blue"}, tmp)){ tint = Vec3{tint.x * tmp.x, tint.y * tmp.y, tint.z * tmp.z}; hasAnyTint = true; }
                if (ReadSceneVectorParam(jm, {"Color Tint"}, tmp))                { tint = Vec3{tint.x * tmp.x, tint.y * tmp.y, tint.z * tmp.z}; hasAnyTint = true; }
                m.baseColor = hasAnyTint ? tint : Vec3{1.0f, 1.0f, 1.0f};

                Vec3 emissionColor{};
                const bool hasEmissionColor = ReadSceneVectorParam(jm, {"Emissive Color Multi", "Emissive Color 1"}, emissionColor);
                if (hasEmissionColor)
                    m.emissionColor = emissionColor;

                m.metallic  = ReadSceneScalarParam(jm, {"Metallic", "Metallic Value", "Metalness Value", "Metalness"}, 0.0f);
                m.roughness = ReadSceneScalarParam(jm, {"Roughness", "Roughness Value", "Base Roughness"}, 0.5f);
                m.opacity   = ReadSceneScalarParam(jm, {"Opacity Multi", "Opacity"}, 1.0f);

                m.baseColorTexIndex = addSrgbTexById(ReadStringField(jm, "base_color_texture_id", std::string{}));
                if (m.baseColorTexIndex < 0)
                    m.baseColorTexIndex = addSrgbTexById(ReadSceneTextureParamId(jm, {"Base Color", "Albedo", "Albedo ", "Albedo Texture", "USED_001"}));

                std::string emissiveTexId = ReadStringField(jm, "emissive_texture_id", std::string{});
                if (emissiveTexId.empty())
                    emissiveTexId = ReadStringField(jm, "emission_texture_id", std::string{});
                if (emissiveTexId.empty())
                    emissiveTexId = ReadSceneTextureParamId(jm, {"Emissive"});

                m.emissionTexIndex = addSrgbTexById(emissiveTexId);

                m.normalTexIndex = addLinearTexById(ReadStringField(jm, "normal_texture_id", std::string{}));
                if (m.normalTexIndex < 0)
                    m.normalTexIndex = addLinearTexById(ReadSceneTextureParamId(jm, {"Normal", "Normal Map", "MaterialExpressionTextureSampleParameter2D_6", "USED_000"}));

                m.ormTexIndex = addLinearTexById(ReadStringField(jm, "orm_texture_id", std::string{}));
                if (m.ormTexIndex < 0)
                    m.ormTexIndex = addLinearTexById(ReadSceneTextureParamId(jm, {"RMA", "Roughness"}));

                m.roughnessTexIndex = addLinearTexById(ReadStringField(jm, "roughness_texture_id", std::string{}));
                m.metallicTexIndex  = addLinearTexById(ReadStringField(jm, "metallic_texture_id", std::string{}));
                m.occlusionTexIndex = addLinearTexById(ReadStringField(jm, "occlusion_texture_id", std::string{}));

                m.decalTilingU         = ReadSceneScalarParam(jm, {"TilingU"}, 1.0f);
                m.decalTilingV         = ReadSceneScalarParam(jm, {"TilingV"}, 1.0f);
                m.decalOpacityPower    = std::max(0.0f, ReadSceneScalarParam(jm, {"Damage Opacity Power"}, 1.0f));
                m.decalNormalIntensity = std::max(0.0f, ReadSceneScalarParam(jm, {"Normal Intenisty", "Normal Intensity"}, 1.0f));
                m.decalRoughnessBias   = ReadSceneScalarParam(jm, {"RoughMultiply"}, 0.0f);

                const std::string opacityTexId = ReadSceneTextureParamId(jm, {"Opacity Map", "Opacity", "USED_002"});
                if (!opacityTexId.empty())
                {
                    const bool srgb = textureSrgbById.count(opacityTexId) ? textureSrgbById[opacityTexId] : false;
                    if (srgb)
                    {
                        m.decalOpacityTexIndex = addSrgbTexById(opacityTexId);
                        m.decalOpacityTexIsLinear = false;
                    }
                    else
                    {
                        m.decalOpacityTexIndex = addLinearTexById(opacityTexId);
                        m.decalOpacityTexIsLinear = true;
                    }
                }

                const std::string detailTexId = ReadSceneTextureParamId(jm, {"USED_003"});
                if (!detailTexId.empty())
                {
                    const bool srgb = textureSrgbById.count(detailTexId) ? textureSrgbById[detailTexId] : false;
                    if (srgb)
                    {
                        m.decalDetailTexIndex = addSrgbTexById(detailTexId);
                        m.decalDetailTexIsLinear = false;
                    }
                    else
                    {
                        m.decalDetailTexIndex = addLinearTexById(detailTexId);
                        m.decalDetailTexIsLinear = true;
                    }
                }

                const float explicitEmissiveStrength =
                    ReadSceneScalarParam(jm, {"Emissive Strength", "Emissive Intensity", "Emissive Power"}, -1.0f);

                const bool emissionColorNonZero =
                    (std::abs(m.emissionColor.x) > 1e-6f) ||
                    (std::abs(m.emissionColor.y) > 1e-6f) ||
                    (std::abs(m.emissionColor.z) > 1e-6f);

                const std::string materialNameLower = ToLowerCopy(m.name);
                const std::string baseMaterialPathLower = ToLowerCopy(ReadStringField(jm, "base_material_asset_path", std::string{}));
                const std::string emissiveTexNameLower = ToLowerCopy(textureNameById.count(emissiveTexId) ? textureNameById[emissiveTexId] : std::string{});
                const std::string emissiveTexPathLower = ToLowerCopy(texturePathById.count(emissiveTexId) ? texturePathById[emissiveTexId] : std::string{});
                const int materialDomain = ReadIntField(jm, "material_domain", 0);

                const bool materialLooksExplicitEmissive = IsSpecialVisibleEmissiveSurface(materialNameLower, baseMaterialPathLower);
                const bool materialLooksHousing          = IsLikelyLightHousingMaterialName(materialNameLower) && !materialLooksExplicitEmissive;
                const bool materialLooksNonEmissive      = IsStrongNonEmissiveMaterialName(materialNameLower);
                const bool texLooksUtility               = IsUtilityTexturePathOrName(emissiveTexNameLower) || IsUtilityTexturePathOrName(emissiveTexPathLower);
                const bool texLooksEmissive              = IsStrongEmissiveTexturePathOrName(emissiveTexNameLower) || IsStrongEmissiveTexturePathOrName(emissiveTexPathLower);
                const bool isDecalMaterial               = (materialDomain == 1);

                if (texLooksUtility)
                    m.emissionTexIndex = -1;

                bool allowEmission = false;
                if (!isDecalMaterial && !materialLooksNonEmissive && !materialLooksHousing)
                {
                    if (explicitEmissiveStrength > 0.0f)
                    {
                        allowEmission = true;
                    }
                    else if (emissionColorNonZero && materialLooksExplicitEmissive)
                    {
                        allowEmission = true;
                    }
                    else if (m.emissionTexIndex >= 0 && (texLooksEmissive || materialLooksExplicitEmissive))
                    {
                        allowEmission = true;
                    }
                }

                if (allowEmission)
                {
                    if (explicitEmissiveStrength >= 0.0f)
                    {
                        m.emissionStrength = explicitEmissiveStrength;
                    }
                    else if (emissionColorNonZero)
                    {
                        m.emissionStrength = 1000.0f;
                    }
                    else if (m.emissionTexIndex >= 0)
                    {
                        m.emissionColor = Vec3{1.0f, 1.0f, 1.0f};
                        m.emissionStrength = 1000.0f;
                    }
                }
                else
                {
                    m.emissionTexIndex = -1;
                    m.emissionColor = Vec3{0.0f, 0.0f, 0.0f};
                    m.emissionStrength = 0.0f;
                }

                parsedMaterials.push_back(m);

                const int32_t idx = (int32_t)parsedMaterials.size() - 1;
                const std::string stableId = ReadStringField(jm, "stable_id", std::string{});
                if (!stableId.empty())
                    materialIndexById[stableId] = idx;
                materialIndexByName[m.name] = idx;
                materialIndexByNorm[NormalizeKey(m.name)] = idx;
            }
        }

        auto resolveMaterialIndex = [&](const std::string& key) -> int32_t
        {
            if (key.empty())
                return -1;

            auto itId = materialIndexById.find(key);
            if (itId != materialIndexById.end())
                return itId->second;

            auto itName = materialIndexByName.find(key);
            if (itName != materialIndexByName.end())
                return itName->second;

            const std::string norm = NormalizeKey(key);
            auto itNorm = materialIndexByNorm.find(norm);
            if (itNorm != materialIndexByNorm.end())
                return itNorm->second;

            return -1;
        };

        auto isFilteredMaterialIndex = [&](int32_t idx) -> bool
        {
            if (idx < 0 || (std::size_t)idx >= parsedMaterials.size())
                return false;

            const SceneMetaMaterial& m = parsedMaterials[(std::size_t)idx];
            const std::string nameLower = NormalizeKey(m.name);
            if (LooksFogLike(nameLower))
                return true;

            if (!IsUnsupportedBlendModeForCurrentTracer(m.blendMode))
                return false;

            return !IsExplicitEmissiveMaterialName(nameLower);
        };

        scene.clearLights();
        if (auto itL = j.find("lights"); itL != j.end() && itL->is_array())
        {
            for (const json& jl : *itL)
            {
                if (!jl.is_object())
                    continue;

                Light l{};

                const std::string typeStr = ReadStringField(jl, "light_type",
                                              ReadStringField(jl, "type", std::string("point")));

                if (typeStr == "point")
                    l.type = LightType::Point;
                else if (typeStr == "spot")
                    l.type = LightType::Spot;
                else if (typeStr == "directional" || typeStr == "sun")
                    l.type = LightType::Directional;
                else if (typeStr == "rect" || typeStr == "area")
                    l.type = LightType::Area;
                else
                    l.type = LightType::Point;

                const std::array<float,16> m = ReadMatrix4Field(jl, "transform_matrix");
                l.position  = MatrixPosition(m);
                l.direction = NormalizeSafe(MatrixAxisX(m), Vec3{1, 0, 0});
                l.color     = ReadVec3Field(jl, "color", Vec3{1, 1, 1});

                l.position.x *= unitScale;
                l.position.y *= unitScale;
                l.position.z *= unitScale;

                l.intensity = ReadFloatField(jl, "intensity", 1.0f);
                l.radius    = ReadFloatField(jl, "source_radius", 0.0f) * unitScale;

                const float innerDeg = ReadFloatField(jl, "inner_cone_angle", 0.0f);
                const float outerDeg = ReadFloatField(jl, "outer_cone_angle", 0.0f);
                l.spotSize  = outerDeg * kPI / 180.0f;
                l.spotBlend = 0.0f;
                if (outerDeg > 1e-6f && innerDeg >= 0.0f && innerDeg < outerDeg)
                    l.spotBlend = std::clamp(1.0f - (innerDeg / outerDeg), 0.0f, 1.0f);

                l.attenuationRadius = ReadFloatField(jl, "attenuation_radius", 0.0f) * unitScale;

                scene.addLight(l);
            }
        }

        const auto& objs = scene.getObjects();
        for (const SceneObject& objConst : objs)
        {
            SceneObject* obj = scene.getObject(objConst.getName());
            if (!obj)
                continue;

            const auto& srcTris = obj->getTriangles();
            if (srcTris.empty())
                continue;

            std::vector<Triangle> filtered;
            filtered.reserve(srcTris.size());

            for (std::size_t ti = 0; ti < srcTris.size(); ++ti)
            {
                Triangle tri = srcTris[ti];
                int32_t matIdx = tri.materialIndex;

                if (matIdx < 0)
                {
                    if (const std::string* triMatName = obj->getTriangleMaterialName(ti))
                        matIdx = resolveMaterialIndex(*triMatName);
                }

                tri.materialIndex = matIdx;

                if (matIdx >= 0 && (std::size_t)matIdx < parsedMaterials.size())
                {
                    if (isFilteredMaterialIndex(matIdx))
                        continue;

                    const SceneMetaMaterial& sm = parsedMaterials[(std::size_t)matIdx];
                    tri.color = sm.baseColor;

                    const float strength = sm.emissionStrength * 0.001f;
                    tri.emission = Vec3{
                        sm.emissionColor.x * strength,
                        sm.emissionColor.y * strength,
                        sm.emissionColor.z * strength
                    };

                    tri.metallic  = sm.metallic;
                    tri.roughness = sm.roughness;
                }

                filtered.push_back(tri);
            }

            if (LooksFogLike(NormalizeKey(obj->getName())))
                filtered.clear();

            obj->setTriangles(std::move(filtered));
        }

        std::vector<SceneMetaDecal> parsedDecals;
        if (auto itD = j.find("decals"); itD != j.end() && itD->is_array())
        {
            parsedDecals.reserve(itD->size());
            for (const json& jd : *itD)
            {
                if (!jd.is_object())
                    continue;

                SceneMetaDecal d{};
                d.name = ReadStringField(jd, "name", std::string{});

                const std::array<float,16> m = ReadMatrix4Field(jd, "transform_matrix");
                const Vec3 matrixAxisX = MatrixAxisX(m);
                const Vec3 matrixAxisY = MatrixAxisY(m);
                const Vec3 matrixAxisZ = MatrixAxisZ(m);

                d.position = ReadVec3Field(jd, "position", MatrixPosition(m));
                d.axisX = NormalizeSafe(ReadVec3Field(jd, "axis_x", matrixAxisX), Vec3{1,0,0});
                d.axisY = NormalizeSafe(ReadVec3Field(jd, "axis_y", matrixAxisY), Vec3{0,1,0});
                d.axisZ = NormalizeSafe(ReadVec3Field(jd, "axis_z", matrixAxisZ), Vec3{0,0,1});

                // Prefer an orthonormal in-plane basis derived from the projection axis and one in-plane axis.
                // This keeps the decal box stable even when the exported transform matrix carries non-uniform scale.
                d.axisY = NormalizeSafe(Cross3(d.axisX, d.axisZ), d.axisY);
                if (Length3(d.axisY) <= 1.0e-4f)
                    d.axisY = NormalizeSafe(matrixAxisY, Vec3{0,1,0});
                d.axisZ = NormalizeSafe(Cross3(d.axisY, d.axisX), d.axisZ);

                d.position.x *= unitScale; d.position.y *= unitScale; d.position.z *= unitScale;

                auto itSize = jd.find("size");
                if (itSize != jd.end() && itSize->is_array() && itSize->size() >= 3)
                    d.size = ReadVec3(*itSize, Vec3{0,0,0});
                else if ((itSize = jd.find("decal_size")) != jd.end() && itSize->is_array() && itSize->size() >= 3)
                    d.size = ReadVec3(*itSize, Vec3{0,0,0});
                else
                    d.size = ReadVec3Field(jd, "size", Vec3{0,0,0});

                bool sizeIncludesComponentScale = false;
                if (auto itScaled = jd.find("size_includes_component_scale"); itScaled != jd.end() && itScaled->is_boolean())
                    sizeIncludesComponentScale = itScaled->get<bool>();

                if (!sizeIncludesComponentScale)
                {
                    // UE documents DecalSize as local-space extents that do not include component scale.
                    // Older scene.json exports wrote raw DecalSize into `size`, while the transform matrix already
                    // contained the component scale. Compensate here by folding the matrix scale into the extents.
                    const float sx = Length3(matrixAxisX);
                    const float sy = Length3(matrixAxisY);
                    const float sz = Length3(matrixAxisZ);
                    if (sx > 1.0e-6f) d.size.x *= sx;
                    if (sy > 1.0e-6f) d.size.y *= sy;
                    if (sz > 1.0e-6f) d.size.z *= sz;
                }

                d.size.x *= unitScale; d.size.y *= unitScale; d.size.z *= unitScale;
                d.sortOrder = ReadIntField(jd, "sort_order", 0);
                d.fadeScreenSize = ReadFloatField(jd, "fade_screen_size", 0.0f);

                d.materialIndex = resolveMaterialIndex(ReadStringField(jd, "material_id", std::string{}));
                if (d.materialIndex < 0)
                    d.materialIndex = resolveMaterialIndex(ReadStringField(jd, "material", std::string{}));
                if (d.materialIndex < 0)
                    d.materialIndex = resolveMaterialIndex(ReadStringField(jd, "material_name", std::string{}));

                if (d.materialIndex >= 0)
                    parsedDecals.push_back(d);
            }

            std::stable_sort(parsedDecals.begin(), parsedDecals.end(),
                [](const SceneMetaDecal& a, const SceneMetaDecal& b)
                {
                    return a.sortOrder < b.sortOrder;
                });
        }

        if (outRes)
        {
            outRes->baseColorTextures = std::move(baseColorTextures);
            outRes->linearTextures    = std::move(linearTextures);
            outRes->materials         = std::move(parsedMaterials);
            outRes->materialsPBR      = outRes->materials;
            outRes->cameras           = std::move(parsedCameras);
            outRes->decals            = std::move(parsedDecals);
            outRes->hasPostProcess    = hasPostProcess;
            outRes->postProcess       = postProcess;
            outRes->hasFog            = hasFog;
            outRes->fog               = fog;
        }

        return true;
    }


    // Debug unresolved names (set to 1 if needed)
    // #define SCENE_META_DEBUG_UNRESOLVED 1
}

bool LoadCamerasFromMeta(const std::string& metaPath,
                         std::vector<SceneMetaCameraInfo>& outCameras)
{
    outCameras.clear();

    json j;
    {
        std::ifstream file(metaPath);
        if (!file)
        {
            std::cerr << "SceneMetaLoader: cannot open meta file: " << metaPath << "\n";
            return false;
        }
        try { file >> j; }
        catch (const std::exception& e)
        {
            std::cerr << "SceneMetaLoader: JSON parse error: " << e.what() << "\n";
            return false;
        }
    }

    if (IsSceneExportFormat(j))
        return LoadCamerasFromSceneExportJson(j, outCameras);

    float unitScale = 1.0f;
    if (auto itu = j.find("units"); itu != j.end() && itu->is_object())
    {
        unitScale = (float)itu->value("scale", 1.0);
        if (unitScale <= 0.0f) unitScale = 1.0f;
    }

    auto parseCam = [&](const json& jc) -> SceneMetaCameraInfo
    {
        return ParseCameraInfoFromJson(jc, unitScale);
    };

    std::string mainCamName;
    if (auto itc = j.find("camera"); itc != j.end() && itc->is_object())
        mainCamName = itc->value("name", std::string{});

    if (auto it = j.find("cameras"); it != j.end() && it->is_array())
    {
        outCameras.reserve(it->size());
        for (const json& jc : *it)
            outCameras.push_back(parseCam(jc));

        if (!mainCamName.empty())
        {
            auto it2 = std::find_if(outCameras.begin(), outCameras.end(),
                                    [&](const SceneMetaCameraInfo& c){ return c.name == mainCamName; });
            if (it2 != outCameras.end() && it2 != outCameras.begin())
                std::rotate(outCameras.begin(), it2, it2 + 1);
        }
    }
    else if (auto it2 = j.find("camera"); it2 != j.end() && it2->is_object())
    {
        outCameras.push_back(parseCam(*it2));
    }

    if (outCameras.empty())
    {
        std::cerr << "SceneMetaLoader: no cameras found in meta: " << metaPath << "\n";
        return false;
    }

    return true;
}

bool ApplyMetaCameraToCamera(const SceneMetaCameraInfo& metaCam,
                             Camera& camera,
                             int targetWidth,
                             int targetHeight)
{
    const Vec3 target{
        metaCam.position.x + metaCam.forward.x,
        metaCam.position.y + metaCam.forward.y,
        metaCam.position.z + metaCam.forward.z
    };

    camera.lookAt(metaCam.position, target, metaCam.up);

    const float fovYDegrees = metaCam.fovY * 180.0f / kPI;

    float aspect = 1.0f;
    if (targetWidth > 0 && targetHeight > 0)
        aspect = (float)targetWidth / (float)targetHeight;

    camera.setPerspective(fovYDegrees, aspect);
    camera.setViewport(targetWidth, targetHeight);
    camera.setClipPlanes(metaCam.clipStart, metaCam.clipEnd);
    camera.setFocusDistance(metaCam.focusDistance);

    return true;
}

bool LoadLightsAndMaterialsFromMeta(const std::string &metaPath,
                                    Scene &scene,
                                    SceneMetaResources *outRes)
{
    json j;
    {
        std::ifstream file(metaPath);
        if (!file)
        {
            std::cerr << "SceneMetaLoader: cannot open meta file: " << metaPath << "\n";
            return false;
        }

        try { file >> j; }
        catch (const std::exception& e)
        {
            std::cerr << "SceneMetaLoader: JSON parse error: " << e.what() << "\n";
            return false;
        }
    }

    if (IsSceneExportFormat(j))
        return LoadLightsAndMaterialsFromSceneExportJson(j, metaPath, scene, outRes);

    const std::filesystem::path metaDir = std::filesystem::path(metaPath).parent_path();

    float unitScale = 1.0f;
    if (auto itu = j.find("units"); itu != j.end() && itu->is_object())
    {
        unitScale = (float)itu->value("scale", 1.0);
        if (unitScale <= 0.0f) unitScale = 1.0f;
    }


    // --- global post_process / fog (optional) ---
    bool hasPostProcess = false;
    SceneMetaPostProcess postProcess{};
    if (auto itPP = j.find("post_process"); itPP != j.end() && itPP->is_object())
        hasPostProcess = ParsePostProcessFromJson(*itPP, postProcess);

    bool hasFog = false;
    SceneMetaFog fog{};
    if (auto itFog = j.find("fog"); itFog != j.end() && itFog->is_object())
        hasFog = ParseFogFromJson(*itFog, fog);

    std::vector<SceneMetaCameraInfo> parsedCameras;
    std::string mainCamName;
    if (auto itc = j.find("camera"); itc != j.end() && itc->is_object())
        mainCamName = itc->value("name", std::string{});
    if (auto it = j.find("cameras"); it != j.end() && it->is_array())
    {
        parsedCameras.reserve(it->size());
        for (const json& jc : *it)
            parsedCameras.push_back(ParseCameraInfoFromJson(jc, unitScale));
        if (!mainCamName.empty())
        {
            auto it2 = std::find_if(parsedCameras.begin(), parsedCameras.end(),
                                    [&](const SceneMetaCameraInfo& c){ return c.name == mainCamName; });
            if (it2 != parsedCameras.end() && it2 != parsedCameras.begin())
                std::rotate(parsedCameras.begin(), it2, it2 + 1);
        }
    }
    else if (auto it2 = j.find("camera"); it2 != j.end() && it2->is_object())
    {
        parsedCameras.push_back(ParseCameraInfoFromJson(*it2, unitScale));
    }

    // --- lights ---
    if (auto itL = j.find("lights"); itL != j.end() && itL->is_array())
    {
        scene.clearLights();
        for (const json& jl : *itL)
        {
            Light l{};

            const std::string typeStr = ReadStringField(jl, "type", std::string("point"));

            if (typeStr == "point")
                l.type = LightType::Point;
            else if (typeStr == "spot")
                l.type = LightType::Spot;
            else if (typeStr == "directional" || typeStr == "sun")
                l.type = LightType::Directional;
            else if (typeStr == "rect" || typeStr == "area")
                l.type = LightType::Area;
            else
                l.type = LightType::Point;

            l.position  = ReadVec3Field(jl, "position",  Vec3{0,0,0});
            l.direction = ReadVec3Field(jl, "direction", Vec3{0,-1,0});
            l.color     = ReadVec3Field(jl, "color",     Vec3{1,1,1});

            l.position.x *= unitScale; l.position.y *= unitScale; l.position.z *= unitScale;

            l.intensity = ReadFloatField(jl, "intensity", 1.0f);
            l.radius    = ReadFloatField(jl, "radius", 0.0f) * unitScale;

            l.spotSize  = ReadFloatField(jl, "spot_size", 0.0f);
            l.spotBlend = clamp01(ReadFloatField(jl, "spot_blend", 0.0f));

            l.attenuationRadius = ReadFloatField(jl, "attenuation_radius", 0.0f) * unitScale;

            scene.addLight(l);
        }
    }

    // --- materials ---
    std::vector<SceneMetaMaterial> parsedMaterials;
    std::unordered_map<std::string, int32_t> materialIndexByName;
    std::unordered_map<std::string, int32_t> materialIndexByNorm;

    std::vector<std::string> baseColorTextures;
    std::unordered_map<std::string, int32_t> baseColorTexIndexByPath;

    std::vector<std::string> linearTextures;
    std::unordered_map<std::string, int32_t> linearTexIndexByPath;

    if (auto itM = j.find("materials"); itM != j.end() && itM->is_array())
    {
        const json& jMats = *itM;
        parsedMaterials.reserve(jMats.size());
        materialIndexByName.reserve(jMats.size());
        materialIndexByNorm.reserve(jMats.size());

        baseColorTextures.reserve(jMats.size());
        baseColorTexIndexByPath.reserve(jMats.size());

        linearTextures.reserve(jMats.size());
        linearTexIndexByPath.reserve(jMats.size());

        for (std::size_t i = 0; i < jMats.size(); ++i)
        {
            const json& jm = jMats[i];

            SceneMetaMaterial m{};
            m.name = ReadStringField(jm, "name", std::string{});
            if (m.name.empty())
                m.name = "Material_" + std::to_string(i);

            m.baseColor = ReadVec3Field(jm, "base_color", Vec3{1,1,1});
            const Vec3 colorTint = ReadNamedVectorParam(jm, "Color Tint", Vec3{1,1,1});
            m.baseColor = Vec3{m.baseColor.x * colorTint.x, m.baseColor.y * colorTint.y, m.baseColor.z * colorTint.z};
            m.blendMode = ReadIntField(jm, "blend_mode", 0);
            if (auto itTS = jm.find("two_sided"); itTS != jm.end())
            {
                if (itTS->is_boolean())
                    m.twoSided = itTS->get<bool>();
                else if (itTS->is_number_integer())
                    m.twoSided = (itTS->get<long long>() != 0);
                else if (itTS->is_number())
                    m.twoSided = (itTS->get<double>() != 0.0);
            }

            const std::string relBaseTex = ReadStringField(jm, "base_color_texture", std::string{});
            const std::string absBaseTex = ResolveTexturePath(metaDir, relBaseTex);

            if (!absBaseTex.empty())
            {
                auto [it, inserted] = baseColorTexIndexByPath.try_emplace(absBaseTex, (int32_t)baseColorTextures.size());
                if (inserted)
                    baseColorTextures.push_back(absBaseTex);

                m.baseColorTexIndex = it->second;
            }
            else
            {
                m.baseColorTexIndex = -1;
            }

            // Optional emission texture (stored in same texture array)
            const std::string relEmTex = ReadStringField(jm, "emission_texture", std::string{});
            const std::string absEmTex = ResolveTexturePath(metaDir, relEmTex);

            if (!absEmTex.empty())
            {
                auto [itE, insertedE] = baseColorTexIndexByPath.try_emplace(absEmTex, (int32_t)baseColorTextures.size());
                if (insertedE)
                    baseColorTextures.push_back(absEmTex);

                m.emissionTexIndex = itE->second;
            }
            else
            {
                m.emissionTexIndex = -1;
            }


            // --- New PBR fields (LevelMeta13+) ---
            auto addSrgbTex = [&](const std::string& relPath) -> int32_t
            {
                const std::string abs = ResolveTexturePath(metaDir, relPath);
                if (abs.empty())
                    return -1;
                auto [itS, insertedS] = baseColorTexIndexByPath.try_emplace(abs, (int32_t)baseColorTextures.size());
                if (insertedS)
                    baseColorTextures.push_back(abs);
                return itS->second;
            };

            auto addLinearTex = [&](const std::string& relPath) -> int32_t
            {
                const std::string abs = ResolveTexturePath(metaDir, relPath);
                if (abs.empty())
                    return -1;
                auto [itL, insertedL] = linearTexIndexByPath.try_emplace(abs, (int32_t)linearTextures.size());
                if (insertedL)
                    linearTextures.push_back(abs);
                return itL->second;
            };

            m.normalTexIndex    = addLinearTex(ReadStringField(jm, "normal_texture", std::string{}));
            m.ormTexIndex       = addLinearTex(ReadStringField(jm, "orm_texture", std::string{}));
            m.roughnessTexIndex = addLinearTex(ReadStringField(jm, "roughness_texture", std::string{}));
            m.metallicTexIndex  = addLinearTex(ReadStringField(jm, "metallic_texture", std::string{}));
            m.occlusionTexIndex = addLinearTex(ReadStringField(jm, "occlusion_texture", std::string{}));

            // Decal helpers: not a full UE material graph export, but enough to reconstruct
            // a DBuffer-like projected decal with opacity/detail modulation.
            m.decalTilingU         = ReadNamedScalarParam(jm, "TilingU", 1.0f);
            m.decalTilingV         = ReadNamedScalarParam(jm, "TilingV", 1.0f);
            m.decalOpacityPower    = std::max(0.0f, ReadNamedScalarParam(jm, "Damage Opacity Power", 1.0f));
            m.decalNormalIntensity = std::max(0.0f, ReadNamedScalarParam(jm, "Normal Intenisty", 1.0f));
            m.decalRoughnessBias   = ReadNamedScalarParam(jm, "RoughMultiply", 0.0f);

            bool opacityTexSRGB = false;
            std::string opacityRel = ReadTextureParamExportedPath(jm,
                {"Opacity", "Opacity Mask", "Cone Mask", "TilingMask", "Noise Texture", "USED_002"},
                &opacityTexSRGB);
            if (!opacityRel.empty())
            {
                if (opacityTexSRGB)
                {
                    m.decalOpacityTexIndex = addSrgbTex(opacityRel);
                    m.decalOpacityTexIsLinear = false;
                }
                else
                {
                    m.decalOpacityTexIndex = addLinearTex(opacityRel);
                    m.decalOpacityTexIsLinear = true;
                }
            }

            bool detailTexSRGB = false;
            std::string detailRel = ReadTextureParamExportedPath(jm, {"USED_003"}, &detailTexSRGB);
            if (!detailRel.empty())
            {
                if (detailTexSRGB)
                {
                    m.decalDetailTexIndex = addSrgbTex(detailRel);
                    m.decalDetailTexIsLinear = false;
                }
                else
                {
                    m.decalDetailTexIndex = addLinearTex(detailRel);
                    m.decalDetailTexIsLinear = true;
                }
            }

            // packed ORM channel mapping (optional)
            if (auto itOC = jm.find("orm_channels"); itOC != jm.end() && itOC->is_object())
            {
                const json& ch = *itOC;
                m.ormChannels.occlusion = ChannelFromString(ch, "occlusion", 0);
                m.ormChannels.roughness = ChannelFromString(ch, "roughness", 1);
                m.ormChannels.metallic  = ChannelFromString(ch, "metallic", 2);
            }
            m.emissionColor    = ReadVec3Field(jm, "emission_color", Vec3{0,0,0});
            m.emissionStrength = ReadFloatField(jm, "emission_strength", 0.0f);
            m.metallic         = ReadFloatField(jm, "metallic", 0.0f);
            m.roughness        = ReadFloatField(jm, "roughness", 0.5f);
            m.opacity          = ReadNamedScalarParam(jm, "Opacity Multi", 1.0f);

            // Fix: if emissive but no explicit emission_texture — use baseColor texture as emission mask.
            if (m.emissionStrength > 0.0f && m.emissionTexIndex < 0 && m.baseColorTexIndex >= 0)
                m.emissionTexIndex = m.baseColorTexIndex;

            parsedMaterials.push_back(m);

            const int32_t idx = (int32_t)i; // stable = JSON order
            materialIndexByName[parsedMaterials.back().name] = idx;
            materialIndexByNorm[NormalizeKey(parsedMaterials.back().name)] = idx;
        }
    }

    auto resolveMaterialIndex = [&](const std::string& name) -> int32_t
    {
        if (name.empty())
            return -1;

        auto it = materialIndexByName.find(name);
        if (it != materialIndexByName.end())
            return it->second;

        const std::string key = NormalizeKey(name);
        auto it2 = materialIndexByNorm.find(key);
        if (it2 != materialIndexByNorm.end())
            return it2->second;

        return -1;
    };

    auto isFilteredMaterialIndex = [&](int32_t idx) -> bool
    {
        if (idx < 0 || (std::size_t)idx >= parsedMaterials.size())
            return false;

        const SceneMetaMaterial& m = parsedMaterials[(std::size_t)idx];
        const std::string nameLower = NormalizeKey(m.name);
        return IsUnsupportedBlendModeForCurrentTracer(m.blendMode) || LooksFogLike(nameLower);
    };

    // --- Build per-object material slots from meta (optional) ---
    struct ObjMatSlots
    {
        std::vector<int32_t> indices;
        std::vector<std::string> namesLower; // normalized
        int32_t fallback = -1;
        bool allFiltered = false;
    };

    std::unordered_map<std::string, ObjMatSlots> metaObjSlots; // key = normalized object name
    if (auto itO = j.find("objects"); itO != j.end() && itO->is_array())
    {
        metaObjSlots.reserve(itO->size());
        for (const json& jo : *itO)
        {
            const std::string objName = ReadStringField(jo, "name", std::string{});
            if (objName.empty())
                continue;

            ObjMatSlots slots;

            if (auto itOM = jo.find("materials"); itOM != jo.end() && itOM->is_array())
            {
                slots.indices.reserve(itOM->size());
                slots.namesLower.reserve(itOM->size());

                for (const json& jm : *itOM)
                {
                    const std::string matName = ReadStringField(jm, "material", std::string{});
                    const int32_t idx = resolveMaterialIndex(matName);
                    if (idx >= 0)
                    {
                        slots.indices.push_back(idx);
                        slots.namesLower.push_back(NormalizeKey(matName));
                    }
                }
            }

            // Choose fallback
            if (slots.indices.size() == 1)
            {
                slots.fallback = slots.indices[0];
            }
            else if (!slots.indices.empty())
            {
                // Prefer non-emissive as default
                for (std::size_t i = 0; i < slots.indices.size(); ++i)
                {
                    if (i < slots.namesLower.size() && !LooksEmissive(slots.namesLower[i]))
                    {
                        slots.fallback = slots.indices[i];
                        break;
                    }
                }
                if (slots.fallback < 0)
                    slots.fallback = slots.indices[0];
            }

            const std::string objNameLower = NormalizeKey(objName);
            if (!slots.indices.empty())
            {
                slots.allFiltered = true;
                for (int32_t idx : slots.indices)
                {
                    if (!isFilteredMaterialIndex(idx))
                    {
                        slots.allFiltered = false;
                        break;
                    }
                }
            }
            else
            {
                slots.allFiltered = LooksFogLike(objNameLower);
            }

            metaObjSlots.emplace(objNameLower, std::move(slots));
        }
    }

    // --- Apply materials to ALL scene objects (important: some meshes may be absent in meta objects[]) ---
    const auto& objs = scene.getObjects();
    for (const SceneObject& objConst : objs)
    {
        SceneObject* obj = scene.getObject(objConst.getName());
        if (!obj)
            continue;

        int32_t fallbackMat = -1;

        // Try exact name (normalized) lookup in metaObjSlots
        const std::string objKey = NormalizeKey(obj->getName());
        auto itSlot = metaObjSlots.find(objKey);
        if (itSlot != metaObjSlots.end())
            fallbackMat = itSlot->second.fallback;

        auto& tris = MutableTriangles(obj);

        // Cull unsupported translucent/additive helper geometry (fog sheets, god rays, etc.)
        // before BVH build. Geometry export should stay lossless; the loader decides what the
        // current opaque path tracer can safely keep.
        if ((itSlot != metaObjSlots.end() && itSlot->second.allFiltered) || LooksFogLike(objKey))
        {
            tris.clear();
            continue;
        }

#if defined(SCENE_META_DEBUG_UNRESOLVED)
        int unresolved = 0;
        std::unordered_map<std::string, int> unresolvedNames;
        unresolvedNames.reserve(64);
#endif

        for (std::size_t ti = 0; ti < tris.size(); ++ti)
        {
            int32_t matIdx = -1;

            if (const std::string* triMatName = obj->getTriangleMaterialName(ti))
            {
                if (!triMatName->empty())
                    matIdx = resolveMaterialIndex(*triMatName);
            }

            if (matIdx < 0)
                matIdx = fallbackMat;

#if defined(SCENE_META_DEBUG_UNRESOLVED)
            if (matIdx < 0)
            {
                ++unresolved;
                if (const std::string* triMatName = obj->getTriangleMaterialName(ti))
                    if (!triMatName->empty()) unresolvedNames[*triMatName]++;
            }
#endif

            tris[ti].materialIndex = matIdx;

            if (matIdx >= 0 && (std::size_t)matIdx < parsedMaterials.size())
            {
                const SceneMetaMaterial& sm = parsedMaterials[(std::size_t)matIdx];

                // Base params (used if texture missing)
                tris[ti].color = sm.baseColor;

                // Emission
                float strength = sm.emissionStrength * 0.001f;
                tris[ti].emission = Vec3{
                    sm.emissionColor.x * strength,
                    sm.emissionColor.y * strength,
                    sm.emissionColor.z * strength
                };

                tris[ti].metallic  = sm.metallic;
                tris[ti].roughness = sm.roughness;
            }
        }

#if defined(SCENE_META_DEBUG_UNRESOLVED)
        if (unresolved > 0)
        {
            std::cerr << "Meta: unresolved materials for object [" << obj->getName() << "] "
                      << unresolved << "/" << tris.size() << "\n";

            int printed = 0;
            for (auto &kv : unresolvedNames)
            {
                std::cerr << "  triMatName: " << kv.first << "  count=" << kv.second << "\n";
                if (++printed >= 8)
                    break;
            }
        }
#endif
    }


    // --- decals ---
    std::vector<SceneMetaDecal> parsedDecals;
    if (auto itD = j.find("decals"); itD != j.end() && itD->is_array())
    {
        parsedDecals.reserve(itD->size());
        for (const json& jd : *itD)
        {
            SceneMetaDecal d{};
            d.name = ReadStringField(jd, "name", std::string{});
            d.position = ReadVec3Field(jd, "position", Vec3{0,0,0});
            d.position.x *= unitScale; d.position.y *= unitScale; d.position.z *= unitScale;
            d.axisX = ReadVec3Field(jd, "axis_x", Vec3{1,0,0});
            d.axisY = ReadVec3Field(jd, "axis_y", Vec3{0,1,0});
            d.axisZ = ReadVec3Field(jd, "axis_z", Vec3{0,0,1});
            d.size = ReadVec3Field(jd, "size", Vec3{0,0,0});
            d.size.x *= unitScale; d.size.y *= unitScale; d.size.z *= unitScale;
            d.sortOrder = ReadIntField(jd, "sort_order", 0);
            d.fadeScreenSize = ReadFloatField(jd, "fade_screen_size", 0.0f);

            const std::string decalMatName = ReadStringField(jd, "material", std::string{});
            d.materialIndex = resolveMaterialIndex(decalMatName);

            if (d.materialIndex >= 0)
                parsedDecals.push_back(d);
        }

        std::stable_sort(parsedDecals.begin(), parsedDecals.end(), [](const SceneMetaDecal& a, const SceneMetaDecal& b)
        {
            return a.sortOrder < b.sortOrder;
        });
    }

    if (outRes)
    {
        outRes->baseColorTextures = std::move(baseColorTextures);
        outRes->linearTextures    = std::move(linearTextures);
        outRes->materials         = std::move(parsedMaterials);
        outRes->materialsPBR      = outRes->materials; // keep naming stable for backends
        outRes->cameras           = std::move(parsedCameras);
        outRes->decals            = std::move(parsedDecals);
        outRes->hasPostProcess    = hasPostProcess;
        outRes->postProcess       = postProcess;
        outRes->hasFog            = hasFog;
        outRes->fog               = fog;
    }

    return true;
}
