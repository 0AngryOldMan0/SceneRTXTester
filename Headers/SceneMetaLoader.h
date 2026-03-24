#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "Point3D.h"   // Vec3
#include "Point4D.h"   // Vec4

// Forward declarations to keep this header lightweight.
class Scene;
class Camera;

// Enable PBR texture bridge in MetalRenderer.
#ifndef SCENE_META_HAS_PBR_TEXTURES
#define SCENE_META_HAS_PBR_TEXTURES 1
#endif

// Packed ORM channel mapping: 0=R,1=G,2=B,3=A
struct SceneMetaOrmChannels
{
    uint8_t occlusion = 0; // default UE: R
    uint8_t roughness = 1; // default UE: G
    uint8_t metallic  = 2; // default UE: B
};

struct SceneMetaPostProcess
{
    // Exposure
    int   autoExposureMethod = 0;
    float autoExposureBias = 0.0f;
    float autoExposureMin = 1.0f;
    float autoExposureMax = 1.0f;

    // Bloom & lens
    float bloomIntensity = 0.0f;
    float bloomThreshold = 0.0f;
    float vignetteIntensity = 0.0f;
    float chromaticAberration = 0.0f;

    // Film
    float filmGrainIntensity = 0.0f;
    float filmSlope = 0.88f;
    float filmToe = 0.55f;
    float filmShoulder = 0.26f;
    float filmBlackClip = 0.0f;
    float filmWhiteClip = 0.04f;

    // Optional UE color grading approximation (from color_saturation array)
    Vec3  colorSaturation{1.0f, 1.0f, 1.0f};
};

struct SceneMetaFog
{
    float fogDensity = 0.0f;
    float heightFalloff = 0.0f;
    Vec3  inscatteringColor{1.0f, 1.0f, 1.0f};
    float maxOpacity = 1.0f;
    float startDistance = 0.0f;
    float heightReferenceZ = 0.0f;
    bool  hasHeightReference = false;

    std::string actorPath;
    std::string componentPath;

    bool  volumetricFog = false;
    float scatteringG = 0.0f;          // HG g, from volumetric_scattering_distribution
    Vec3  volumetricAlbedo{1.0f, 1.0f, 1.0f};
    float extinctionScale = 1.0f;
};

// Материал из LevelMeta JSON.
enum class SceneMetaMasterMaterialModel : std::uint32_t
{
    GenericPBR = 0,
    MM_Material_01a,
    MM_Decal_01a,
    MM_Tunnel_Floor_01a,
    MM_Tunnel_Wall_01a,
    MM_Emissive_Headlights_01a,
    MM_Concrete_Edging_01a,
    MM_Mesh_Decal_01a,
    MM_Webs_01a,
    MM_Vent_Offset_01a,
    MM_Black_01a
};

struct SceneMetaAuxTextureEntry
{
    std::string name;
    std::string sourceRef;
    std::string textureName;
    std::string textureAssetPath;
    std::string resolvedPath;
    int32_t textureIndex = -1;
    bool isLinear = false;
};

struct SceneMetaAuxScalarEntry
{
    std::string name;
    float value = 0.0f;
};

struct SceneMetaAuxVec4Entry
{
    std::string name;
    Vec4 value{0.0f, 0.0f, 0.0f, 0.0f};
};

struct SceneMetaTunnelFloorMaterialParams
{
    float roughnessValue = 1.0f;
    float roughnessMulti = 1.0f;
    float normalFlatness = 0.0f;
    float aoRoughnessMulti = 1.0f;
    float puddlesVertexColorMulti = 1.0f;
    float puddlesBlendSharpness = 0.5f;
    float puddlesMaskPower = 1.0f;
    float puddlesMaskMultiply = 1.0f;
    float dirtBlendSharpness = 0.5f;
    float dirtVertexColorMulti = 1.0f;
    float dirtUvScale = 1.0f;
    int32_t puddlesMixMapUvSet = 0;
    int32_t dirtMixMapUvSet = 0;
    int32_t dirtMaskTexIndex = -1;
    int32_t puddlesMaskTexIndex = -1;
    int32_t dirtAlbedoTexIndex = -1;
    int32_t concreteFillAlbedoTexIndex = -1;
    int32_t concreteFillNormalTexIndex = -1;
    int32_t dirtNormalTexIndex = -1;
};

struct SceneMetaTunnelSurfaceMaterialParams
{
    float roughness = 0.5f;
    float roughnessMulti = 1.0f;
    float roughnessPower = 1.0f;
    float metalnessValue = 0.0f;
    float dirtRoughness = 0.9f;
};

struct SceneMetaMaterial
{
    std::string name;
    std::string baseMaterialAssetPath;
    SceneMetaMasterMaterialModel masterMaterialModel = SceneMetaMasterMaterialModel::GenericPBR;
    std::vector<SceneMetaAuxTextureEntry> auxTex;
    std::vector<SceneMetaAuxScalarEntry> auxScalar;
    std::vector<SceneMetaAuxVec4Entry> auxVec4;

    Vec3  baseColor{1.0f, 1.0f, 1.0f};
    Vec3  emissionColor{0.0f, 0.0f, 0.0f};
    float emissionStrength = 0.0f;
    float metallic         = 0.0f;
    float roughness        = 0.5f;
    float opacity          = 1.0f;  // decal/translucency helper from scalar param "Opacity Multi"
    int32_t blendMode      = 0;     // UE EBlendMode, 0=Opaque, 1=Masked, 2+=translucent-like
    bool    twoSided       = false;
    bool    thinEmissiveSurface = false;
    bool    emissionUseAlphaMask = false;
    int32_t specialModel = 0;
    int32_t specialTex0Index = -1;
    int32_t specialTex1Index = -1;
    float   specialScalar0 = 0.0f;
    float   specialScalar1 = 0.0f;
    float   specialScalar2 = 0.0f;
    float   specialScalar3 = 0.0f;
    float   specialScalar4 = 0.0f;
    float   specialScalar5 = 0.0f;

    // Decal-oriented material approximation (DBuffer-style subset).
    // These values are CPU-side only; MetalRenderer compacts them into DecalGPU.
    float decalTilingU         = 1.0f;
    float decalTilingV         = 1.0f;
    float decalOpacityPower    = 1.0f;
    float decalNormalIntensity = 1.0f;
    float decalRoughnessBias   = 0.0f;
    int32_t decalOpacityTexIndex = -1; // index into baseColorTextures or linearTextures depending on flag below
    bool    decalOpacityTexIsLinear = false;
    int32_t decalDetailTexIndex = -1;  // optional auxiliary detail/mask texture
    bool    decalDetailTexIsLinear = false;

    // Indices into SceneMetaResources::baseColorTextures (sRGB), -1 if absent
    int32_t baseColorTexIndex = -1;
    int32_t emissionTexIndex  = -1;
    int32_t baseColorUvSet    = 0;
    int32_t emissionUvSet     = 0;

    // Indices into SceneMetaResources::linearTextures (linear UNORM), -1 if absent
    int32_t normalTexIndex    = -1;
    int32_t ormTexIndex       = -1;
    int32_t roughnessTexIndex = -1;
    int32_t metallicTexIndex  = -1;
    int32_t occlusionTexIndex = -1;
    int32_t normalUvSet       = 0;
    int32_t ormUvSet          = 0;
    int32_t roughnessUvSet    = 0;
    int32_t metallicUvSet     = 0;
    int32_t occlusionUvSet    = 0;

    SceneMetaTunnelFloorMaterialParams tunnelFloorParams{};
    SceneMetaTunnelSurfaceMaterialParams tunnelSurfaceParams{};
    SceneMetaOrmChannels ormChannels{};
};

struct SceneMetaCameraInfo
{
    std::string name;

    Vec3 position{0, 0, 0};
    Vec3 forward{0, 0, -1};
    Vec3 up{0, 1, 0};
    Vec3 right{1, 0, 0};

    float fovY = 1.0471975512f; // ~60° in radians
    float aspectRatio = 16.0f / 9.0f;
    float clipStart = 0.1f;
    float clipEnd = 10000.0f;
    float focusDistance = 10.0f;
    bool constrainAspectRatio = false;

    bool hasPostProcess = false;
    SceneMetaPostProcess postProcess{};

    bool hasFog = false;
    SceneMetaFog fog{};
};

struct SceneMetaDecal
{
    std::string name;
    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 axisX{1.0f, 0.0f, 0.0f};
    Vec3 axisY{0.0f, 1.0f, 0.0f};
    Vec3 axisZ{0.0f, 0.0f, 1.0f};
    Vec3 size{0.0f, 0.0f, 0.0f};
    int32_t materialIndex = -1;
    int32_t sortOrder = 0;
    float fadeScreenSize = 0.0f;
};

struct SceneMetaAirDustVolume
{
    std::string name;
    std::string actorPath;
    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 extent{0.0f, 0.0f, 0.0f};
    Vec3 linkedLightPosition{0.0f, 0.0f, 0.0f};
    Vec3 linkedLightColor{1.0f, 1.0f, 1.0f};
    float linkedLightIntensity = 0.0f;
    float linkedLightRadius = 0.0f;
};

struct SceneMetaResources
{
    // Unique absolute paths to sRGB textures (base color + emission), used in Metal texture array.
    std::vector<std::string> baseColorTextures;

    // Unique absolute paths to linear UNORM textures (normal/orm/rough/metal/ao).
    std::vector<std::string> linearTextures;

    // Materials in the same order as materials[] in JSON (this is materialIndex).
    std::vector<SceneMetaMaterial> materials;

    // Convenience alias for render backends: PBR-capable material list (same order as materials).
    // Currently identical to 'materials' (copied), but kept as a separate field to keep naming stable.
    std::vector<SceneMetaMaterial> materialsPBR;

    // Cameras from meta (used by runtime to match current camera and pick per-camera post process).
    std::vector<SceneMetaCameraInfo> cameras;

    // Projected decals from meta (UE DecalActor / DecalComponent).
    std::vector<SceneMetaDecal> decals;

    // Localized dust / haze volumes reconstructed from exported particle helpers like P_AirDust_01.
    std::vector<SceneMetaAirDustVolume> airDustVolumes;

    // Conversion from scene world units to meters for physically-based effects like volumetrics.
    float worldUnitToMeters = 1.0f;

    // Optional global settings
    bool hasPostProcess = false;
    SceneMetaPostProcess postProcess{};

    bool hasFog = false;
    SceneMetaFog fog{};
};

bool LoadCamerasFromMeta(const std::string &metaPath,
                         std::vector<SceneMetaCameraInfo> &outCameras);

// apply meta-camera to Camera (lookAt/setPerspective/...)
bool ApplyMetaCameraToCamera(const SceneMetaCameraInfo &metaCam,
                             Camera &camera,
                             int targetWidth,
                             int targetHeight);

// load lights + materials + textures + set triangle.materialIndex
bool LoadLightsAndMaterialsFromMeta(const std::string &metaPath,
                                    Scene &scene,
                                    SceneMetaResources *outRes);
