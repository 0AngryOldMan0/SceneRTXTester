#pragma once

#include <cstddef>
#include <cstdint>

#include "Structures/AABB.h"
#include "Structures/BVHNode.h"
#include "Structures/CameraData.h"
#include "Structures/Triangles.h"

constexpr std::uint32_t HIP_MAX_SCENE_TEXTURES = 124u;
constexpr std::uint32_t HIP_TEXTURE_FLAG_SRGB = 1u;

constexpr int HIP_MATERIAL_FLAG_EMISSION_USE_ALPHA_MASK = 1;
constexpr int HIP_MATERIAL_FLAG_THIN_EMISSIVE_SURFACE = 2;
constexpr int HIP_SPECIAL_MATERIAL_UE_HEADLIGHT = 1;
constexpr int HIP_SPECIAL_MATERIAL_UE_TRAFFIC_LIGHT = 2;
constexpr int HIP_MASTER_MATERIAL_GENERIC_PBR = 0;
constexpr int HIP_MASTER_MATERIAL_MM_MATERIAL_01A = 1;
constexpr int HIP_MASTER_MATERIAL_MM_TUNNEL_FLOOR_01A = 3;
constexpr int HIP_MASTER_MATERIAL_MM_TUNNEL_WALL_01A = 4;
constexpr std::uint32_t HIP_INSTANCE_FLAG_CASTS_SHADOW = 1u;
constexpr std::uint32_t HIP_LIGHT_FLAG_CASTS_SHADOW = 1u;

enum class HIPDebugView : std::uint32_t
{
    Disabled = 0u,
    Ns = 1u,
    AO = 2u,
    NsMinusNg = 3u,
    BaseColor = 4u,
    Roughness = 5u,
    Metallic = 6u,
    Emissive = 7u,
    VertexColor = 8u,
    MaterialModel = 9u
};

struct HIPSceneInstanceGPU
{
    float objectToWorld[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    float worldToObject[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    float normalToWorld[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    AABB worldBounds{};
    int blasRootIndex = -1;
    std::uint32_t flags = 1u;
    std::uint32_t ownerId = 0u;
    int _pad2 = 0;
};

struct HIPLightGPU
{
    int type = 0;
    std::uint32_t flags = HIP_LIGHT_FLAG_CASTS_SHADOW;

    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 direction{0.0f, -1.0f, 0.0f};
    Vec3 color{1.0f, 1.0f, 1.0f};

    float intensity = 1.0f;
    float radius = 0.0f;
    float sourceLength = 0.0f;
    float softSourceRadius = 0.0f;
    float spotSize = 0.0f;
    float spotBlend = 0.0f;

    float attenuationRadius = 0.0f;
    std::uint32_t ownerId = 0u;
};

struct HIPMaterialGPU
{
    std::int32_t baseColorTexIndex = -1;
    std::int32_t emissionTexIndex = -1;
    std::int32_t baseColorUvSet = 0;
    std::int32_t emissionUvSet = 0;
};

struct HIPTunnelFloorParamsGPU
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
    std::int32_t puddlesMixMapUvSet = 0;
    std::int32_t dirtMixMapUvSet = 0;
    std::int32_t dirtMaskTexIndex = -1;
    std::int32_t puddlesMaskTexIndex = -1;
    std::int32_t dirtAlbedoTexIndex = -1;
    std::int32_t concreteFillAlbedoTexIndex = -1;
    std::int32_t concreteFillNormalTexIndex = -1;
    std::int32_t dirtNormalTexIndex = -1;
};

struct HIPTunnelSurfaceParamsGPU
{
    float roughness = 0.5f;
    float roughnessMulti = 1.0f;
    float roughnessPower = 1.0f;
    float metalnessValue = 0.0f;
    float dirtRoughness = 0.9f;
    float primaryUvScale = 1.0f;
    float normalUvScale = 1.0f;
    float damageUvScale = 1.0f;
    float detailUvScale = 1.0f;
    float baseNormalIntensity = 1.0f;
    float damagedNormalIntensity = 1.0f;
    float fillNormalIntensity = 1.0f;
    float damageBlendSharpness = 0.5f;
    float damageMaskMultiply = 1.0f;
    float damageMaskPower = 1.0f;
    float vertexBlueMulti = 1.0f;
    float vertexRedMulti = 1.0f;
    float redMaskUvScale = 1.0f;
    float damagedRoughnessMulti = 1.0f;
    float fillBlendRatio = 0.0f;
    float fillBlendPower = 1.0f;
    std::int32_t blendMaskTexIndex = -1;
    std::int32_t damagedAlbedoTexIndex = -1;
    std::int32_t damagedNormalTexIndex = -1;
    std::int32_t damagedOrmTexIndex = -1;
    std::int32_t fillAlbedoTexIndex = -1;
    std::int32_t fillNormalTexIndex = -1;
};

struct HIPMaterialPBRGPU
{
    std::int32_t baseColorTexIndex = -1;
    std::int32_t emissionTexIndex = -1;
    std::int32_t normalTexIndex = -1;
    std::int32_t ormTexIndex = -1;
    std::int32_t roughnessTexIndex = -1;
    std::int32_t metallicTexIndex = -1;
    std::int32_t occlusionTexIndex = -1;
    std::int32_t flags = 0;

    std::int32_t baseColorUvSet = 0;
    std::int32_t emissionUvSet = 0;
    std::int32_t normalUvSet = 0;
    std::int32_t ormUvSet = 0;
    std::int32_t roughnessUvSet = 0;
    std::int32_t metallicUvSet = 0;
    std::int32_t occlusionUvSet = 0;
    std::int32_t specialModel = 0;

    std::int32_t specialTex0Index = -1;
    std::int32_t specialTex1Index = -1;
    std::uint8_t ormChannelOcclusion = 0u;
    std::uint8_t ormChannelRoughness = 1u;
    std::uint8_t ormChannelMetallic = 2u;
    std::uint8_t _pad0 = 0u;
    std::int32_t _pad1 = 0;

    float specialScalar0 = 0.0f;
    float specialScalar1 = 0.0f;
    float specialScalar2 = 0.0f;
    float specialScalar3 = 0.0f;
    float specialScalar4 = 0.0f;
    float specialScalar5 = 0.0f;

    std::int32_t masterMaterialModel = HIP_MASTER_MATERIAL_GENERIC_PBR;
    std::int32_t _pad2 = 0;
    HIPTunnelFloorParamsGPU tunnelFloor{};
    HIPTunnelSurfaceParamsGPU tunnelSurface{};
};

struct HIPDecalGPU
{
    float posX = 0.0f;
    float posY = 0.0f;
    float posZ = 0.0f;
    float sizeX = 0.0f;

    float axisXx = 1.0f;
    float axisXy = 0.0f;
    float axisXz = 0.0f;
    float sizeY = 0.0f;

    float axisYx = 0.0f;
    float axisYy = 1.0f;
    float axisYz = 0.0f;
    float sizeZ = 0.0f;

    float axisZx = 0.0f;
    float axisZy = 0.0f;
    float axisZz = 1.0f;
    float opacity = 0.0f;

    std::int32_t baseColorTexIndex = -1;
    std::int32_t ormTexIndex = -1;
    std::int32_t roughnessTexIndex = -1;
    std::int32_t normalTexIndex = -1;

    std::int32_t opacityTexIndex = -1;
    std::int32_t detailTexIndex = -1;
    std::int32_t flags = 0;
    std::int32_t _pad0 = 0;

    float baseColorX = 1.0f;
    float baseColorY = 1.0f;
    float baseColorZ = 1.0f;
    float roughnessBias = 0.0f;

    float tilingU = 1.0f;
    float tilingV = 1.0f;
    float opacityPower = 1.0f;
    float normalIntensity = 1.0f;
};

struct HIPEmissiveTriangleGPU
{
    std::uint32_t triIndex = 0u;
    std::uint32_t instanceIndex = 0u;
    float area = 0.0f;
    float selectionPdf = 0.0f;
    float cdf = 0.0f;
};

struct HIPAirDustVolumeGPU
{
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float density = 0.0f;

    float extentX = 0.0f;
    float extentY = 0.0f;
    float extentZ = 0.0f;
    float anisotropy = 0.0f;

    float lightPosX = 0.0f;
    float lightPosY = 0.0f;
    float lightPosZ = 0.0f;
    float lightIntensity = 0.0f;

    float lightColorX = 1.0f;
    float lightColorY = 1.0f;
    float lightColorZ = 1.0f;
    float lightRadius = 0.0f;
};

struct HIPPostProcessParams
{
    float exposure = 1.0f;
    float bloomIntensity = 0.0f;
    float bloomThreshold = 0.0f;
    float vignetteIntensity = 0.0f;

    float chromaticAberration = 0.0f;
    float filmGrainIntensity = 0.0f;
    float filmSlope = 0.88f;
    float filmToe = 0.55f;

    float filmShoulder = 0.26f;
    float filmBlackClip = 0.0f;
    float filmWhiteClip = 0.04f;
    float fogDensity = 0.0f;

    float fogHeightFalloff = 0.0f;
    float fogScatteringG = 0.0f;
    float fogColorX = 1.0f;
    float fogColorY = 1.0f;

    float fogColorZ = 1.0f;
    float fogExtinctionScale = 1.0f;
    float fogAlbedoX = 1.0f;
    float fogAlbedoY = 1.0f;

    float fogAlbedoZ = 1.0f;
    float volumetricFog = 0.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;

    float time = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float _pad0 = 0.0f;

    float colorSaturationX = 1.0f;
    float colorSaturationY = 1.0f;
    float colorSaturationZ = 1.0f;
    float shadowLift = 0.0f;

    float fogStartDistance = 0.0f;
    float fogMaxOpacity = 1.0f;
    float fogHeightZ = 0.0f;
    float worldUnitToMeters = 1.0f;

    std::uint32_t debugView = 0u;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
    float _pad3 = 0.0f;
};

struct HIPTextureDescGPU
{
    std::uint64_t texelOffset = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint32_t flags = 0u;
};

struct HIPTextureExecutionProfile
{
    double ensureOutputMs = 0.0;
    double uploadSceneMs = 0.0;
    double kernelMs = 0.0;
    double readbackMs = 0.0;
    double accumulationMs = 0.0;
    double postProcessMs = 0.0;
    double totalMs = 0.0;
};

static_assert(sizeof(HIPSceneInstanceGPU) == 184, "HIPSceneInstanceGPU size must match SceneInstanceGPU");
static_assert(sizeof(HIPLightGPU) == 76, "HIPLightGPU size must be 76 bytes");
static_assert(sizeof(HIPMaterialGPU) == 16, "HIPMaterialGPU size must be 16 bytes");
static_assert(sizeof(HIPMaterialPBRGPU) == 296, "HIPMaterialPBRGPU size must be 296 bytes");
static_assert(sizeof(HIPDecalGPU) == 128, "HIPDecalGPU size must be 128 bytes");
static_assert(sizeof(HIPEmissiveTriangleGPU) == 20, "HIPEmissiveTriangleGPU size must be 20 bytes");
static_assert(sizeof(HIPAirDustVolumeGPU) == 64, "HIPAirDustVolumeGPU size must be 64 bytes");
static_assert(sizeof(HIPPostProcessParams) == 160, "HIPPostProcessParams size must be 160 bytes");
static_assert(sizeof(HIPTextureDescGPU) == 24, "HIPTextureDescGPU size must be 24 bytes");

extern "C" bool HIP_RenderFrameTexture_C(const BVHNode *tlasNodes,
                                         std::uint32_t tlasNodeCount,
                                         const BVHNode *meshNodes,
                                         std::uint32_t meshNodeCount,
                                         const Triangle *triangles,
                                         std::uint32_t triCount,
                                         const HIPSceneInstanceGPU *instances,
                                         std::uint32_t instanceCount,
                                         const HIPLightGPU *lights,
                                         std::uint32_t lightCount,
                                         int rootIndex,
                                         const CameraDataCPU *camera,
                                         const HIPMaterialGPU *materials,
                                         std::uint32_t materialCount,
                                         const HIPMaterialPBRGPU *materialsPBR,
                                         std::uint32_t materialPBRCount,
                                         const HIPEmissiveTriangleGPU *emissiveTriangles,
                                         std::uint32_t emissiveTriangleCount,
                                         const HIPDecalGPU *decals,
                                         std::uint32_t decalCount,
                                         const HIPAirDustVolumeGPU *airDustVolumes,
                                         std::uint32_t airDustVolumeCount,
                                         const HIPPostProcessParams *postParams,
                                         const HIPTextureDescGPU *textureDescs,
                                         std::uint32_t textureCount,
                                         const std::uint8_t *textureTexels,
                                         std::size_t textureTexelBytes,
                                         std::uint64_t sceneRevision,
                                         std::uint64_t metaGeneration,
                                         std::uint32_t accumulatedSampleCountBefore,
                                         Vec3 *framebuffer);

extern "C" bool HIP_GetLastTextureExecutionProfile_C(HIPTextureExecutionProfile *outProfile);

extern "C" void HIP_ResetAccumulation_C();
