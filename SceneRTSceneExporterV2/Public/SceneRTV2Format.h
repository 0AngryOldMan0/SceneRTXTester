#pragma once

#include "CoreMinimal.h"

/**
 * SceneRTSceneExporterV2 — binary container layout.
 *
 * Format goal: stable across editor sessions, deterministic, versioned, parseable
 * without scene.json (each block self-describes geometry + bone bindings).
 *
 * File map written by V2 pipeline:
 *   meshes_v2.bin    — geometry per mesh asset (LODs, sections, skin, morph)
 *   textures_v2.bin  — packed texture payloads (DDS or raw mip chain)
 *   scene.json       — actors / components / instancing / overrides
 *   materials.json   — material graphs (parent ref + resolved parameters)
 *   lights.json      — light component dump (incl. IES paths, sky cubemap ref)
 *   postprocess.json — full FPostProcessSettings reflection dump
 *   manifest.json    — version, scene GUID, file list, checksums
 *
 * Strict naming convention: every artifact references the manifest scene_guid,
 * so consumer can reject mismatched bundles.
 */
namespace SceneRTSceneExporterV2
{
    static constexpr uint32 kV2MeshMagic    = 0x32525453u; // "STR2"
    static constexpr uint32 kV2TextureMagic = 0x32585431u; // "1TX2"
    static constexpr uint32 kV2FormatMajor  = 2;
    static constexpr uint32 kV2FormatMinor  = 0;

    enum class EVertexAttributeMask : uint32
    {
        None         = 0,
        Position     = 1 << 0,
        Normal       = 1 << 1,
        Tangent      = 1 << 2,
        Color        = 1 << 3,
        UV0          = 1 << 4,
        UV1          = 1 << 5,
        UV2          = 1 << 6,
        UV3          = 1 << 7,
        SkinIndices  = 1 << 8,
        SkinWeights  = 1 << 9,
        ArclenAlongSpline = 1 << 10,
    };

    enum class EMeshSourceKind : uint8
    {
        StaticMesh         = 0,
        SkeletalMesh       = 1,
        InstancedStatic    = 2,
        HierarchicalISM    = 3,
        SplineMeshSegment  = 4,
        NaniteFallback     = 5,
        LandscapeTriangle  = 6,
        ProceduralMesh     = 7,
    };

    enum class EMaterialDomain : uint8
    {
        Surface        = 0,
        Decal          = 1,
        LightFunction  = 2,
        Volume         = 3,
        UI             = 4,
        PostProcess    = 5,
    };

    enum class EShadingModel : uint8
    {
        Unlit               = 0,
        DefaultLit          = 1,
        Subsurface          = 2,
        PreintegratedSkin   = 3,
        ClearCoat           = 4,
        SubsurfaceProfile   = 5,
        TwoSidedFoliage     = 6,
        Hair                = 7,
        Cloth               = 8,
        Eye                 = 9,
        SingleLayerWater    = 10,
        ThinTranslucent     = 11,
        Strata              = 12,
    };

#pragma pack(push, 1)

    /** meshes_v2.bin header. Followed by [NumMeshes] FMeshAssetHeader records. */
    struct FMeshContainerHeader
    {
        uint32 Magic       = kV2MeshMagic;
        uint16 VersionMajor = kV2FormatMajor;
        uint16 VersionMinor = kV2FormatMinor;
        uint32 NumMeshes    = 0;
        uint32 TocOffset    = 0;     // offset to TOC (uint64[NumMeshes] of asset offsets)
        uint64 PayloadStart = 0;     // first byte after TOC
        uint64 TotalSize    = 0;
        uint64 SceneGuidLo  = 0;
        uint64 SceneGuidHi  = 0;
    };

    /** Per-mesh header. Followed by [NumLods] FMeshLodHeader. */
    struct FMeshAssetHeader
    {
        uint8  Kind       = 0;       // EMeshSourceKind
        uint8  NumLods    = 0;
        uint16 NumSections = 0;
        uint32 AttributeMask = 0;    // EVertexAttributeMask bits across all LODs
        uint64 BoundsCenterX = 0;    // double bit-cast — preserves full precision
        uint64 BoundsCenterY = 0;
        uint64 BoundsCenterZ = 0;
        uint64 BoundsExtentX = 0;
        uint64 BoundsExtentY = 0;
        uint64 BoundsExtentZ = 0;
        uint64 PayloadOffset = 0;    // relative to container start
        uint64 PayloadSize   = 0;
        uint8  StableIdHash[20];     // SHA1 of stable id string
        uint8  Pad[4]        = {0,0,0,0}; // align to 96 bytes
    };

    struct FMeshLodHeader
    {
        uint32 LodIndex      = 0;
        uint32 NumVertices   = 0;
        uint32 NumIndices    = 0;
        uint32 NumSections   = 0;
        uint32 IndexStride   = 0;    // 2 or 4 bytes
        uint32 SkinInfluences = 0;   // 0/4/8
        uint64 PositionsOffset = 0;
        uint64 NormalsOffset   = 0;
        uint64 TangentsOffset  = 0;
        uint64 ColorsOffset    = 0;
        uint64 UVsOffset       = 0;  // interleaved per-channel
        uint64 IndicesOffset   = 0;
        uint64 SkinIndicesOffset = 0;
        uint64 SkinWeightsOffset = 0;
        uint64 SectionsOffset    = 0;
        uint64 ArclenOffset      = 0; // only for spline mesh segments
    };

    struct FMeshSectionRange
    {
        uint32 SectionIndex      = 0;
        uint32 MaterialSlotIndex = 0;
        uint32 FirstIndex        = 0;
        uint32 IndexCount        = 0;
        uint32 MinVertexIndex    = 0;
        uint32 MaxVertexIndex    = 0;
        uint32 NumUvChannels     = 0;
        uint32 CastShadowFlags   = 0;
    };

    /** textures_v2.bin header. Followed by [NumTextures] FTextureRecord. */
    struct FTextureContainerHeader
    {
        uint32 Magic        = kV2TextureMagic;
        uint16 VersionMajor = kV2FormatMajor;
        uint16 VersionMinor = kV2FormatMinor;
        uint32 NumTextures  = 0;
        uint32 Reserved     = 0; // align to 48
        uint64 TocOffset    = 0;
        uint64 TotalSize    = 0;
        uint64 SceneGuidLo  = 0;
        uint64 SceneGuidHi  = 0;
    };

    enum class ETexturePixelFormat : uint16
    {
        Unknown    = 0,
        R8         = 1,
        RG8        = 2,
        RGBA8      = 3,
        BGRA8      = 4,
        R16F       = 5,
        RG16F      = 6,
        RGBA16F    = 7,
        R32F       = 8,
        RGBA32F    = 9,
        BC1        = 10,
        BC3        = 11,
        BC4        = 12,
        BC5        = 13,
        BC6H       = 14,
        BC7        = 15,
    };

    struct FTextureRecord
    {
        uint16 PixelFormat       = 0; // ETexturePixelFormat
        uint16 NumMips           = 0;
        uint32 Width             = 0;
        uint32 Height            = 0;
        uint32 Depth             = 0;   // for VolumeTexture
        uint32 NumSlices         = 0;   // for cube/array (cube=6)
        uint32 FlagsSrgb         : 1;
        uint32 FlagsNormalMap    : 1;
        uint32 FlagsHDR          : 1;
        uint32 FlagsRVTBaked     : 1;
        uint32 FlagsCompositeBaked : 1;
        uint32 FlagsReserved     : 27;
        uint64 PayloadOffset     = 0;
        uint64 PayloadSize       = 0;
        uint8  StableIdHash[20];        // SHA1 of stable id string
    };

#pragma pack(pop)

    static_assert(sizeof(FMeshContainerHeader)    == 48,  "FMeshContainerHeader layout drift");
    static_assert(sizeof(FMeshAssetHeader)        == 96,  "FMeshAssetHeader layout drift");
    static_assert(sizeof(FMeshLodHeader)          == 104, "FMeshLodHeader layout drift");
    static_assert(sizeof(FMeshSectionRange)       == 32,  "FMeshSectionRange layout drift");
    static_assert(sizeof(FTextureContainerHeader) == 48,  "FTextureContainerHeader layout drift");
    static_assert(sizeof(FTextureRecord)          == 60,  "FTextureRecord layout drift");
}
