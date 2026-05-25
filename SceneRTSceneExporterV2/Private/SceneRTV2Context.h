#pragma once

#include "CoreMinimal.h"
#include "SceneRTV2Format.h"

class UWorld;
class UMaterialInterface;
class UTexture;
class USceneRTV2ExportSettings;

namespace SceneRTV2
{
    /** Stable, content-derived identifier. Format: "<kind>:<sha1-prefix>". */
    struct FStableId
    {
        FString Value;
        bool IsValid() const { return !Value.IsEmpty(); }
        bool operator==(const FStableId& O) const { return Value == O.Value; }
    };
    FORCEINLINE uint32 GetTypeHash(const FStableId& Id) { return GetTypeHash(Id.Value); }

    enum class EParamKind : uint8 { Scalar, Vector, Texture, Switch, RuntimeVirtualTexture };

    struct FMaterialParamRecord
    {
        FString Name;
        FString GroupName;
        EParamKind Kind = EParamKind::Scalar;
        float Scalar = 0.f;
        FLinearColor Vector = FLinearColor::Black;
        FStableId TextureId;             // empty if no texture
        bool bSwitch = false;
        uint8 ParameterAssociation = 0;  // EMaterialParameterAssociation
        int32 LayerIndex = INDEX_NONE;
        FString SourceFunctionPath;      // material function that owns the parameter
    };

    struct FMaterialRecord
    {
        FStableId Id;
        FString ParentMaterialPath;      // resolved base UMaterial path
        FString MaterialInstancePath;    // MI path if this came from an MI
        SceneRTSceneExporterV2::EMaterialDomain Domain = SceneRTSceneExporterV2::EMaterialDomain::Surface;
        SceneRTSceneExporterV2::EShadingModel ShadingModel = SceneRTSceneExporterV2::EShadingModel::DefaultLit;
        bool bTwoSided = false;
        bool bUsedWithNanite = false;
        bool bIsThinTranslucent = false;
        TArray<FMaterialParamRecord> Params;
        uint64 ResolvedParamsHash = 0;   // dedup key (parent + params, not just MI name)
    };

    struct FTextureRecord
    {
        FStableId Id;
        FString SourcePath;
        int32 Width = 0;
        int32 Height = 0;
        int32 Depth = 1;
        int32 NumMips = 0;
        int32 NumSlices = 1;
        SceneRTSceneExporterV2::ETexturePixelFormat PixelFormat = SceneRTSceneExporterV2::ETexturePixelFormat::Unknown;
        bool bSrgb = false;
        bool bNormalMap = false;
        bool bHDR = false;
        bool bRvtBaked = false;
        bool bCompositeBaked = false;
        TArray<uint8> Payload;           // mip chain bytes
    };

    struct FMeshSection
    {
        uint32 SectionIndex = 0;
        uint32 MaterialSlotIndex = 0;
        FString MaterialSlotName;
        uint32 FirstIndex = 0;
        uint32 IndexCount = 0;
        uint32 MinVertex = 0;
        uint32 MaxVertex = 0;
        bool bCastShadow = true;
        bool bVisibleInRayTracing = true;
    };

    struct FMeshLod
    {
        int32 LodIndex = 0;
        TArray<FVector3f> Positions;
        TArray<FVector3f> Normals;
        TArray<FVector4f> Tangents;       // xyz = tangent, w = bitangent sign
        TArray<FColor> Colors;
        TArray<TArray<FVector2f>> UVs;    // [channel][vertex]
        TArray<uint32> Indices;
        TArray<FMeshSection> Sections;
        TArray<float> ArclenAlongSpline;  // populated only for spline-mesh segments
        TArray<FVector4f> SkinIndices;    // 4 bone indices (cast from uint16)
        TArray<FVector4f> SkinWeights;
        bool bHasVertexColors = false;
    };

    struct FMeshAsset
    {
        FStableId Id;
        FString DisplayName;
        FString SourceAssetPath;
        SceneRTSceneExporterV2::EMeshSourceKind Kind = SceneRTSceneExporterV2::EMeshSourceKind::StaticMesh;
        TArray<FMeshLod> Lods;
        FBox LocalBounds = FBox(ForceInitToZero);
        TArray<FStableId> SlotMaterialIds; // per material slot — references into Materials table
    };

    struct FLandscapeLayerWeight
    {
        FString LayerName;
        FStableId WeightTextureId;   // packed weightmap (one channel per layer slot)
        int32 ChannelIndex = 0;      // 0..3 = R/G/B/A in the weightmap
        FLinearColor PhysicalMaterialBaseColor = FLinearColor::Black; // optional hint
    };

    struct FLandscapeComponentInfo
    {
        FStableId MeshId;             // triangulated component mesh asset id
        FStableId InstanceId;         // primitive instance id (links into FPrimitiveInstance table)
        FString OwnerActorPath;
        FString ComponentPath;
        int32 SectionBaseX = 0;
        int32 SectionBaseY = 0;
        FStableId HeightmapTextureId; // per-component heightmap (optional)
        TArray<FLandscapeLayerWeight> LayerWeights;
    };

    struct FLandscapeRecord
    {
        FStableId Id;
        FString ActorPath;
        FString LandscapeGuid;        // ALandscape::LandscapeGuid — shared by streaming proxies
        FTransform Transform;
        int32 ComponentSizeQuads = 0;
        int32 SubsectionSizeQuads = 0;
        int32 NumSubsections = 0;
        int32 ExportLOD = 0;
        FStableId BaseMaterialId;
        FStableId HoleMaterialId;
        TArray<FLandscapeComponentInfo> Components;
    };
    {
        FStableId Id;
        FStableId MeshId;
        FString OwnerActorPath;
        FString ComponentPath;
        FTransform Transform;
        TArray<FStableId> OverrideMaterialIds; // per-slot overrides applied on top of mesh defaults
        TArray<FString> Tags;
        TArray<FString> DataLayers;
        bool bVisible = true;
        bool bCastShadow = true;
        bool bAffectsRayTracing = true;
        uint32 LightingChannels = 0x1;
        // Spline continuity hints (only for spline-mesh-segment instances):
        FString SplineGroupId;
        int32 SplineSegmentIndex = INDEX_NONE;
        float SplineArclenStart = 0.f;
        float SplineArclenEnd = 0.f;
    };

    struct FLightRecord
    {
        FStableId Id;
        FString OwnerActorPath;
        FString ComponentPath;
        FString LightType;               // "Directional" / "Point" / "Spot" / "Rect" / "Sky"
        FTransform Transform;
        FLinearColor Color = FLinearColor::White;
        float Intensity = 0.f;
        FString IntensityUnits;          // "Lumens" / "Candelas" / "EV100" / "Unitless"
        float AttenuationRadius = 0.f;
        float SourceRadius = 0.f;
        float SoftSourceRadius = 0.f;
        float SourceLength = 0.f;
        float InnerConeAngle = 0.f;
        float OuterConeAngle = 0.f;
        float Temperature = 6500.f;
        bool bUseTemperature = false;
        bool bCastShadows = true;
        bool bCastRaytracedShadows = true;
        FStableId IesProfileTextureId;
        FStableId SkyCubemapTextureId;
        TSharedPtr<FJsonObject> ExtraReflection; // engine-specific extras
    };

    struct FDecalRecord
    {
        FStableId Id;
        FString OwnerActorPath;
        FString ComponentPath;
        FStableId MaterialId;
        FTransform Transform;
        FVector DecalSize = FVector(64, 256, 256);
        int32 SortOrder = 0;
        float FadeScreenSize = 0.01f;
        float Opacity = 1.f;
    };

    struct FCameraRecord
    {
        FStableId Id;
        FString OwnerActorPath;
        FTransform Transform;
        float FieldOfView = 90.f;
        float AspectRatio = 16.f/9.f;
        float Focal = 0.f;
        float Aperture = 0.f;
        float FocusDistance = 0.f;
        TSharedPtr<FJsonObject> PostProcess;
    };

    struct FPostProcessRecord
    {
        FStableId Id;
        FString OwnerActorPath;
        float Priority = 0.f;
        float BlendRadius = 0.f;
        float BlendWeight = 1.f;
        bool bUnbound = false;
        FBox Bounds = FBox(ForceInitToZero);
        TSharedPtr<FJsonObject> Settings; // full reflection of FPostProcessSettings
    };

    struct FAtmosphereRecord
    {
        FString Kind;                    // SkyAtmosphere / ExponentialHeightFog / VolumetricCloud
        FString OwnerActorPath;
        TSharedPtr<FJsonObject> Settings;
    };

    struct FValidationIssue
    {
        FString Severity;                // "warn" / "error"
        FString Category;                // "texture" / "material" / "mesh" / ...
        FString Message;
        FString RelatedId;
    };

    /** Central mutable container passed through every collector. */
    struct FExportContext
    {
        const USceneRTV2ExportSettings* Settings = nullptr;
        UWorld* World = nullptr;
        FString OutputDir;
        FString BundleName;
        FGuid SceneGuid;

        TArray<FMeshAsset> Meshes;
        TArray<FMaterialRecord> Materials;
        TArray<FTextureRecord> Textures;
        TArray<FPrimitiveInstance> Primitives;
        TArray<FLightRecord> Lights;
        TArray<FDecalRecord> Decals;
        TArray<FCameraRecord> Cameras;
        TArray<FPostProcessRecord> PostProcess;
        TArray<FAtmosphereRecord> Atmosphere;
        TArray<FLandscapeRecord> Landscapes;
        TArray<FValidationIssue> Issues;

        // Dedup tables.
        TMap<FString, int32> MeshAssetByKey;
        TMap<uint64,  int32> MaterialByHash;
        TMap<FString, int32> TextureByPath;
        TMap<FString, int32> LightByPath;
        TMap<FString, int32> LandscapeByGuid;     // ALandscape::LandscapeGuid → index in Landscapes

        void AddIssue(const FString& Severity, const FString& Category,
                      const FString& Message, const FString& RelatedId = FString());
    };
}
