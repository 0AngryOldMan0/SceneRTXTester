#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "SceneRTV2ExportSettings.generated.h"

UENUM(BlueprintType)
enum class ESceneRTV2CoordinateConvention : uint8
{
    /** Keep Unreal native coordinates: left-handed, Z-up, cm units. */
    UnrealNative      UMETA(DisplayName = "Unreal Native (LH Z-up cm)"),
    /** Right-handed, Y-up, metres. Standard for most external ray tracers. */
    RightHandedYUpM   UMETA(DisplayName = "RH Y-up metres"),
};

UENUM(BlueprintType)
enum class ESceneRTV2TextureFormat : uint8
{
    /** Preserve UE source PixelFormat byte-for-byte (BCn included). */
    PreserveSource    UMETA(DisplayName = "Preserve source format"),
    /** Decompress + write RGBA8 (sRGB respected). Reliable cross-platform. */
    RGBA8             UMETA(DisplayName = "Decompress to RGBA8"),
    /** Float HDR-safe path. */
    RGBA16F           UMETA(DisplayName = "Decompress to RGBA16F"),
};

UCLASS(Config = Editor, defaultconfig, BlueprintType)
class SCENERTSCENEEXPORTERV2_API USceneRTV2ExportSettings : public UObject
{
    GENERATED_BODY()
public:
    USceneRTV2ExportSettings();

    UPROPERTY(EditAnywhere, Config, Category = "Output")
    FString OutputDirectory;

    UPROPERTY(EditAnywhere, Config, Category = "Output")
    FString BundleName;

    UPROPERTY(EditAnywhere, Config, Category = "Output")
    ESceneRTV2CoordinateConvention CoordinateConvention = ESceneRTV2CoordinateConvention::UnrealNative;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    bool bExportAllLODs = false;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    int32 LowestLODIndex = 0;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    bool bExportNaniteFallback = true;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    bool bExportSplineMeshGeometry = true;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    int32 SplineMeshSegmentsPerSpline = 16;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    bool bWriteArclenUv = true;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    bool bExportInstancedFoliage = true;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    bool bExportLandscape = true;

    UPROPERTY(EditAnywhere, Config, Category = "Geometry")
    int32 LandscapeExportLOD = 0;

    UPROPERTY(EditAnywhere, Config, Category = "Materials")
    bool bResolveAllParameterAssociations = true;

    UPROPERTY(EditAnywhere, Config, Category = "Materials")
    bool bExportMaterialGraphHints = true;

    UPROPERTY(EditAnywhere, Config, Category = "Materials")
    bool bDeduplicateOnlyByResolvedParams = true;

    UPROPERTY(EditAnywhere, Config, Category = "Textures")
    ESceneRTV2TextureFormat TextureFormat = ESceneRTV2TextureFormat::PreserveSource;

    UPROPERTY(EditAnywhere, Config, Category = "Textures", meta = (ClampMin = "0"))
    int32 MaxTextureExportSize = 0;

    UPROPERTY(EditAnywhere, Config, Category = "Textures")
    bool bBakeRuntimeVirtualTextures = true;

    UPROPERTY(EditAnywhere, Config, Category = "Textures")
    bool bBakeCompositeTextures = true;

    UPROPERTY(EditAnywhere, Config, Category = "Textures")
    int32 RvtBakeTileResolution = 1024;

    UPROPERTY(EditAnywhere, Config, Category = "Lighting")
    bool bExportLights = true;

    UPROPERTY(EditAnywhere, Config, Category = "Lighting")
    bool bExportIESProfiles = true;

    UPROPERTY(EditAnywhere, Config, Category = "Lighting")
    bool bExportSkyLightCubemap = true;

    UPROPERTY(EditAnywhere, Config, Category = "Lighting")
    bool bExportLumenSettings = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportDecals = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportCameras = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportAtmosphere = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportFog = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportVolumetricClouds = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportPostProcess = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportActorTags = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportDataLayers = true;

    UPROPERTY(EditAnywhere, Config, Category = "Scene")
    bool bExportHiddenActors = false;

    UPROPERTY(EditAnywhere, Config, Category = "Diagnostics")
    bool bWriteValidationReport = true;

    UPROPERTY(EditAnywhere, Config, Category = "Diagnostics")
    bool bFailOnUnresolvedTexture = false;
};
