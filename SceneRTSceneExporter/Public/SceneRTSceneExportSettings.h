#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "SceneRTSceneExportSettings.generated.h"

UCLASS(Config=EditorPerProjectUserSettings)
class SCENERTSCENEEXPORTER_API USceneRTSceneExportSettings : public UObject
{
    GENERATED_BODY()

public:
    USceneRTSceneExportSettings();

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    FString OutputDirectory;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    FString SceneFileName;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    FString MeshesFileName;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    float UnitScale;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bMirrorYToRightHanded;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportOnlyVisibleComponents;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportHiddenActors;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportAllStaticMeshLODs;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportSplineMeshGeometry;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportLandscapeGeometry;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportSkeletalMeshMetadata;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportMaterialTextures;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportPostProcess;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportFog;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportAtmosphere;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportDecals;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportLights;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportCameras;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportActorTags;

    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export")
    bool bExportDataLayers;

    /** 0 = original size, otherwise longest side limit for exported textures */
    UPROPERTY(EditAnywhere, Config, Category = "SceneRTX Export", meta=(ClampMin="0"))
    int32 MaxTextureExportSize;
};
