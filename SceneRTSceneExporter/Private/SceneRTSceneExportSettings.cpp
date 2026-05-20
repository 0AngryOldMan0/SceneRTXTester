#include "SceneRTSceneExportSettings.h"

#include "Misc/Paths.h"

USceneRTSceneExportSettings::USceneRTSceneExportSettings()
{
    OutputDirectory = TEXT("");
    SceneFileName = TEXT("scene.json");
    MeshesFileName = TEXT("meshes.bin");

    UnitScale = 1.0f;
    bMirrorYToRightHanded = true;
    bExportOnlyVisibleComponents = true;
    bExportHiddenActors = false;
    bExportAllStaticMeshLODs = true;
    bExportSplineMeshGeometry = true;
    bExportLandscapeGeometry = true;
    bExportSkeletalMeshMetadata = true;
    bExportMaterialTextures = true;
    bExportPostProcess = true;
    bExportFog = true;
    bExportAtmosphere = true;
    bExportDecals = true;
    bExportLights = true;
    bExportCameras = true;
    bExportActorTags = true;
    bExportDataLayers = true;
    MaxTextureExportSize = 0;
}
