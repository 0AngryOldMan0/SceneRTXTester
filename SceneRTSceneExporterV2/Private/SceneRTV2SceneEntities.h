#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
}
class ULightComponent;
class UDecalComponent;
class UCameraComponent;
class UPostProcessComponent;
class APostProcessVolume;

namespace SceneRTV2::Lights
{
    void CollectLight(ULightComponent* LightComp, FExportContext& Ctx);
    void CollectSkyLight(class USkyLightComponent* SkyLight, FExportContext& Ctx);
}

namespace SceneRTV2::Decals
{
    void CollectDecal(UDecalComponent* DecalComp, FExportContext& Ctx);
}

namespace SceneRTV2::Cameras
{
    void CollectCamera(UCameraComponent* CameraComp, FExportContext& Ctx);
}

namespace SceneRTV2::PostProcess
{
    /**
     * Fully serializes FPostProcessSettings via UE Reflection.
     *
     * V1 manually dumped 9 fields, leaving exposure / Lumen / tonemap / bloom
     * silently zero on the consumer side — main cause of "lighting looks
     * different". V2 walks FPostProcessSettings::StaticStruct() and emits every
     * BoolOverride_* flag together with its companion value.
     */
    void CollectPostProcessComponent(UPostProcessComponent* Comp, FExportContext& Ctx);
    void CollectPostProcessVolume(APostProcessVolume* Volume, FExportContext& Ctx);

    /** Helper used by camera path too. */
    TSharedPtr<FJsonObject> ReflectPostProcessSettings(const struct FPostProcessSettings& Settings);
}

namespace SceneRTV2::Atmosphere
{
    void CollectSkyAtmosphere(class USkyAtmosphereComponent* Comp, FExportContext& Ctx);
    void CollectHeightFog(class UExponentialHeightFogComponent* Comp, FExportContext& Ctx);
    void CollectVolumetricCloud(class UVolumetricCloudComponent* Comp, FExportContext& Ctx);
}
