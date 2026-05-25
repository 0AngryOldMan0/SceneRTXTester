#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
}

namespace SceneRTV2::Writer
{
    /** Writes meshes_v2.bin. */
    bool WriteMeshContainer(const FExportContext& Ctx, FString* OutError);
    /** Writes textures_v2.bin. */
    bool WriteTextureContainer(const FExportContext& Ctx, FString* OutError);
    /** Writes scene.json (actors, primitives, instancing, refs). */
    bool WriteSceneJson(const FExportContext& Ctx, FString* OutError);
    /** Writes materials.json. */
    bool WriteMaterialsJson(const FExportContext& Ctx, FString* OutError);
    /** Writes lights.json. */
    bool WriteLightsJson(const FExportContext& Ctx, FString* OutError);
    /** Writes decals.json. */
    bool WriteDecalsJson(const FExportContext& Ctx, FString* OutError);
    /** Writes cameras.json. */
    bool WriteCamerasJson(const FExportContext& Ctx, FString* OutError);
    /** Writes atmosphere.json (SkyAtmosphere, ExponentialHeightFog, VolumetricCloud). */
    bool WriteAtmosphereJson(const FExportContext& Ctx, FString* OutError);
    /** Writes postprocess.json. */
    bool WritePostProcessJson(const FExportContext& Ctx, FString* OutError);
    /** Writes landscapes.json (per-landscape metadata + layer weight refs). */
    bool WriteLandscapesJson(const FExportContext& Ctx, FString* OutError);
    /** Writes manifest.json (versions, GUID, sizes, sha256 of every artifact). */
    bool WriteManifest(const FExportContext& Ctx, FString* OutError);
    /** Writes validation.json (issues collected during pipeline). */
    bool WriteValidationReport(const FExportContext& Ctx, FString* OutError);
}
