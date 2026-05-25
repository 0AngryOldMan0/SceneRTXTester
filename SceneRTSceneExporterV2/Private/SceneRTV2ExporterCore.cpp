#include "SceneRTV2ExporterCore.h"
#include "SceneRTV2ExportSettings.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"
#include "SceneRTV2Material.h"
#include "SceneRTV2Texture.h"
#include "SceneRTV2Mesh.h"
#include "SceneRTV2SplineMesh.h"
#include "SceneRTV2Landscape.h"
#include "SceneRTV2SceneEntities.h"
#include "SceneRTV2Writer.h"

#include "Editor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "Logging/LogMacros.h"
#include "GameFramework/Actor.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFilemanager.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/HierarchicalInstancedStaticMeshComponent.h"
#include "Components/SplineMeshComponent.h"
#include "Components/SplineComponent.h"
#include "Components/DecalComponent.h"
#include "Components/LightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/PostProcessComponent.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Components/VolumetricCloudComponent.h"
#include "Camera/CameraComponent.h"
#include "Engine/PostProcessVolume.h"
#include "Landscape.h"
#include "LandscapeProxy.h"

DEFINE_LOG_CATEGORY_STATIC(LogSceneRTV2, Log, All);

namespace
{
    using namespace SceneRTV2;

    /** Ensure output directory exists; resolve relative paths against project dir. */
    bool PrepareOutputDir(FExportContext& Ctx, FString* OutError)
    {
        FString Dir = Ctx.Settings->OutputDirectory;
        if (Dir.IsEmpty())
        {
            Dir = FPaths::ProjectSavedDir() / TEXT("SceneRTV2") / Ctx.BundleName;
        }
        Dir = FPaths::ConvertRelativePathToFull(Dir);
        IFileManager::Get().MakeDirectory(*Dir, /*Tree=*/true);
        if (!IFileManager::Get().DirectoryExists(*Dir))
        {
            if (OutError) { *OutError = FString::Printf(TEXT("Cannot create output dir %s"), *Dir); }
            return false;
        }
        Ctx.OutputDir = Dir;
        return true;
    }

    /** Phase 1: traverse the world, dispatching each component to the right collector. */
    void CollectScene(FExportContext& Ctx)
    {
        UWorld* World = Ctx.World;
        check(World);

        TSet<AActor*> SplineActorsHandled;

        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (!Actor || Actor->IsPendingKillPending()) { continue; }
            if (!Ctx.Settings->bExportHiddenActors && Actor->IsHidden()) { continue; }

            // ---- Spline-mesh actors: one pass per actor so we can compute arclen
            //      offsets across the chain instead of per-component.
            bool bHasSplineMesh = false;
            for (UActorComponent* C : Actor->GetComponents())
            {
                if (Cast<USplineMeshComponent>(C)) { bHasSplineMesh = true; break; }
            }
            if (bHasSplineMesh && Ctx.Settings->bExportSplineMeshGeometry)
            {
                SplineMesh::CollectActorSplineChain(Actor, Ctx);
                SplineActorsHandled.Add(Actor);
            }

            // ---- Landscape
            if (Ctx.Settings->bExportLandscape)
            {
                if (ALandscape* L = Cast<ALandscape>(Actor))
                {
                    Landscape::CollectLandscape(L, Ctx);
                    continue;
                }
                if (ALandscapeProxy* LP = Cast<ALandscapeProxy>(Actor))
                {
                    Landscape::CollectLandscapeProxy(LP, Ctx);
                    continue;
                }
            }

            // ---- Post-process volumes
            if (Ctx.Settings->bExportPostProcess)
            {
                if (APostProcessVolume* V = Cast<APostProcessVolume>(Actor))
                {
                    PostProcess::CollectPostProcessVolume(V, Ctx);
                }
            }

            // ---- Per-component dispatch
            for (UActorComponent* Component : Actor->GetComponents())
            {
                if (!Component || !Component->IsRegistered()) { continue; }

                if (USplineMeshComponent* /*Spl*/ = Cast<USplineMeshComponent>(Component))
                {
                    continue; // handled above
                }
                if (UInstancedStaticMeshComponent* ISM = Cast<UInstancedStaticMeshComponent>(Component))
                {
                    Mesh::CollectInstancedStatic(ISM, Ctx);
                    continue;
                }
                if (UStaticMeshComponent* SMC = Cast<UStaticMeshComponent>(Component))
                {
                    Mesh::CollectStaticMesh(SMC, Ctx);
                    continue;
                }
                if (USkeletalMeshComponent* SKC = Cast<USkeletalMeshComponent>(Component))
                {
                    Mesh::CollectSkeletalMesh(SKC, Ctx);
                    continue;
                }
                if (UDecalComponent* DC = Cast<UDecalComponent>(Component))
                {
                    if (Ctx.Settings->bExportDecals) { Decals::CollectDecal(DC, Ctx); }
                    continue;
                }
                if (USkyLightComponent* SL = Cast<USkyLightComponent>(Component))
                {
                    if (Ctx.Settings->bExportLights) { Lights::CollectSkyLight(SL, Ctx); }
                    continue;
                }
                if (ULightComponent* LC = Cast<ULightComponent>(Component))
                {
                    if (Ctx.Settings->bExportLights) { Lights::CollectLight(LC, Ctx); }
                    continue;
                }
                if (UCameraComponent* CC = Cast<UCameraComponent>(Component))
                {
                    if (Ctx.Settings->bExportCameras) { Cameras::CollectCamera(CC, Ctx); }
                    continue;
                }
                if (UPostProcessComponent* PPC = Cast<UPostProcessComponent>(Component))
                {
                    if (Ctx.Settings->bExportPostProcess)
                    {
                        PostProcess::CollectPostProcessComponent(PPC, Ctx);
                    }
                    continue;
                }
                if (USkyAtmosphereComponent* SAC = Cast<USkyAtmosphereComponent>(Component))
                {
                    if (Ctx.Settings->bExportAtmosphere) { Atmosphere::CollectSkyAtmosphere(SAC, Ctx); }
                    continue;
                }
                if (UExponentialHeightFogComponent* EHC = Cast<UExponentialHeightFogComponent>(Component))
                {
                    if (Ctx.Settings->bExportFog) { Atmosphere::CollectHeightFog(EHC, Ctx); }
                    continue;
                }
                if (UVolumetricCloudComponent* VCC = Cast<UVolumetricCloudComponent>(Component))
                {
                    if (Ctx.Settings->bExportVolumetricClouds) { Atmosphere::CollectVolumetricCloud(VCC, Ctx); }
                    continue;
                }
            }
        }
    }

    /** Phase 2: serialize all artifacts to disk in dependency order. */
    bool WriteAll(FExportContext& Ctx, FString* OutError)
    {
        // Textures first (materials reference them by id), then materials (primitives
        // reference them), then meshes, then scene.
        if (!Writer::WriteTextureContainer(Ctx, OutError))   { return false; }
        if (!Writer::WriteMaterialsJson(Ctx, OutError))      { return false; }
        if (!Writer::WriteMeshContainer(Ctx, OutError))      { return false; }
        if (!Writer::WriteSceneJson(Ctx, OutError))          { return false; }
        if (Ctx.Settings->bExportLights      && !Writer::WriteLightsJson(Ctx, OutError))      { return false; }
        if (Ctx.Settings->bExportDecals      && !Writer::WriteDecalsJson(Ctx, OutError))      { return false; }
        if (Ctx.Settings->bExportCameras     && !Writer::WriteCamerasJson(Ctx, OutError))     { return false; }
        if ((Ctx.Settings->bExportAtmosphere || Ctx.Settings->bExportFog || Ctx.Settings->bExportVolumetricClouds)
            && !Writer::WriteAtmosphereJson(Ctx, OutError)) { return false; }
        if (Ctx.Settings->bExportPostProcess && !Writer::WritePostProcessJson(Ctx, OutError)) { return false; }
        if (Ctx.Settings->bExportLandscape   && !Writer::WriteLandscapesJson(Ctx, OutError))  { return false; }
        if (!Writer::WriteManifest(Ctx, OutError)) { return false; }
        if (Ctx.Settings->bWriteValidationReport && !Writer::WriteValidationReport(Ctx, OutError))
        {
            return false;
        }
        return true;
    }
}

bool FSceneRTV2ExporterCore::ExportEditorWorld(const USceneRTV2ExportSettings* InSettings, FString* OutError)
{
    if (!GEditor)
    {
        if (OutError) { *OutError = TEXT("No GEditor available"); }
        return false;
    }
    UWorld* World = GEditor->GetEditorWorldContext().World();
    return ExportWorld(World, InSettings, OutError);
}

bool FSceneRTV2ExporterCore::ExportWorld(UWorld* InWorld, const USceneRTV2ExportSettings* InSettings, FString* OutError)
{
    if (!InWorld)
    {
        if (OutError) { *OutError = TEXT("World is null"); }
        return false;
    }
    if (!InSettings)
    {
        if (OutError) { *OutError = TEXT("Settings is null"); }
        return false;
    }

    SceneRTV2::FExportContext Ctx;
    Ctx.Settings = InSettings;
    Ctx.World = InWorld;
    Ctx.BundleName = InSettings->BundleName.IsEmpty()
        ? InWorld->GetName()
        : InSettings->BundleName;
    Ctx.SceneGuid = FGuid::NewGuid();

    if (!PrepareOutputDir(Ctx, OutError)) { return false; }

    UE_LOG(LogSceneRTV2, Log, TEXT("V2 export: world=%s out=%s guid=%s"),
        *InWorld->GetName(), *Ctx.OutputDir, *Ctx.SceneGuid.ToString());

    CollectScene(Ctx);

    if (!WriteAll(Ctx, OutError)) { return false; }

    UE_LOG(LogSceneRTV2, Log,
        TEXT("V2 export complete: meshes=%d materials=%d textures=%d prims=%d lights=%d decals=%d issues=%d"),
        Ctx.Meshes.Num(), Ctx.Materials.Num(), Ctx.Textures.Num(), Ctx.Primitives.Num(),
        Ctx.Lights.Num(), Ctx.Decals.Num(), Ctx.Issues.Num());

    return true;
}
