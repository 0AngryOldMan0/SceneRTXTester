#include "SceneRTV2SceneEntities.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"
#include "SceneRTV2Material.h"
#include "SceneRTV2Texture.h"

#include "Components/LightComponent.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/SpotLightComponent.h"
#include "Components/RectLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/DecalComponent.h"
#include "Components/PostProcessComponent.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Components/VolumetricCloudComponent.h"
#include "Camera/CameraComponent.h"
#include "Engine/PostProcessVolume.h"
#include "Engine/Texture.h"
#include "Engine/TextureCube.h"
#include "GameFramework/Actor.h"
#include "JsonObjectConverter.h"
#include "UObject/UnrealType.h"

namespace SceneRTV2::Lights
{
    void CollectLight(ULightComponent* LightComp, FExportContext& Ctx)
    {
        if (!LightComp) { return; }

        FLightRecord Rec;
        Rec.Id = Identity::ForComponent(Ctx.SceneGuid, LightComp);
        Rec.OwnerActorPath = LightComp->GetOwner() ? LightComp->GetOwner()->GetPathName() : FString();
        Rec.ComponentPath = LightComp->GetPathName();
        Rec.Transform = LightComp->GetComponentTransform();
        Rec.Color = LightComp->GetLightColor();
        Rec.Intensity = LightComp->Intensity;
        Rec.bCastShadows = LightComp->CastShadows;
        Rec.Temperature = LightComp->Temperature;
        Rec.bUseTemperature = LightComp->bUseTemperature;

        if (UPointLightComponent* P = Cast<UPointLightComponent>(LightComp))
        {
            Rec.LightType = TEXT("Point");
            Rec.AttenuationRadius = P->AttenuationRadius;
            Rec.SourceRadius = P->SourceRadius;
            Rec.SoftSourceRadius = P->SoftSourceRadius;
            Rec.SourceLength = P->SourceLength;
            Rec.IntensityUnits = TEXT("Lumens");
            if (USpotLightComponent* S = Cast<USpotLightComponent>(P))
            {
                Rec.LightType = TEXT("Spot");
                Rec.InnerConeAngle = S->InnerConeAngle;
                Rec.OuterConeAngle = S->OuterConeAngle;
            }
        }
        else if (URectLightComponent* R = Cast<URectLightComponent>(LightComp))
        {
            Rec.LightType = TEXT("Rect");
            Rec.AttenuationRadius = R->AttenuationRadius;
            Rec.SourceRadius = R->SourceWidth;
            Rec.SourceLength = R->SourceHeight;
            Rec.IntensityUnits = TEXT("Lumens");
        }
        else if (UDirectionalLightComponent* /*D*/ = Cast<UDirectionalLightComponent>(LightComp))
        {
            Rec.LightType = TEXT("Directional");
            Rec.IntensityUnits = TEXT("Lux");
        }

        // IES profile texture: critical for "light spreads incorrectly".
        if (Ctx.Settings->bExportIESProfiles)
        {
            if (UPointLightComponent* P = Cast<UPointLightComponent>(LightComp))
            {
                if (P->IESTexture)
                {
                    Rec.IesProfileTextureId = SceneRTV2::Texture::Resolve(P->IESTexture, Ctx);
                }
            }
        }

        // Capture extra reflection fields (light functions, lighting channels, etc).
        Rec.ExtraReflection = FJsonObjectConverter::UStructToJsonObject(
            LightComp->GetClass(), LightComp,
            /*CheckFlags*/ 0, /*SkipFlags*/ CPF_Transient | CPF_Deprecated);

        Ctx.Lights.Add(MoveTemp(Rec));
    }

    void CollectSkyLight(USkyLightComponent* SkyLight, FExportContext& Ctx)
    {
        if (!SkyLight) { return; }

        FLightRecord Rec;
        Rec.Id = Identity::ForComponent(Ctx.SceneGuid, SkyLight);
        Rec.OwnerActorPath = SkyLight->GetOwner() ? SkyLight->GetOwner()->GetPathName() : FString();
        Rec.ComponentPath = SkyLight->GetPathName();
        Rec.Transform = SkyLight->GetComponentTransform();
        Rec.LightType = TEXT("Sky");
        Rec.Intensity = SkyLight->Intensity;
        Rec.Color = SkyLight->GetLightColor();
        Rec.IntensityUnits = TEXT("Unitless");

        if (Ctx.Settings->bExportSkyLightCubemap && SkyLight->Cubemap)
        {
            Rec.SkyCubemapTextureId = SceneRTV2::Texture::Resolve(SkyLight->Cubemap, Ctx);
        }

        Rec.ExtraReflection = FJsonObjectConverter::UStructToJsonObject(
            SkyLight->GetClass(), SkyLight, 0, CPF_Transient | CPF_Deprecated);

        Ctx.Lights.Add(MoveTemp(Rec));
    }
}

namespace SceneRTV2::Decals
{
    void CollectDecal(UDecalComponent* DecalComp, FExportContext& Ctx)
    {
        if (!DecalComp) { return; }

        FDecalRecord Rec;
        Rec.Id = Identity::ForComponent(Ctx.SceneGuid, DecalComp);
        Rec.OwnerActorPath = DecalComp->GetOwner() ? DecalComp->GetOwner()->GetPathName() : FString();
        Rec.ComponentPath = DecalComp->GetPathName();
        Rec.Transform = DecalComp->GetComponentTransform();
        Rec.DecalSize = DecalComp->DecalSize;
        Rec.SortOrder = DecalComp->SortOrder;
        Rec.FadeScreenSize = DecalComp->FadeScreenSize;
        Rec.MaterialId = SceneRTV2::Material::Resolve(
            DecalComp->GetDecalMaterial(), nullptr, 0, Ctx);

        Ctx.Decals.Add(MoveTemp(Rec));
    }
}

namespace SceneRTV2::Cameras
{
    void CollectCamera(UCameraComponent* CameraComp, FExportContext& Ctx)
    {
        if (!CameraComp) { return; }

        FCameraRecord Rec;
        Rec.Id = Identity::ForComponent(Ctx.SceneGuid, CameraComp);
        Rec.OwnerActorPath = CameraComp->GetOwner() ? CameraComp->GetOwner()->GetPathName() : FString();
        Rec.Transform = CameraComp->GetComponentTransform();
        Rec.FieldOfView = CameraComp->FieldOfView;
        Rec.AspectRatio = CameraComp->AspectRatio;
        Rec.PostProcess = SceneRTV2::PostProcess::ReflectPostProcessSettings(CameraComp->PostProcessSettings);

        Ctx.Cameras.Add(MoveTemp(Rec));
    }
}

namespace SceneRTV2::PostProcess
{
    TSharedPtr<FJsonObject> ReflectPostProcessSettings(const FPostProcessSettings& Settings)
    {
        return FJsonObjectConverter::UStructToJsonObject(
            FPostProcessSettings::StaticStruct(), &Settings,
            /*CheckFlags*/ 0, /*SkipFlags*/ CPF_Transient | CPF_Deprecated);
    }

    void CollectPostProcessComponent(UPostProcessComponent* Comp, FExportContext& Ctx)
    {
        if (!Comp) { return; }
        FPostProcessRecord Rec;
        Rec.Id = Identity::ForComponent(Ctx.SceneGuid, Comp);
        Rec.OwnerActorPath = Comp->GetOwner() ? Comp->GetOwner()->GetPathName() : FString();
        Rec.Priority = Comp->Priority;
        Rec.BlendRadius = Comp->BlendRadius;
        Rec.BlendWeight = Comp->BlendWeight;
        Rec.bUnbound = Comp->bUnbound;
        Rec.Settings = ReflectPostProcessSettings(Comp->Settings);
        Ctx.PostProcess.Add(MoveTemp(Rec));
    }

    void CollectPostProcessVolume(APostProcessVolume* Volume, FExportContext& Ctx)
    {
        if (!Volume) { return; }
        FPostProcessRecord Rec;
        Rec.Id = Identity::ForActor(Ctx.SceneGuid, Volume);
        Rec.OwnerActorPath = Volume->GetPathName();
        Rec.Priority = Volume->Priority;
        Rec.BlendRadius = Volume->BlendRadius;
        Rec.BlendWeight = Volume->BlendWeight;
        Rec.bUnbound = Volume->bUnbound;
        Rec.Bounds = Volume->GetComponentsBoundingBox(true);
        Rec.Settings = ReflectPostProcessSettings(Volume->Settings);
        Ctx.PostProcess.Add(MoveTemp(Rec));
    }
}

namespace SceneRTV2::Atmosphere
{
    static TSharedPtr<FJsonObject> ReflectComponent(UActorComponent* Comp)
    {
        if (!Comp) { return MakeShared<FJsonObject>(); }
        return FJsonObjectConverter::UStructToJsonObject(
            Comp->GetClass(), Comp, /*CheckFlags*/ 0,
            /*SkipFlags*/ CPF_Transient | CPF_Deprecated);
    }

    void CollectSkyAtmosphere(USkyAtmosphereComponent* Comp, FExportContext& Ctx)
    {
        if (!Comp) { return; }
        FAtmosphereRecord Rec;
        Rec.Kind = TEXT("SkyAtmosphere");
        Rec.OwnerActorPath = Comp->GetOwner() ? Comp->GetOwner()->GetPathName() : FString();
        Rec.Settings = ReflectComponent(Comp);
        Ctx.Atmosphere.Add(MoveTemp(Rec));
    }

    void CollectHeightFog(UExponentialHeightFogComponent* Comp, FExportContext& Ctx)
    {
        if (!Comp) { return; }
        FAtmosphereRecord Rec;
        Rec.Kind = TEXT("ExponentialHeightFog");
        Rec.OwnerActorPath = Comp->GetOwner() ? Comp->GetOwner()->GetPathName() : FString();
        Rec.Settings = ReflectComponent(Comp);
        Ctx.Atmosphere.Add(MoveTemp(Rec));
    }

    void CollectVolumetricCloud(UVolumetricCloudComponent* Comp, FExportContext& Ctx)
    {
        if (!Comp) { return; }
        FAtmosphereRecord Rec;
        Rec.Kind = TEXT("VolumetricCloud");
        Rec.OwnerActorPath = Comp->GetOwner() ? Comp->GetOwner()->GetPathName() : FString();
        Rec.Settings = ReflectComponent(Comp);
        Ctx.Atmosphere.Add(MoveTemp(Rec));
    }
}
