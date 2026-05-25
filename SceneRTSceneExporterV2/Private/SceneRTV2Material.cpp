#include "SceneRTV2Material.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"
#include "SceneRTV2Texture.h"

#include "Materials/MaterialInterface.h"
#include "Materials/MaterialInstance.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Materials/Material.h"
#include "MaterialTypes.h"
#include "Engine/Texture.h"
#include "Components/PrimitiveComponent.h"
#include "Components/MeshComponent.h"

namespace SceneRTV2::Material
{
    namespace
    {
        /**
         * Walks every (Association, LayerIndex) combination supported by 5.2.
         * V1 only handled GlobalParameter — the root cause of "param resolved
         * empty" reports. Material Layers / Blends store their parameters under
         * separate associations.
         */
        void GatherParameterInfos(UMaterialInterface* Material,
                                  TArray<FMaterialParameterInfo>& OutInfos,
                                  TArray<FGuid>& OutGuids,
                                  EMaterialParameterType Type)
        {
            if (!Material) { return; }

            // Global params.
            Material->GetAllParameterInfoOfType(Type, OutInfos, OutGuids);

            // Layered params (Material Layers / Blends).
            TArray<FMaterialParameterInfo> LayerInfos;
            TArray<FGuid> LayerGuids;
            Material->GetAllParametersOfType(Type, LayerInfos, LayerGuids);
            for (int32 i = 0; i < LayerInfos.Num(); ++i)
            {
                // GetAllParametersOfType returns ALL associations including Global; dedup.
                if (!OutInfos.Contains(LayerInfos[i]))
                {
                    OutInfos.Add(LayerInfos[i]);
                    OutGuids.Add(LayerGuids[i]);
                }
            }
        }

        void CapturePbrFlags(UMaterialInterface* Material, FMaterialRecord& Out)
        {
            if (!Material) { return; }
            Out.bTwoSided = Material->IsTwoSided();

            const UMaterial* Base = Material->GetMaterial();
            if (!Base) { return; }
            switch (Base->MaterialDomain)
            {
                case MD_Surface:       Out.Domain = SceneRTSceneExporterV2::EMaterialDomain::Surface;       break;
                case MD_DeferredDecal: Out.Domain = SceneRTSceneExporterV2::EMaterialDomain::Decal;         break;
                case MD_LightFunction: Out.Domain = SceneRTSceneExporterV2::EMaterialDomain::LightFunction; break;
                case MD_Volume:        Out.Domain = SceneRTSceneExporterV2::EMaterialDomain::Volume;        break;
                case MD_UI:            Out.Domain = SceneRTSceneExporterV2::EMaterialDomain::UI;            break;
                case MD_PostProcess:   Out.Domain = SceneRTSceneExporterV2::EMaterialDomain::PostProcess;   break;
                default: break;
            }
            // ShadingModel — handle single + ambiguous (FromMaterialExpression).
            const EMaterialShadingModel SM = Base->GetShadingModels().GetFirstShadingModel();
            Out.ShadingModel = static_cast<SceneRTSceneExporterV2::EShadingModel>(static_cast<uint8>(SM));
        }
    }

    FStableId Resolve(UMaterialInterface* MaterialInput,
                      UPrimitiveComponent* OwnerComponent,
                      int32 SlotIndex,
                      FExportContext& Ctx)
    {
        FStableId Empty;
        if (!MaterialInput) { return Empty; }

        // Per-component MI override: must be respected so two actors using the same
        // master can produce different records.
        UMaterialInterface* Material = MaterialInput;
        if (UMeshComponent* MeshComp = Cast<UMeshComponent>(OwnerComponent))
        {
            if (MeshComp->OverrideMaterials.IsValidIndex(SlotIndex) && MeshComp->OverrideMaterials[SlotIndex])
            {
                Material = MeshComp->OverrideMaterials[SlotIndex];
            }
        }

        FMaterialRecord Rec;
        Rec.MaterialInstancePath = Material->GetPathName();
        if (UMaterialInstance* MI = Cast<UMaterialInstance>(Material))
        {
            Rec.ParentMaterialPath = MI->Parent ? MI->Parent->GetPathName() : FString();
        }
        else
        {
            Rec.ParentMaterialPath = Material->GetPathName();
        }
        CapturePbrFlags(Material, Rec);

        // ---- Texture parameters
        {
            TArray<FMaterialParameterInfo> Infos;
            TArray<FGuid> Guids;
            GatherParameterInfos(Material, Infos, Guids, EMaterialParameterType::Texture);
            for (const FMaterialParameterInfo& Info : Infos)
            {
                UTexture* Texture = nullptr;
                if (!Material->GetTextureParameterValue(Info, Texture)) { continue; }
                if (!Texture) { continue; }

                FMaterialParamRecord P;
                P.Name = Info.Name.ToString();
                P.Kind = EParamKind::Texture;
                P.ParameterAssociation = static_cast<uint8>(Info.Association);
                P.LayerIndex = Info.Index;
                P.TextureId = SceneRTV2::Texture::Resolve(Texture, Ctx);
                Rec.Params.Add(MoveTemp(P));
            }
        }
        // ---- Scalar parameters
        {
            TArray<FMaterialParameterInfo> Infos;
            TArray<FGuid> Guids;
            GatherParameterInfos(Material, Infos, Guids, EMaterialParameterType::Scalar);
            for (const FMaterialParameterInfo& Info : Infos)
            {
                float Value = 0.f;
                if (!Material->GetScalarParameterValue(Info, Value)) { continue; }
                FMaterialParamRecord P;
                P.Name = Info.Name.ToString();
                P.Kind = EParamKind::Scalar;
                P.Scalar = Value;
                P.ParameterAssociation = static_cast<uint8>(Info.Association);
                P.LayerIndex = Info.Index;
                Rec.Params.Add(MoveTemp(P));
            }
        }
        // ---- Vector parameters
        {
            TArray<FMaterialParameterInfo> Infos;
            TArray<FGuid> Guids;
            GatherParameterInfos(Material, Infos, Guids, EMaterialParameterType::Vector);
            for (const FMaterialParameterInfo& Info : Infos)
            {
                FLinearColor Value = FLinearColor::Black;
                if (!Material->GetVectorParameterValue(Info, Value)) { continue; }
                FMaterialParamRecord P;
                P.Name = Info.Name.ToString();
                P.Kind = EParamKind::Vector;
                P.Vector = Value;
                P.ParameterAssociation = static_cast<uint8>(Info.Association);
                P.LayerIndex = Info.Index;
                Rec.Params.Add(MoveTemp(P));
            }
        }
        // ---- Static switch parameters
        {
            TArray<FMaterialParameterInfo> Infos;
            TArray<FGuid> Guids;
            GatherParameterInfos(Material, Infos, Guids, EMaterialParameterType::StaticSwitch);
            for (const FMaterialParameterInfo& Info : Infos)
            {
                bool Value = false;
                FGuid OutGuid;
                if (!Material->GetStaticSwitchParameterValue(Info, Value, OutGuid)) { continue; }
                FMaterialParamRecord P;
                P.Name = Info.Name.ToString();
                P.Kind = EParamKind::Switch;
                P.bSwitch = Value;
                P.ParameterAssociation = static_cast<uint8>(Info.Association);
                P.LayerIndex = Info.Index;
                Rec.Params.Add(MoveTemp(P));
            }
        }

        // Sort by (Name, Association, LayerIndex) for deterministic hashing.
        Rec.Params.Sort([](const FMaterialParamRecord& A, const FMaterialParamRecord& B)
        {
            if (A.Name != B.Name) return A.Name < B.Name;
            if (A.ParameterAssociation != B.ParameterAssociation) return A.ParameterAssociation < B.ParameterAssociation;
            return A.LayerIndex < B.LayerIndex;
        });

        Rec.ResolvedParamsHash = ComputeResolvedHash(Rec);

        // Dedup.
        if (int32* Existing = Ctx.MaterialByHash.Find(Rec.ResolvedParamsHash))
        {
            return Ctx.Materials[*Existing].Id;
        }

        Rec.Id = Identity::ForMaterial(Ctx.SceneGuid, Material, Rec.ResolvedParamsHash);
        const int32 NewIndex = Ctx.Materials.Add(MoveTemp(Rec));
        Ctx.MaterialByHash.Add(Ctx.Materials[NewIndex].ResolvedParamsHash, NewIndex);
        return Ctx.Materials[NewIndex].Id;
    }

    uint64 ComputeResolvedHash(const FMaterialRecord& Rec)
    {
        uint64 H = Identity::HashString(Rec.ParentMaterialPath, 0xC0FFEEULL);
        H = Identity::HashString(FString::Printf(TEXT("d%d/sm%d/ts%d"),
            int32(Rec.Domain), int32(Rec.ShadingModel), Rec.bTwoSided ? 1 : 0), H);
        for (const FMaterialParamRecord& P : Rec.Params)
        {
            H = Identity::HashString(P.Name, H);
            const uint8 Tag = static_cast<uint8>(P.Kind);
            H = Identity::HashBytes(&Tag, 1, H);
            H = Identity::HashBytes(&P.ParameterAssociation, 1, H);
            H = Identity::HashBytes(&P.LayerIndex, sizeof(P.LayerIndex), H);
            switch (P.Kind)
            {
                case EParamKind::Scalar:  H = Identity::HashBytes(&P.Scalar, sizeof(P.Scalar), H); break;
                case EParamKind::Vector:  H = Identity::HashBytes(&P.Vector, sizeof(P.Vector), H); break;
                case EParamKind::Texture: H = Identity::HashString(P.TextureId.Value, H); break;
                case EParamKind::Switch:  { uint8 V = P.bSwitch ? 1 : 0; H = Identity::HashBytes(&V, 1, H); } break;
                default: break;
            }
        }
        return H;
    }

    FString ClassifyMasterMaterial(const UMaterialInterface* Material)
    {
        // No scene-specific names: classify by ShadingModel + Domain + parent name.
        if (!Material) { return TEXT("Unknown"); }
        const UMaterial* Base = Material->GetMaterial();
        if (!Base) { return TEXT("Unknown"); }
        if (Base->MaterialDomain == MD_DeferredDecal) { return TEXT("Decal"); }
        if (Base->MaterialDomain == MD_PostProcess)   { return TEXT("PostProcess"); }
        if (Base->MaterialDomain == MD_LightFunction) { return TEXT("LightFunction"); }
        if (Base->MaterialDomain == MD_Volume)        { return TEXT("Volume"); }
        // Surface — distinguish unlit / lit / translucent.
        if (Base->GetShadingModels().HasShadingModel(MSM_Unlit)) { return TEXT("SurfaceUnlit"); }
        if (Base->BlendMode == BLEND_Translucent || Base->BlendMode == BLEND_Additive)
        {
            return TEXT("SurfaceTranslucent");
        }
        return TEXT("SurfaceOpaque");
    }
}
