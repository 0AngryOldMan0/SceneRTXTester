#include "SceneRTV2Mesh.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"
#include "SceneRTV2Material.h"

#include "Components/StaticMeshComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/SkeletalMesh.h"
#include "StaticMeshResources.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "Rendering/SkinWeightVertexBuffer.h"

#if WITH_EDITORONLY_DATA
#include "MeshDescription.h"
#include "StaticMeshAttributes.h"
#endif

namespace SceneRTV2::Mesh
{
    namespace
    {
        /**
         * Reads geometry from a UStaticMesh LOD. If the mesh disallows CPU
         * access, we temporarily flip bAllowCPUAccess via DerivedDataCache fetch
         * (5.2 supports this via FStaticMeshOperations).
         *
         * For Nanite-only meshes (LODResources empty), falls back to FMeshDescription
         * source data (editor builds only); sets Out.bNaniteFallback = true.
         */
#if WITH_EDITORONLY_DATA
        static bool ReadStaticMeshLodFromMeshDescription(UStaticMesh* SrcMesh, FMeshLod& Out)
        {
            const FMeshDescription* MD = SrcMesh->GetMeshDescription(0);
            if (!MD) { return false; }

            FStaticMeshAttributes Attribs(const_cast<FMeshDescription&>(*MD));

            auto Positions  = Attribs.GetVertexPositions();
            auto Normals    = Attribs.GetVertexInstanceNormals();
            auto Tangents   = Attribs.GetVertexInstanceTangents();
            auto BinorSigns = Attribs.GetVertexInstanceBinormalSigns();
            auto UVChannels = Attribs.GetVertexInstanceUVs();

            const int32 NumVIs     = MD->VertexInstances().GetArraySize();
            const int32 NumUVChans = FMath::Max((int32)UVChannels.GetNumChannels(), 1);

            Out.LodIndex        = 0;
            Out.bNaniteFallback = true;
            Out.Positions.SetNum(NumVIs);
            Out.Normals.SetNum(NumVIs);
            Out.Tangents.SetNum(NumVIs);
            Out.UVs.SetNum(NumUVChans);
            for (auto& Ch : Out.UVs) { Ch.SetNum(NumVIs); }

            for (FVertexInstanceID ViID : MD->VertexInstances().GetElementIDs())
            {
                const int32 i    = ViID.GetValue();
                const FVertexID V = MD->GetVertexInstanceVertex(ViID);
                Out.Positions[i]  = Positions[V];
                Out.Normals[i]    = Normals[ViID];
                Out.Tangents[i]   = FVector4f(Tangents[ViID], BinorSigns[ViID]);
                for (int32 c = 0; c < (int32)UVChannels.GetNumChannels(); ++c)
                    Out.UVs[c][i] = UVChannels.Get(ViID, c);
            }

            // Build flat index buffer and per-polygon-group sections.
            for (FPolygonGroupID PgID : MD->PolygonGroups().GetElementIDs())
            {
                const uint32 FirstIdx = (uint32)Out.Indices.Num();
                for (FPolygonID PolyID : MD->GetPolygonGroupPolygons(PgID))
                {
                    for (FTriangleID TriID : MD->GetPolygonTriangleIDs(PolyID))
                    {
                        TArrayView<const FVertexInstanceID> VIs = MD->GetTriangleVertexInstances(TriID);
                        Out.Indices.Add((uint32)VIs[0].GetValue());
                        Out.Indices.Add((uint32)VIs[1].GetValue());
                        Out.Indices.Add((uint32)VIs[2].GetValue());
                    }
                }
                const uint32 Count = (uint32)Out.Indices.Num() - FirstIdx;
                if (Count == 0) { continue; }
                FMeshSection MS;
                MS.SectionIndex      = Out.Sections.Num();
                MS.MaterialSlotIndex = PgID.GetValue();
                MS.FirstIndex        = FirstIdx;
                MS.IndexCount        = Count;
                Out.Sections.Add(MoveTemp(MS));
            }

            return Out.Indices.Num() > 0;
        }
#endif // WITH_EDITORONLY_DATA

        bool ReadStaticMeshLod(UStaticMesh* SrcMesh, int32 LodIndex, FMeshLod& Out)
        {
            if (!SrcMesh || !SrcMesh->GetRenderData()) { return false; }
            const FStaticMeshRenderData& RD = *SrcMesh->GetRenderData();
            if (!RD.LODResources.IsValidIndex(LodIndex))
            {
#if WITH_EDITORONLY_DATA
                if (LodIndex == 0 && SrcMesh->NaniteSettings.bEnabled)
                {
                    return ReadStaticMeshLodFromMeshDescription(SrcMesh, Out);
                }
#endif
                return false;
            }
            const FStaticMeshLODResources& Src = RD.LODResources[LodIndex];

            const int32 NumVerts = Src.VertexBuffers.PositionVertexBuffer.GetNumVertices();
            const int32 NumUVChans = Src.VertexBuffers.StaticMeshVertexBuffer.GetNumTexCoords();
            Out.LodIndex = LodIndex;
            Out.Positions.SetNumUninitialized(NumVerts);
            Out.Normals.SetNumUninitialized(NumVerts);
            Out.Tangents.SetNumUninitialized(NumVerts);
            Out.UVs.SetNum(FMath::Max(NumUVChans, 1));
            for (int32 c = 0; c < Out.UVs.Num(); ++c) { Out.UVs[c].SetNumUninitialized(NumVerts); }

            const bool bHasColors = Src.VertexBuffers.ColorVertexBuffer.GetNumVertices() == NumVerts;
            if (bHasColors)
            {
                Out.Colors.SetNumUninitialized(NumVerts);
                Out.bHasVertexColors = true;
            }

            for (int32 v = 0; v < NumVerts; ++v)
            {
                Out.Positions[v] = Src.VertexBuffers.PositionVertexBuffer.VertexPosition(v);
                Out.Normals[v]   = Src.VertexBuffers.StaticMeshVertexBuffer.VertexTangentZ(v).ToFVector3f();
                FVector4f T(Src.VertexBuffers.StaticMeshVertexBuffer.VertexTangentX(v).ToFVector3f(), 1.f);
                Out.Tangents[v]  = T;
                for (int32 c = 0; c < NumUVChans; ++c)
                {
                    Out.UVs[c][v] = Src.VertexBuffers.StaticMeshVertexBuffer.GetVertexUV(v, c);
                }
                if (bHasColors)
                {
                    Out.Colors[v] = Src.VertexBuffers.ColorVertexBuffer.VertexColor(v);
                }
            }

            TArray<uint32> Indices;
            Src.IndexBuffer.GetCopy(Indices);
            Out.Indices = MoveTemp(Indices);

            for (int32 s = 0; s < Src.Sections.Num(); ++s)
            {
                const FStaticMeshSection& In = Src.Sections[s];
                FMeshSection MS;
                MS.SectionIndex = s;
                MS.MaterialSlotIndex = In.MaterialIndex;
                MS.FirstIndex = In.FirstIndex;
                MS.IndexCount = In.NumTriangles * 3;
                MS.MinVertex = In.MinVertexIndex;
                MS.MaxVertex = In.MaxVertexIndex;
                Out.Sections.Add(MoveTemp(MS));
            }
            return true;
        }
    }

    int32 CollectStaticMesh(UStaticMeshComponent* Component, FExportContext& Ctx)
    {
        if (!Component || !Component->GetStaticMesh()) { return INDEX_NONE; }
        UStaticMesh* SrcMesh = Component->GetStaticMesh();

        const FString AssetKey = SrcMesh->GetPathName();
        if (int32* Existing = Ctx.MeshAssetByKey.Find(AssetKey))
        {
            // Mesh already collected — only emit a new primitive instance.
            FPrimitiveInstance Prim;
            Prim.Id = Identity::ForComponent(Ctx.SceneGuid, Component);
            Prim.MeshId = Ctx.Meshes[*Existing].Id;
            Prim.OwnerActorPath = Component->GetOwner() ? Component->GetOwner()->GetPathName() : FString();
            Prim.ComponentPath = Component->GetPathName();
            Prim.Transform = Component->GetComponentTransform();
            for (int32 s = 0; s < Component->GetNumMaterials(); ++s)
            {
                Prim.OverrideMaterialIds.Add(
                    SceneRTV2::Material::Resolve(Component->GetMaterial(s), Component, s, Ctx));
            }
            Ctx.Primitives.Add(MoveTemp(Prim));
            return *Existing;
        }

        FMeshAsset Asset;
        Asset.Kind = SceneRTSceneExporterV2::EMeshSourceKind::StaticMesh;
        Asset.SourceAssetPath = AssetKey;
        Asset.DisplayName = SrcMesh->GetName();
        Asset.Id = Identity::ForMesh(Ctx.SceneGuid, SrcMesh);
        Asset.LocalBounds = SrcMesh->GetBoundingBox();

        const int32 NumLods = Ctx.Settings->bExportAllLODs
            ? SrcMesh->GetRenderData()->LODResources.Num()
            : 1;

        // Nanite-only meshes have no CPU LODResources. Gate on bExportNaniteFallback
        // (uses FMeshDescription source data, editor-only) or skip with warning.
        const bool bNaniteOnly = SrcMesh->NaniteSettings.bEnabled
                                 && SrcMesh->GetRenderData()->LODResources.Num() == 0;
        if (bNaniteOnly && Ctx.Settings && !Ctx.Settings->bExportNaniteFallback)
        {
            Ctx.AddIssue(TEXT("warn"), TEXT("mesh"),
                FString::Printf(TEXT("Nanite mesh skipped (bExportNaniteFallback=false): %s"), *AssetKey),
                Asset.Id.Value);
        }
        else
        {
            // For Nanite-only meshes always attempt LOD 0; otherwise respect NumLods.
            const int32 LodCount = (bNaniteOnly) ? 1 : NumLods;
            for (int32 i = 0; i < LodCount; ++i)
            {
                FMeshLod Lod;
                if (!ReadStaticMeshLod(SrcMesh, i, Lod))
                {
                    Ctx.AddIssue(TEXT("warn"), TEXT("mesh"),
                        FString::Printf(TEXT("Cannot read LOD %d for %s"), i, *AssetKey),
                        Asset.Id.Value);
                    continue;
                }
                if (Lod.bNaniteFallback)
                {
                    Asset.Kind = SceneRTSceneExporterV2::EMeshSourceKind::NaniteFallback;
                }
                Asset.Lods.Add(MoveTemp(Lod));
            }
        }

        for (int32 s = 0; s < SrcMesh->GetStaticMaterials().Num(); ++s)
        {
            Asset.SlotMaterialIds.Add(
                SceneRTV2::Material::Resolve(SrcMesh->GetMaterial(s), Component, s, Ctx));
        }

        const int32 Index = Ctx.Meshes.Add(MoveTemp(Asset));
        Ctx.MeshAssetByKey.Add(AssetKey, Index);

        // Now emit the primitive instance pointing at this mesh.
        FPrimitiveInstance Prim;
        Prim.Id = Identity::ForComponent(Ctx.SceneGuid, Component);
        Prim.MeshId = Ctx.Meshes[Index].Id;
        Prim.OwnerActorPath = Component->GetOwner() ? Component->GetOwner()->GetPathName() : FString();
        Prim.ComponentPath = Component->GetPathName();
        Prim.Transform = Component->GetComponentTransform();
        for (int32 s = 0; s < Component->GetNumMaterials(); ++s)
        {
            Prim.OverrideMaterialIds.Add(
                SceneRTV2::Material::Resolve(Component->GetMaterial(s), Component, s, Ctx));
        }
        Ctx.Primitives.Add(MoveTemp(Prim));

        return Index;
    }

    int32 CollectSkeletalMesh(USkeletalMeshComponent* Component, FExportContext& Ctx)
    {
        if (!Component) { return INDEX_NONE; }
        USkeletalMesh* SkelMesh = Component->GetSkeletalMeshAsset();
        if (!SkelMesh) { return INDEX_NONE; }

        const FString AssetKey = SkelMesh->GetPathName();
        if (int32* Existing = Ctx.MeshAssetByKey.Find(AssetKey))
        {
            FPrimitiveInstance Prim;
            Prim.Id = Identity::ForComponent(Ctx.SceneGuid, Component);
            Prim.MeshId = Ctx.Meshes[*Existing].Id;
            Prim.OwnerActorPath = Component->GetOwner() ? Component->GetOwner()->GetPathName() : FString();
            Prim.ComponentPath = Component->GetPathName();
            Prim.Transform = Component->GetComponentTransform();
            Prim.bVisible = Component->IsVisible();
            Prim.bCastShadow = Component->CastShadow;
            Prim.bAffectsRayTracing = Component->bVisibleInRayTracing;
            for (int32 s = 0; s < Component->GetNumMaterials(); ++s)
            {
                Prim.OverrideMaterialIds.Add(
                    SceneRTV2::Material::Resolve(Component->GetMaterial(s), Component, s, Ctx));
            }
            Ctx.Primitives.Add(MoveTemp(Prim));
            return *Existing;
        }

        FSkeletalMeshRenderData* RenderData = SkelMesh->GetResourceForRendering();
        if (!RenderData || RenderData->LODRenderData.IsEmpty())
        {
            Ctx.AddIssue(TEXT("warn"), TEXT("skeletal_mesh"),
                FString::Printf(TEXT("No render data for %s"), *AssetKey),
                Identity::ForMesh(Ctx.SceneGuid, SkelMesh).Value);
            return INDEX_NONE;
        }

        FMeshAsset Asset;
        Asset.Kind = SceneRTSceneExporterV2::EMeshSourceKind::SkeletalMesh;
        Asset.SourceAssetPath = AssetKey;
        Asset.DisplayName = SkelMesh->GetName();
        Asset.Id = Identity::ForMesh(Ctx.SceneGuid, SkelMesh);
        Asset.LocalBounds = SkelMesh->GetBounds().GetBox();

        const int32 NumLods = RenderData->LODRenderData.Num();
        for (int32 li = 0; li < NumLods; ++li)
        {
            const FSkeletalMeshLODRenderData& LODData = RenderData->LODRenderData[li];
            const uint32 NumVerts = LODData.StaticVertexBuffers.PositionVertexBuffer.GetNumVertices();
            if (NumVerts == 0) { continue; }

            FMeshLod Lod;
            Lod.LodIndex = li;

            // Positions
            Lod.Positions.SetNumUninitialized(NumVerts);
            for (uint32 v = 0; v < NumVerts; ++v)
            {
                Lod.Positions[v] = LODData.StaticVertexBuffers.PositionVertexBuffer.VertexPosition(v);
            }

            // Normals & tangents
            Lod.Normals.SetNumUninitialized(NumVerts);
            Lod.Tangents.SetNumUninitialized(NumVerts);
            for (uint32 v = 0; v < NumVerts; ++v)
            {
                Lod.Normals[v]  = LODData.StaticVertexBuffers.StaticMeshVertexBuffer.VertexTangentZ(v).ToFVector3f();
                Lod.Tangents[v] = LODData.StaticVertexBuffers.StaticMeshVertexBuffer.VertexTangentX(v);
            }

            // UVs
            const int32 NumUVChans = LODData.StaticVertexBuffers.StaticMeshVertexBuffer.GetNumTexCoords();
            Lod.UVs.SetNum(FMath::Max(NumUVChans, 1));
            for (int32 c = 0; c < Lod.UVs.Num(); ++c)
            {
                Lod.UVs[c].SetNumUninitialized(NumVerts);
                for (uint32 v = 0; v < NumVerts; ++v)
                {
                    Lod.UVs[c][v] = (c < NumUVChans)
                        ? LODData.StaticVertexBuffers.StaticMeshVertexBuffer.GetVertexUV(v, c)
                        : FVector2f::ZeroVector;
                }
            }

            // Vertex colors
            const FColorVertexBuffer& ColorBuf = LODData.StaticVertexBuffers.ColorVertexBuffer;
            Lod.bHasVertexColors = ColorBuf.GetNumVertices() == NumVerts;
            if (Lod.bHasVertexColors)
            {
                Lod.Colors.SetNumUninitialized(NumVerts);
                for (uint32 v = 0; v < NumVerts; ++v) { Lod.Colors[v] = ColorBuf.VertexColor(v); }
            }

            // Indices
            LODData.MultiSizeIndexContainer.GetIndexBuffer(Lod.Indices);

            // Sections
            for (int32 si = 0; si < LODData.RenderSections.Num(); ++si)
            {
                const FSkelMeshRenderSection& Sec = LODData.RenderSections[si];
                FMeshSection MS;
                MS.SectionIndex       = si;
                MS.MaterialSlotIndex  = Sec.MaterialIndex;
                MS.FirstIndex         = Sec.BaseIndex;
                MS.IndexCount         = Sec.NumTriangles * 3;
                MS.MinVertex          = Sec.BaseVertexIndex;
                MS.MaxVertex          = Sec.BaseVertexIndex + Sec.NumVertices > 0
                                            ? Sec.BaseVertexIndex + Sec.NumVertices - 1 : 0;
                MS.bCastShadow        = Sec.bCastShadow;
                Lod.Sections.Add(MoveTemp(MS));
            }

            // Skin weights (clamped to 4 influences)
            const FSkinWeightVertexBuffer& SkinBuf = LODData.SkinWeightVertexBuffer;
            if (SkinBuf.GetNumVertices() == NumVerts)
            {
                const int32 NumInfluences = FMath::Min(SkinBuf.GetMaxBoneInfluences(), 4);
                Lod.SkinIndices.SetNumZeroed(NumVerts);
                Lod.SkinWeights.SetNumZeroed(NumVerts);
                for (uint32 v = 0; v < NumVerts; ++v)
                {
                    FVector4f& Idx = Lod.SkinIndices[v];
                    FVector4f& Wgt = Lod.SkinWeights[v];
                    for (int32 inf = 0; inf < NumInfluences; ++inf)
                    {
                        const float BoneIndex = static_cast<float>(SkinBuf.GetBoneIndex(v, inf));
                        const float Weight    = SkinBuf.GetBoneWeight(v, inf) / 255.f;
                        switch (inf)
                        {
                        case 0: Idx.X = BoneIndex; Wgt.X = Weight; break;
                        case 1: Idx.Y = BoneIndex; Wgt.Y = Weight; break;
                        case 2: Idx.Z = BoneIndex; Wgt.Z = Weight; break;
                        case 3: Idx.W = BoneIndex; Wgt.W = Weight; break;
                        default: break;
                        }
                    }
                }
            }

            Asset.Lods.Add(MoveTemp(Lod));
        }

        // Bind pose from FReferenceSkeleton
        {
            const FReferenceSkeleton& RefSkel = SkelMesh->GetRefSkeleton();
            const int32 NumBones = RefSkel.GetNum();
            const TArray<FTransform>& BindPose = RefSkel.GetRefBonePose();
            Asset.Skeleton.BoneNames.SetNumUninitialized(NumBones);
            Asset.Skeleton.ParentIndices.SetNumUninitialized(NumBones);
            Asset.Skeleton.LocalBindPose.SetNum(NumBones);
            for (int32 b = 0; b < NumBones; ++b)
            {
                Asset.Skeleton.BoneNames[b]     = RefSkel.GetBoneName(b).ToString();
                Asset.Skeleton.ParentIndices[b]  = RefSkel.GetParentIndex(b);
                Asset.Skeleton.LocalBindPose[b]  = BindPose[b];
            }
        }

        // Material slots
        const TArray<FSkeletalMaterial>& Mats = SkelMesh->GetMaterials();
        for (int32 s = 0; s < Mats.Num(); ++s)
        {
            Asset.SlotMaterialIds.Add(
                SceneRTV2::Material::Resolve(Mats[s].MaterialInterface, Component, s, Ctx));
        }

        const int32 Index = Ctx.Meshes.Add(MoveTemp(Asset));
        Ctx.MeshAssetByKey.Add(AssetKey, Index);

        FPrimitiveInstance Prim;
        Prim.Id = Identity::ForComponent(Ctx.SceneGuid, Component);
        Prim.MeshId = Ctx.Meshes[Index].Id;
        Prim.OwnerActorPath = Component->GetOwner() ? Component->GetOwner()->GetPathName() : FString();
        Prim.ComponentPath = Component->GetPathName();
        Prim.Transform = Component->GetComponentTransform();
        Prim.bVisible = Component->IsVisible();
        Prim.bCastShadow = Component->CastShadow;
        Prim.bAffectsRayTracing = Component->bVisibleInRayTracing;
        for (int32 s = 0; s < Component->GetNumMaterials(); ++s)
        {
            Prim.OverrideMaterialIds.Add(
                SceneRTV2::Material::Resolve(Component->GetMaterial(s), Component, s, Ctx));
        }
        Ctx.Primitives.Add(MoveTemp(Prim));
        return Index;
    }

    void CollectInstancedStatic(UInstancedStaticMeshComponent* Component, FExportContext& Ctx)
    {
        if (!Component || !Component->GetStaticMesh()) { return; }

        // Collect the underlying mesh once (reuses path if cached).
        const int32 MeshIdx = CollectStaticMesh(Component, Ctx);
        if (MeshIdx == INDEX_NONE) { return; }

        // Pop the auto-added primitive instance — instanced components produce
        // many transforms from a single component.
        if (Ctx.Primitives.Num() > 0 && Ctx.Primitives.Last().ComponentPath == Component->GetPathName())
        {
            Ctx.Primitives.Pop(EAllowShrinking::No);
        }

        const int32 NumInstances = Component->GetInstanceCount();
        for (int32 i = 0; i < NumInstances; ++i)
        {
            FTransform InstanceXform;
            Component->GetInstanceTransform(i, InstanceXform, /*bWorldSpace*/ true);

            FPrimitiveInstance Prim;
            Prim.Id = Identity::Make(Ctx.SceneGuid, TEXT("inst"),
                Component->GetPathName(), FString::Printf(TEXT("i%d"), i));
            Prim.MeshId = Ctx.Meshes[MeshIdx].Id;
            Prim.OwnerActorPath = Component->GetOwner() ? Component->GetOwner()->GetPathName() : FString();
            Prim.ComponentPath = Component->GetPathName();
            Prim.Transform = InstanceXform;
            for (int32 s = 0; s < Component->GetNumMaterials(); ++s)
            {
                Prim.OverrideMaterialIds.Add(
                    SceneRTV2::Material::Resolve(Component->GetMaterial(s), Component, s, Ctx));
            }
            Ctx.Primitives.Add(MoveTemp(Prim));
        }
    }
}
