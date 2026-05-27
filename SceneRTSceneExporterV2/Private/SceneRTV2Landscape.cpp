#include "SceneRTV2Landscape.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"
#include "SceneRTV2Material.h"
#include "SceneRTV2Texture.h"

#include "Landscape.h"
#include "LandscapeProxy.h"
#include "LandscapeComponent.h"
#include "LandscapeInfo.h"
#include "LandscapeLayerInfoObject.h"
#include "LandscapeEdit.h"

#include "MeshDescription.h"
#include "StaticMeshAttributes.h"

#include "Engine/Texture2D.h"
#include "Materials/MaterialInterface.h"

namespace SceneRTV2::Landscape
{
    bool ConvertMeshDescriptionToLod(const FMeshDescription& Mesh, FMeshLod& OutLod)
    {
        FStaticMeshConstAttributes Attrs(Mesh);
        const TVertexAttributesConstRef<FVector3f> VertexPositions = Attrs.GetVertexPositions();
        const TVertexInstanceAttributesConstRef<FVector3f> VtxNormals = Attrs.GetVertexInstanceNormals();
        const TVertexInstanceAttributesConstRef<FVector3f> VtxTangents = Attrs.GetVertexInstanceTangents();
        const TVertexInstanceAttributesConstRef<float> VtxBinormalSign = Attrs.GetVertexInstanceBinormalSigns();
        const TVertexInstanceAttributesConstRef<FVector4f> VtxColors = Attrs.GetVertexInstanceColors();
        const TVertexInstanceAttributesConstRef<FVector2f> VtxUVs = Attrs.GetVertexInstanceUVs();

        const int32 NumUVChans = FMath::Max(1, VtxUVs.GetNumChannels());
        const int32 NumVertexInstances = Mesh.VertexInstances().Num();
        if (NumVertexInstances == 0) { return false; }

        OutLod.LodIndex = 0;
        OutLod.Positions.Reset(NumVertexInstances);
        OutLod.Normals.Reset(NumVertexInstances);
        OutLod.Tangents.Reset(NumVertexInstances);
        OutLod.UVs.SetNum(NumUVChans);
        for (int32 c = 0; c < NumUVChans; ++c)
        {
            OutLod.UVs[c].Reset(NumVertexInstances);
        }
        OutLod.Indices.Reset();
        OutLod.Sections.Reset();
        OutLod.bHasVertexColors = VtxColors.IsValid() && VtxColors.GetNumChannels() > 0;
        if (OutLod.bHasVertexColors) { OutLod.Colors.Reset(NumVertexInstances); }

        // Each vertex instance becomes one exported vertex (landscape has no
        // shared corners across triangles in practice, and this preserves
        // per-corner normals).
        TMap<FVertexInstanceID, uint32> InstanceToVertex;
        InstanceToVertex.Reserve(NumVertexInstances);

        for (const FPolygonGroupID& GroupId : Mesh.PolygonGroups().GetElementIDs())
        {
            FMeshSection Section;
            Section.SectionIndex = OutLod.Sections.Num();
            Section.MaterialSlotIndex = GroupId.GetValue();
            Section.FirstIndex = OutLod.Indices.Num();
            Section.MinVertex = OutLod.Positions.Num();

            for (const FPolygonID& PolyId : Mesh.GetPolygonGroupPolygonIDs(GroupId))
            {
                const TArray<FTriangleID>& Tris = Mesh.GetPolygonTriangles(PolyId);
                for (const FTriangleID& TriId : Tris)
                {
                    const TArrayView<const FVertexInstanceID> Corners = Mesh.GetTriangleVertexInstances(TriId);
                    for (int32 Corner = 0; Corner < 3; ++Corner)
                    {
                        const FVertexInstanceID VI = Corners[Corner];
                        uint32* Existing = InstanceToVertex.Find(VI);
                        uint32 VertexIndex;
                        if (Existing)
                        {
                            VertexIndex = *Existing;
                        }
                        else
                        {
                            const FVertexID V = Mesh.GetVertexInstanceVertex(VI);
                            VertexIndex = OutLod.Positions.Add(VertexPositions[V]);
                            const FVector3f N = VtxNormals.IsValid() ? VtxNormals[VI] : FVector3f::ZAxisVector;
                            OutLod.Normals.Add(N);
                            const FVector3f T = VtxTangents.IsValid() ? VtxTangents[VI] : FVector3f::XAxisVector;
                            const float Sign = VtxBinormalSign.IsValid() ? VtxBinormalSign[VI] : 1.f;
                            OutLod.Tangents.Add(FVector4f(T.X, T.Y, T.Z, Sign));
                            for (int32 c = 0; c < NumUVChans; ++c)
                            {
                                const FVector2f UV = (c < VtxUVs.GetNumChannels())
                                    ? VtxUVs.Get(VI, c) : FVector2f::ZeroVector;
                                OutLod.UVs[c].Add(UV);
                            }
                            if (OutLod.bHasVertexColors)
                            {
                                const FVector4f C = VtxColors[VI];
                                OutLod.Colors.Add(FColor(
                                    FMath::Clamp(int32(C.X * 255.f), 0, 255),
                                    FMath::Clamp(int32(C.Y * 255.f), 0, 255),
                                    FMath::Clamp(int32(C.Z * 255.f), 0, 255),
                                    FMath::Clamp(int32(C.W * 255.f), 0, 255)));
                            }
                            InstanceToVertex.Add(VI, VertexIndex);
                        }
                        OutLod.Indices.Add(VertexIndex);
                    }
                }
            }

            Section.IndexCount = OutLod.Indices.Num() - Section.FirstIndex;
            Section.MaxVertex = OutLod.Positions.Num() > 0 ? OutLod.Positions.Num() - 1 : 0;
            if (Section.IndexCount > 0)
            {
                OutLod.Sections.Add(MoveTemp(Section));
            }
        }
        return OutLod.Indices.Num() > 0;
    }

    namespace
    {
        FStableId EmitComponentMeshAsset(ULandscapeComponent* Component, FExportContext& Ctx,
                                         FMeshLod&& Lod, UMaterialInterface* MaterialOverride,
                                         int32& OutMeshIndex)
        {
            FMeshAsset Asset;
            Asset.Kind = SceneRTSceneExporterV2::EMeshSourceKind::LandscapeTriangle;
            Asset.SourceAssetPath = Component->GetPathName();
            Asset.DisplayName = Component->GetName();
            Asset.Id = Identity::ForComponent(Ctx.SceneGuid, Component);

            FBox Bounds(ForceInitToZero);
            for (const FVector3f& P : Lod.Positions) { Bounds += FVector(P); }
            Asset.LocalBounds = Bounds;
            Asset.Lods.Add(MoveTemp(Lod));

            // Single slot, single material — landscape components use one MI per
            // component (auto-generated by the engine) layered by weightmap.
            UMaterialInterface* MatToUse = MaterialOverride
                ? MaterialOverride
                : Component->GetLandscapeMaterial();
            Asset.SlotMaterialIds.Add(
                SceneRTV2::Material::Resolve(MatToUse, /*Owner*/ nullptr, /*Slot*/ 0, Ctx));

            OutMeshIndex = Ctx.Meshes.Add(MoveTemp(Asset));
            return Ctx.Meshes[OutMeshIndex].Id;
        }

        void GatherComponentLayerWeights(ULandscapeComponent* Component,
                                         FLandscapeComponentInfo& Info,
                                         FExportContext& Ctx)
        {
            if (!Component) { return; }

            const TArray<FWeightmapLayerAllocationInfo>& Allocs = Component->GetWeightmapLayerAllocations();
            const TArray<TObjectPtr<UTexture2D>>& WeightmapTextures = Component->GetWeightmapTextures();

            for (const FWeightmapLayerAllocationInfo& A : Allocs)
            {
                if (!A.LayerInfo) { continue; }
                FLandscapeLayerWeight Lw;
                Lw.LayerName = A.LayerInfo->LayerName.ToString();
                Lw.ChannelIndex = A.WeightmapTextureChannel;
                if (WeightmapTextures.IsValidIndex(A.WeightmapTextureIndex) &&
                    WeightmapTextures[A.WeightmapTextureIndex])
                {
                    Lw.WeightTextureId = SceneRTV2::Texture::Resolve(
                        WeightmapTextures[A.WeightmapTextureIndex], Ctx);
                }
                Lw.PhysicalMaterialBaseColor = A.LayerInfo->LayerUsageDebugColor;
                Info.LayerWeights.Add(MoveTemp(Lw));
            }

            if (UTexture2D* Heightmap = Component->GetHeightmap())
            {
                Info.HeightmapTextureId = SceneRTV2::Texture::Resolve(Heightmap, Ctx);
            }
        }

        FLandscapeRecord& GetOrAddLandscapeRecord(ALandscapeProxy* Proxy, FExportContext& Ctx)
        {
            // Streaming proxies share LandscapeGuid; we want one logical record.
            FString GuidStr;
            if (ALandscape* L = Cast<ALandscape>(Proxy))
            {
                GuidStr = L->GetLandscapeGuid().ToString();
            }
            else
            {
                GuidStr = Proxy->GetLandscapeGuid().ToString();
            }
            if (int32* Existing = Ctx.LandscapeByGuid.Find(GuidStr))
            {
                return Ctx.Landscapes[*Existing];
            }
            FLandscapeRecord Rec;
            Rec.Id = Identity::ForActor(Ctx.SceneGuid, Proxy);
            Rec.ActorPath = Proxy->GetPathName();
            Rec.LandscapeGuid = GuidStr;
            Rec.Transform = Proxy->GetActorTransform();
            Rec.ComponentSizeQuads = Proxy->ComponentSizeQuads;
            Rec.SubsectionSizeQuads = Proxy->SubsectionSizeQuads;
            Rec.NumSubsections = Proxy->NumSubsections;
            Rec.ExportLOD = FMath::Max(0, Ctx.Settings ? Ctx.Settings->LandscapeExportLOD : 0);
            Rec.BaseMaterialId = SceneRTV2::Material::Resolve(
                Proxy->GetLandscapeMaterial(), nullptr, 0, Ctx);
            Rec.HoleMaterialId = SceneRTV2::Material::Resolve(
                Proxy->GetLandscapeHoleMaterial(), nullptr, 0, Ctx);
            const int32 Idx = Ctx.Landscapes.Add(MoveTemp(Rec));
            Ctx.LandscapeByGuid.Add(GuidStr, Idx);
            return Ctx.Landscapes[Idx];
        }
    }

    void CollectProxyLike(ALandscapeProxy* Proxy, FExportContext& Ctx)
    {
        if (!Proxy) { return; }
        if (!Ctx.Settings || !Ctx.Settings->bExportLandscape) { return; }

        FLandscapeRecord& Record = GetOrAddLandscapeRecord(Proxy, Ctx);
        const int32 ExportLod = Record.ExportLOD;

        for (ULandscapeComponent* Component : Proxy->LandscapeComponents)
        {
            if (!Component || !Component->IsRegistered()) { continue; }

            FMeshDescription Mesh;
            FStaticMeshAttributes(Mesh).Register();

            bool bExported = false;
#if WITH_EDITOR
            // UE5.2: ExportToRawMesh(int32 InExportLOD, FMeshDescription& OutRawMesh) — void, 2 params.
            Component->ExportToRawMesh(ExportLod, Mesh);
            bExported = Mesh.VertexInstances().Num() > 0;
#endif
            FMeshLod Lod;
            bool bHaveGeometry = bExported && ConvertMeshDescriptionToLod(Mesh, Lod);

            if (!bHaveGeometry)
            {
                Ctx.AddIssue(TEXT("warn"), TEXT("landscape"),
                    FString::Printf(TEXT("ExportToRawMesh failed for component %s — emitting placeholder quad"),
                        *Component->GetPathName()),
                    Identity::ForComponent(Ctx.SceneGuid, Component).Value);

                // Fallback: a single quad spanning the component bounds.
                const FBoxSphereBounds B = Component->Bounds;
                const FVector C = B.Origin;
                const FVector E = B.BoxExtent;
                Lod.LodIndex = 0;
                Lod.Positions = {
                    FVector3f(C - FVector(E.X, E.Y, 0)),
                    FVector3f(C + FVector(E.X, -E.Y, 0)),
                    FVector3f(C + FVector(E.X, E.Y, 0)),
                    FVector3f(C + FVector(-E.X, E.Y, 0)),
                };
                Lod.Normals.Init(FVector3f::ZAxisVector, 4);
                Lod.Tangents.Init(FVector4f(1, 0, 0, 1), 4);
                Lod.UVs.SetNum(1);
                Lod.UVs[0] = { {0,0}, {1,0}, {1,1}, {0,1} };
                Lod.Indices = { 0,1,2, 0,2,3 };
                FMeshSection MS;
                MS.SectionIndex = 0;
                MS.MaterialSlotIndex = 0;
                MS.FirstIndex = 0;
                MS.IndexCount = 6;
                MS.MinVertex = 0;
                MS.MaxVertex = 3;
                Lod.Sections.Add(MoveTemp(MS));
            }

            int32 MeshIndex = INDEX_NONE;
            const FStableId MeshId = EmitComponentMeshAsset(
                Component, Ctx, MoveTemp(Lod), Component->GetMaterialInstance(0, false), MeshIndex);

            // Component → primitive instance.
            FPrimitiveInstance Prim;
            Prim.Id = Identity::ForComponent(Ctx.SceneGuid, Component);
            Prim.MeshId = MeshId;
            Prim.OwnerActorPath = Proxy->GetPathName();
            Prim.ComponentPath = Component->GetPathName();
            Prim.Transform = Component->GetComponentTransform();
            Prim.OverrideMaterialIds.Add(
                SceneRTV2::Material::Resolve(Component->GetMaterialInstance(0, false), nullptr, 0, Ctx));
            Prim.bVisible = Component->IsVisible();
            Prim.bCastShadow = Component->CastShadow;
            Prim.bAffectsRayTracing = Component->bVisibleInRayTracing;
            Ctx.Primitives.Add(Prim);

            // Per-component metadata in the landscape record.
            FLandscapeComponentInfo Info;
            Info.MeshId = MeshId;
            Info.InstanceId = Prim.Id;
            Info.OwnerActorPath = Proxy->GetPathName();
            Info.ComponentPath = Component->GetPathName();
            Info.SectionBaseX = Component->SectionBaseX;
            Info.SectionBaseY = Component->SectionBaseY;
            GatherComponentLayerWeights(Component, Info, Ctx);
            Record.Components.Add(MoveTemp(Info));
        }
    }

    void CollectLandscape(ALandscape* Landscape, FExportContext& Ctx)
    {
        CollectProxyLike(Landscape, Ctx);
    }

    void CollectLandscapeProxy(ALandscapeProxy* Proxy, FExportContext& Ctx)
    {
        CollectProxyLike(Proxy, Ctx);
    }
}
