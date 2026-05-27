#include "SceneRTV2SplineMesh.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"
#include "SceneRTV2Material.h"

#include "Components/SplineMeshComponent.h"
#include "Components/SplineComponent.h"
#include "GameFramework/Actor.h"
#include "Engine/StaticMesh.h"
#include "StaticMeshResources.h"
#include "Materials/MaterialInterface.h"

namespace SceneRTV2::SplineMesh
{
    namespace
    {
        /**
         * Re-implementation of USplineMeshComponent::CalcSliceTransform that we
         * can apply per-vertex in CPU. Mirrors the engine math so the exported
         * geometry matches in-game appearance bit-for-bit.
         */
        FTransform SliceTransformAt(const USplineMeshComponent* Spline, float DistanceAlong)
        {
            return Spline->CalcSliceTransform(DistanceAlong);
        }
    }

    int32 CollectSplineMeshSegment(USplineMeshComponent* Component,
                                   const USplineComponent* OwnerSpline,
                                   int32 SegmentIndex,
                                   float SplineArclenOffset,
                                   FExportContext& Ctx)
    {
        if (!Component || !Component->GetStaticMesh()) { return INDEX_NONE; }
        UStaticMesh* SrcMesh = Component->GetStaticMesh();
        const FStaticMeshRenderData* RenderData = SrcMesh->GetRenderData();
        if (!RenderData || RenderData->LODResources.Num() == 0) { return INDEX_NONE; }

        const FStaticMeshLODResources& LODSrc = RenderData->LODResources[0];

        FMeshAsset Asset;
        Asset.Kind = SceneRTSceneExporterV2::EMeshSourceKind::SplineMeshSegment;
        Asset.SourceAssetPath = SrcMesh->GetPathName();
        Asset.DisplayName = FString::Printf(TEXT("%s_seg%d"), *SrcMesh->GetName(), SegmentIndex);
        Asset.Id = Identity::ForSplineSegment(Ctx.SceneGuid, Component, SegmentIndex);

        FMeshLod Lod;
        Lod.LodIndex = 0;
        const int32 NumVerts = LODSrc.VertexBuffers.PositionVertexBuffer.GetNumVertices();
        const int32 NumUVChans = LODSrc.VertexBuffers.StaticMeshVertexBuffer.GetNumTexCoords();
        Lod.Positions.SetNumUninitialized(NumVerts);
        Lod.Normals.SetNumUninitialized(NumVerts);
        Lod.Tangents.SetNumUninitialized(NumVerts);
        Lod.UVs.SetNum(FMath::Max(NumUVChans, 1));
        for (int32 c = 0; c < Lod.UVs.Num(); ++c) { Lod.UVs[c].SetNumUninitialized(NumVerts); }
        Lod.ArclenAlongSpline.SetNumUninitialized(NumVerts);

        const FVector MeshMin = SrcMesh->GetBoundingBox().Min;
        const FVector MeshMax = SrcMesh->GetBoundingBox().Max;
        const float MeshLength = FMath::Max(KINDA_SMALL_NUMBER,
            float((&MeshMax.X)[Component->ForwardAxis] - (&MeshMin.X)[Component->ForwardAxis]));

        const float SegmentLen = Component->GetSplineLength(); // local along forward axis

        // Resolve tile world size from material scalar parameter.
        // Fallback: MeshLength — 1 tile per segment, UV.X stays continuous via
        // cumulative arclen so no seam artifact even with the fallback value.
        float TileWorldSize = MeshLength;
        if (UMaterialInterface* Mat = Component->GetMaterial(0))
        {
            static const FName TileParamNames[] = {
                TEXT("TileWorldSize"), TEXT("TileSize"),   TEXT("WorldSize"),
                TEXT("TextureTile"),   TEXT("TilingSize"),
            };
            for (const FName& ParamName : TileParamNames)
            {
                float ParamValue = 0.f;
                if (Mat->GetScalarParameterValue(FMaterialParameterInfo(ParamName), ParamValue)
                    && ParamValue > KINDA_SMALL_NUMBER)
                {
                    TileWorldSize = ParamValue;
                    break;
                }
            }
        }

        for (int32 v = 0; v < NumVerts; ++v)
        {
            const FVector3f LocalPos = LODSrc.VertexBuffers.PositionVertexBuffer.VertexPosition(v);
            const float AlphaAlongMesh = (LocalPos[Component->ForwardAxis] - float((&MeshMin.X)[Component->ForwardAxis])) / MeshLength;
            const float DistanceAlong = AlphaAlongMesh * SegmentLen;
            const FTransform Slice = SliceTransformAt(Component, DistanceAlong);

            // Strip forward component before applying slice, mirroring engine behaviour.
            FVector3f Stripped = LocalPos;
            Stripped[Component->ForwardAxis] = 0.f;
            const FVector World = Slice.TransformPosition(FVector(Stripped));

            Lod.Positions[v] = FVector3f(World);

            const FVector4f TZ = LODSrc.VertexBuffers.StaticMeshVertexBuffer.VertexTangentZ(v);
            FVector3f Normal = FVector3f(TZ.X, TZ.Y, TZ.Z);
            Lod.Normals[v] = FVector3f(Slice.TransformVectorNoScale(FVector(Normal)));

            const FVector4f TX = LODSrc.VertexBuffers.StaticMeshVertexBuffer.VertexTangentX(v);
            const FVector3f Tan = FVector3f(TX.X, TX.Y, TX.Z);
            const FVector TanWorld = Slice.TransformVectorNoScale(FVector(Tan));
            Lod.Tangents[v] = FVector4f(FVector3f(TanWorld), 1.f);

            // UV channels: rewrite channel 0 to be arclen-periodic across the chain.
            for (int32 c = 0; c < Lod.UVs.Num() && c < NumUVChans; ++c)
            {
                FVector2f Uv = LODSrc.VertexBuffers.StaticMeshVertexBuffer.GetVertexUV(v, c);
                if (c == 0 && Ctx.Settings && Ctx.Settings->bWriteArclenUv)
                {
                    Uv.X = (SplineArclenOffset + DistanceAlong) / TileWorldSize;
                }
                Lod.UVs[c][v] = Uv;
            }
            Lod.ArclenAlongSpline[v] = SplineArclenOffset + DistanceAlong;
        }

        // Indices
        TArray<uint32> Indices;
        LODSrc.IndexBuffer.GetCopy(Indices);
        Lod.Indices = MoveTemp(Indices);

        // Sections
        for (int32 s = 0; s < LODSrc.Sections.Num(); ++s)
        {
            const FStaticMeshSection& Src = LODSrc.Sections[s];
            FMeshSection MS;
            MS.SectionIndex = s;
            MS.MaterialSlotIndex = Src.MaterialIndex;
            MS.FirstIndex = Src.FirstIndex;
            MS.IndexCount = Src.NumTriangles * 3;
            MS.MinVertex = Src.MinVertexIndex;
            MS.MaxVertex = Src.MaxVertexIndex;
            Lod.Sections.Add(MoveTemp(MS));
        }

        Asset.Lods.Add(MoveTemp(Lod));

        // Per-slot material resolution (component-level overrides applied).
        for (int32 s = 0; s < SrcMesh->GetStaticMaterials().Num(); ++s)
        {
            UMaterialInterface* Mat = Component->GetMaterial(s);
            Asset.SlotMaterialIds.Add(SceneRTV2::Material::Resolve(Mat, Component, s, Ctx));
        }

        const int32 Idx = Ctx.Meshes.Add(MoveTemp(Asset));
        return Idx;
    }

    void CollectActorSplineChain(AActor* Actor, FExportContext& Ctx)
    {
        if (!Actor) { return; }
        TArray<USplineMeshComponent*> Segments;
        Actor->GetComponents<USplineMeshComponent>(Segments);
        if (Segments.Num() == 0) { return; }

        // Sort by Y coordinate of start of segment to maintain a stable order.
        Segments.Sort([](USplineMeshComponent& A, USplineMeshComponent& B)
        {
            return A.GetName() < B.GetName();
        });

        // Find a parent spline component to use for arclen offsets.
        USplineComponent* ParentSpline = nullptr;
        Actor->GetComponents<USplineComponent>().FindItemByClass(&ParentSpline);

        const FString GroupId = Identity::ForActor(Ctx.SceneGuid, Actor).Value;

        float ArclenAccum = 0.f;
        for (int32 i = 0; i < Segments.Num(); ++i)
        {
            USplineMeshComponent* Seg = Segments[i];
            if (!Seg || !Seg->IsRegistered()) { continue; }

            const float SegLen = Seg->GetSplineLength();
            const int32 MeshIdx = CollectSplineMeshSegment(Seg, ParentSpline, i, ArclenAccum, Ctx);

            if (MeshIdx != INDEX_NONE)
            {
                FPrimitiveInstance Prim;
                Prim.Id = Identity::ForComponent(Ctx.SceneGuid, Seg);
                Prim.MeshId = Ctx.Meshes[MeshIdx].Id;
                Prim.OwnerActorPath = Actor->GetPathName();
                Prim.ComponentPath = Seg->GetPathName();
                Prim.Transform = Seg->GetComponentTransform();
                Prim.SplineGroupId = GroupId;
                Prim.SplineSegmentIndex = i;
                Prim.SplineArclenStart = ArclenAccum;
                Prim.SplineArclenEnd = ArclenAccum + SegLen;
                Prim.bVisible = Seg->IsVisible();
                Prim.bCastShadow = Seg->CastShadow;
                Prim.bAffectsRayTracing = Seg->bVisibleInRayTracing;

                // Per-slot overrides resolved as material ids.
                for (int32 s = 0; s < Seg->GetNumMaterials(); ++s)
                {
                    Prim.OverrideMaterialIds.Add(
                        SceneRTV2::Material::Resolve(Seg->GetMaterial(s), Seg, s, Ctx));
                }

                Ctx.Primitives.Add(MoveTemp(Prim));
            }

            ArclenAccum += SegLen;
        }
    }
}
