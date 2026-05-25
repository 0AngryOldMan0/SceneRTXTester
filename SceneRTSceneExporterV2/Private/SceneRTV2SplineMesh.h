#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
}
class USplineMeshComponent;
class USplineComponent;
class AActor;

namespace SceneRTV2::SplineMesh
{
    /**
     * Collects a single USplineMeshComponent as a deformed mesh segment.
     *
     * Critical for fixing "tiling mismatch at spline joints":
     *  - UV.x of every vertex is rewritten to (arclen_along_segment +
     *    spline_arclen_offset) / TileWorldSize, where TileWorldSize comes from
     *    the material's WorldArclen / TileWidth parameter (resolved through
     *    Material::Resolve, with a default fallback). This makes UV strictly
     *    periodic along the entire chain instead of restarting per-segment.
     *  - ArclenAlongSpline buffer is filled and written to the binary so the
     *    consumer can validate or recompute UVs.
     *  - Each spline-mesh component is tagged with SplineGroupId =
     *    StableId(actor) + ":" + StableId(parent_spline), letting the consumer
     *    treat the chain as one logical curve.
     */
    int32 CollectSplineMeshSegment(USplineMeshComponent* Component,
                                   const USplineComponent* OwnerSpline,
                                   int32 SegmentIndex,
                                   float SplineArclenOffset,
                                   FExportContext& Ctx);

    /**
     * Walks all spline-mesh segments owned by `Actor`, computes per-segment
     * arclen offsets, and emits both the segment meshes and the primitive
     * instances pointing at them.
     */
    void CollectActorSplineChain(AActor* Actor, FExportContext& Ctx);
}
