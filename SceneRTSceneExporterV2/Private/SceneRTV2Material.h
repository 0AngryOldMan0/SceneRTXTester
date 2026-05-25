#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
    struct FMaterialRecord;
    struct FStableId;
}
class UMaterialInterface;
class UPrimitiveComponent;

namespace SceneRTV2::Material
{
    /**
     * Resolves a UMaterialInterface into a fully-parameterized FMaterialRecord.
     *
     * Critical guarantees (the source of "missing texture" / "wrong texture
     * shared between objects" bugs in V1):
     *  - Walks every EMaterialParameterAssociation (Global, Layer, Blend) and
     *    every Material Function input — V1 only handled Global.
     *  - Captures parent material + ALL resolved parameter values as the dedup
     *    key, so two MIs that override different scalars produce two records,
     *    while two identical MIs collapse into one.
     *  - Returns a stable id derived from (parent_path, resolved_params_hash).
     *  - For decal materials, additionally records projector blend mode.
     *  - For Substrate materials, falls back to a captured "snapshot" parameter
     *    set if the Strata graph cannot be flattened (rare in 5.2).
     *
     * `OwnerComponent` is optional and used to read per-component override
     * material instances (UMeshComponent::OverrideMaterials).
     */
    FStableId Resolve(UMaterialInterface* Material,
                      UPrimitiveComponent* OwnerComponent,
                      int32 SlotIndex,
                      FExportContext& Ctx);

    /** Build the dedup hash for a fully-resolved material record. */
    uint64 ComputeResolvedHash(const FMaterialRecord& Record);

    /** Detect master material category from parent path (no scene-specific names). */
    FString ClassifyMasterMaterial(const UMaterialInterface* Material);
}
