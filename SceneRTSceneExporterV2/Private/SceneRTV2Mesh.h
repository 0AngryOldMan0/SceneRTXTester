#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
    struct FMeshAsset;
    struct FStableId;
}
class UStaticMeshComponent;
class USkeletalMeshComponent;
class UInstancedStaticMeshComponent;

namespace SceneRTV2::Mesh
{
    /**
     * Collects geometry for a UStaticMeshComponent (or HISM/ISM).
     *
     * Handles, regardless of asset:
     *  - Nanite-only meshes (no CPU LOD): falls back to FMeshDescription source
     *    data (editor builds only), or — if bExportNaniteFallback=false —
     *    records the asset without LODs and emits a validation warning.
     *  - bAllowCPUAccess=false meshes: temporarily renders the LOD via
     *    FStaticMeshOperations to obtain CPU triangles.
     *  - Multiple LODs per settings.
     *  - Per-section vertex range for accurate slot→triangle mapping.
     */
    int32 CollectStaticMesh(UStaticMeshComponent* Component, FExportContext& Ctx);

    /** Skeletal mesh path. Includes skin indices/weights and reference pose. */
    int32 CollectSkeletalMesh(USkeletalMeshComponent* Component, FExportContext& Ctx);

    /** Instanced static mesh: collects mesh once, emits per-instance transforms. */
    void CollectInstancedStatic(UInstancedStaticMeshComponent* Component, FExportContext& Ctx);
}
