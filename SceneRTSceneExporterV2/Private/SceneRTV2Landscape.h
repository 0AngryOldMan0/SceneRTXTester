#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
    struct FStableId;
    struct FMeshLod;
}
class ALandscape;
class ALandscapeProxy;
class ULandscapeComponent;
struct FMeshDescription;

namespace SceneRTV2::Landscape
{
    /**
     * Triangulates an ALandscape or ALandscapeProxy at the configured LOD and
     * emits one FMeshAsset + FPrimitiveInstance per ULandscapeComponent, plus a
     * single FLandscapeRecord containing the metadata that links the components
     * back together (heightmap, layer weights, shared base material).
     *
     * Strategy:
     *  1. Resolve LandscapeMaterial / LandscapeHoleMaterial once per record.
     *  2. For each ULandscapeComponent:
     *       - Build a FMeshDescription via ULandscapeComponent::ExportToRawMesh
     *         (5.2 editor-only API). On failure, fall back to component bounds
     *         triangulation so the ray tracer at least has a quad placeholder.
     *       - Convert FMeshDescription → SceneRTV2::FMeshLod.
     *       - Resolve the component material override (per-component MIs are
     *         common when paint layers vary) through Material::Resolve.
     *       - Sample the weightmap layer allocations and store layer name +
     *         packed weightmap channel index. The weightmap textures are
     *         registered as FTextureRecord via Texture::Resolve so the ray
     *         tracer can blend layers exactly as UE does.
     *  3. Bundle everything into one FLandscapeRecord keyed by LandscapeGuid so
     *     streaming proxies coalesce into the same logical landscape.
     */
    void CollectLandscape(ALandscape* Landscape, FExportContext& Ctx);
    void CollectLandscapeProxy(ALandscapeProxy* Proxy, FExportContext& Ctx);

    /** Shared dispatch: handles both ALandscape and ALandscapeProxy. */
    void CollectProxyLike(ALandscapeProxy* Proxy, FExportContext& Ctx);

    /** Internal helper exposed for unit tests / writer reuse. */
    bool ConvertMeshDescriptionToLod(const FMeshDescription& Mesh, FMeshLod& OutLod);
}
