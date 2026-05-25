#pragma once

#include "CoreMinimal.h"

namespace SceneRTV2
{
    struct FExportContext;
    struct FStableId;
}
class UTexture;
class URuntimeVirtualTexture;

namespace SceneRTV2::Texture
{
    /**
     * Resolves any UTexture (Texture2D, TextureCube, TextureRenderTarget2D,
     * VolumeTexture) to a stable id. Idempotent — second call returns the
     * cached id.
     *
     * Handles, transparently:
     *  - CompositeTexture (texture references another as detail) → flattens via
     *    GPU bake if bBakeCompositeTextures.
     *  - sRGB / NormalMap / HDR flags carried through to FTextureRecord.
     *  - Source-art readback path for editor-only data (preserves original
     *    resolution before mip clamp).
     */
    FStableId Resolve(UTexture* Texture, FExportContext& Ctx);

    /**
     * Bakes a Runtime Virtual Texture into one or more Texture2D tiles and
     * registers them. Returns the id of the baked composite (a single virtual
     * Texture2D with the union of tiles), or an empty id if RVT baking is
     * disabled or fails.
     */
    FStableId BakeRuntimeVirtualTexture(URuntimeVirtualTexture* Rvt,
                                        const FBox& WorldBounds,
                                        FExportContext& Ctx);

    /** Per-channel ORM auto-detection from texture name + sampler usage. */
    void DetectOrmChannelLayout(const UTexture* Texture,
                                int32& OutAoChan, int32& OutRoughChan,
                                int32& OutMetalChan);
}
