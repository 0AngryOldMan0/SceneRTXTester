#pragma once

#include "CoreMinimal.h"

// Forward-declared outside any namespace — MSVC treats inline 'class T' declarations
// as scoped to the enclosing namespace, creating phantom types like
// SceneRTV2::Identity::USplineMeshComponent that don't match the engine types.
class UObject;
class UMaterialInterface;
class UTexture;
class AActor;
class UActorComponent;
class USplineMeshComponent;

namespace SceneRTV2
{
    struct FStableId;

    /**
     * Identity contract:
     *   id = "<kind>:<sha1_hex_16>(scene_guid + ':' + path + ':' + qualifier)"
     *
     * The scene GUID prefix ensures bundles from different levels cannot collide.
     * The path is the full UObject path name (Outermost + SubpathString) — stable
     * across editor sessions for asset-backed objects, stable across PIE for
     * components.
     */
    namespace Identity
    {
        FStableId Make(const FGuid& SceneGuid, const FString& Kind, const FString& Path,
                       const FString& Qualifier = FString());

        /** Convenience wrappers for the common kinds. */
        FStableId ForMesh(const FGuid& SceneGuid, const UObject* Asset, const FString& Qualifier = FString());
        FStableId ForMaterial(const FGuid& SceneGuid, const UMaterialInterface* Material, uint64 ResolvedHash);
        FStableId ForTexture(const FGuid& SceneGuid, const UTexture* Texture);
        FStableId ForActor(const FGuid& SceneGuid, const AActor* Actor);
        FStableId ForComponent(const FGuid& SceneGuid, const UActorComponent* Component);
        FStableId ForSplineSegment(const FGuid& SceneGuid, const USplineMeshComponent* Component, int32 SegmentIndex);

        /** Hash arbitrary key payload (used for dedup, not for identity). */
        uint64 HashBytes(const void* Data, SIZE_T Size, uint64 Seed = 0);
        uint64 HashString(const FString& Str, uint64 Seed = 0);
    }
}
