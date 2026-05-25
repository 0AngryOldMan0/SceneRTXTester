#include "SceneRTV2Identity.h"
#include "SceneRTV2Context.h"

#include "Misc/SecureHash.h"
#include "Components/ActorComponent.h"
#include "Components/SplineMeshComponent.h"
#include "Engine/Texture.h"
#include "GameFramework/Actor.h"
#include "Materials/MaterialInterface.h"

namespace SceneRTV2::Identity
{
    static FString HashToHexPrefix(const FString& Seed)
    {
        FSHA1 Sha;
        const FTCHARToUTF8 Utf8(*Seed);
        Sha.Update(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
        Sha.Final();
        uint8 Bytes[20];
        Sha.GetHash(Bytes);
        return BytesToHex(Bytes, 8); // 16 hex chars, 64 bits — plenty for collision safety
    }

    FStableId Make(const FGuid& SceneGuid, const FString& Kind, const FString& Path,
                   const FString& Qualifier)
    {
        FStableId Id;
        const FString Seed = SceneGuid.ToString(EGuidFormats::Digits)
                           + TEXT(":") + Path
                           + (Qualifier.IsEmpty() ? TEXT("") : (TEXT(":") + Qualifier));
        Id.Value = Kind + TEXT(":") + HashToHexPrefix(Seed).ToLower();
        return Id;
    }

    FStableId ForMesh(const FGuid& SceneGuid, const UObject* Asset, const FString& Qualifier)
    {
        return Make(SceneGuid, TEXT("mesh"),
                    Asset ? Asset->GetPathName() : FString(TEXT("null")), Qualifier);
    }

    FStableId ForMaterial(const FGuid& SceneGuid, const UMaterialInterface* Material, uint64 ResolvedHash)
    {
        const FString Path = Material ? Material->GetPathName() : FString(TEXT("null"));
        return Make(SceneGuid, TEXT("mat"), Path, FString::Printf(TEXT("h%llx"), ResolvedHash));
    }

    FStableId ForTexture(const FGuid& SceneGuid, const UTexture* Texture)
    {
        return Make(SceneGuid, TEXT("tex"),
                    Texture ? Texture->GetPathName() : FString(TEXT("null")));
    }

    FStableId ForActor(const FGuid& SceneGuid, const AActor* Actor)
    {
        return Make(SceneGuid, TEXT("actor"),
                    Actor ? Actor->GetPathName() : FString(TEXT("null")));
    }

    FStableId ForComponent(const FGuid& SceneGuid, const UActorComponent* Component)
    {
        return Make(SceneGuid, TEXT("prim"),
                    Component ? Component->GetPathName() : FString(TEXT("null")));
    }

    FStableId ForSplineSegment(const FGuid& SceneGuid, const USplineMeshComponent* Component, int32 SegmentIndex)
    {
        return Make(SceneGuid, TEXT("splseg"),
                    Component ? Component->GetPathName() : FString(TEXT("null")),
                    FString::Printf(TEXT("seg%d"), SegmentIndex));
    }

    uint64 HashBytes(const void* Data, SIZE_T Size, uint64 Seed)
    {
        return CityHash64WithSeed(static_cast<const char*>(Data), Size, Seed);
    }

    uint64 HashString(const FString& Str, uint64 Seed)
    {
        const FTCHARToUTF8 Utf8(*Str);
        return HashBytes(Utf8.Get(), Utf8.Length(), Seed);
    }
}
