#include "SceneRTSceneExporterCore.h"
#include "SceneRTSceneExportSettings.h"
#include "SceneRTExportFormat.h"

#if WITH_EDITOR
#include "Editor.h"
#endif

#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "EngineUtils.h"
#include "Engine/StaticMesh.h"
#include "StaticMeshResources.h"

#include "Components/ActorComponent.h"
#include "Components/SceneComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SplineMeshComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/HierarchicalInstancedStaticMeshComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/DecalComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Components/LightComponent.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/SpotLightComponent.h"
#include "Components/RectLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/PostProcessComponent.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/VolumetricCloudComponent.h"

#include "Camera/CameraComponent.h"
#include "Camera/CameraTypes.h"
#include "CineCameraComponent.h"

#include "Materials/Material.h"
#include "Materials/MaterialInstance.h"
#include "Materials/MaterialInterface.h"

#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"
#include "Engine/SkeletalMesh.h"
#include "UObject/UnrealType.h"

#include "Landscape.h"
#include "LandscapeComponent.h"
#include "LandscapeProxy.h"
#include "LandscapeDataAccess.h"

#include "WorldPartition/WorldPartition.h"

#include "ImageUtils.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/Crc.h"
#include "Misc/SecureHash.h"
#include "HAL/PlatformFilemanager.h"

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

#include "Components/StaticMeshComponent.h"
#include "StaticMeshComponentLODInfo.h"

DEFINE_LOG_CATEGORY_STATIC(LogSceneRTSceneExporter, Log, All);

namespace SceneRTSceneExporter
{
    using namespace SceneRTExportFormat;

    struct FSettings
    {
        FString OutputDirectory;
        FString SceneJsonPath;
        FString MeshesBinaryPath;
        FString TexturesDir;

        double UnitScale = 1.0;
        bool bMirrorYToRightHanded = true;
        bool bExportOnlyVisibleComponents = true;
        bool bExportHiddenActors = false;
        bool bExportAllStaticMeshLODs = true;
        bool bExportSplineMeshGeometry = true;
        bool bExportLandscapeGeometry = true;
        bool bExportSkeletalMeshMetadata = true;
        bool bExportMaterialTextures = true;
        bool bExportPostProcess = true;
        bool bExportFog = true;
        bool bExportAtmosphere = true;
        bool bExportDecals = true;
        bool bExportLights = true;
        bool bExportCameras = true;
        bool bExportActorTags = true;
        bool bExportDataLayers = true;
        int32 MaxTextureExportSize = 0;
    };

    struct FMeshSection
    {
        uint32 SectionIndex = 0;
        uint32 MaterialSlotIndex = 0;
        uint32 FirstIndex = 0;
        uint32 IndexCount = 0;
        FString MaterialSlotName;
    };

    struct FMeshLODData
    {
        int32 LODIndex = 0;
        TArray<FVector3f> Positions;
        TArray<FVector3f> Normals;
        TArray<FVector4f> Tangents;
        TArray<FColor> Colors;
        TArray<TArray<FVector2f>> UVSets;
        TArray<uint32> Indices;
        TArray<FMeshSection> Sections;

        bool bHasVertexColors = false;
        bool bUsesComponentVertexColors = false;

        uint64 PositionsOffset = 0;
        uint64 NormalsOffset = 0;
        uint64 TangentsOffset = 0;
        uint64 ColorsOffset = 0;
        uint64 UVsOffset = 0;
        uint64 IndicesOffset = 0;
        uint64 SectionsOffset = 0;
    };

    struct FMeshAsset
    {
        FString StableId;
        FString Name;
        FString Key;
        FString PrimitiveType;
        FString SourceAssetPath;
        FString MeshSpace;
        TArray<FMeshLODData> LODs;

        uint64 BinaryOffset = 0;
        uint64 BinarySize = 0;
    };

    struct FExportContext
    {
        FSettings Settings;

        TArray<TSharedPtr<FJsonValue>> ActorsJson;
        TArray<TSharedPtr<FJsonValue>> PrimitivesJson;
        TArray<TSharedPtr<FJsonValue>> MaterialsJson;
        TArray<TSharedPtr<FJsonValue>> LightsJson;
        TArray<TSharedPtr<FJsonValue>> CamerasJson;
        TArray<TSharedPtr<FJsonValue>> AtmosphereJson;
        TArray<TSharedPtr<FJsonValue>> FogJson;
        TArray<TSharedPtr<FJsonValue>> PostProcessJson;
        TArray<TSharedPtr<FJsonValue>> DecalsJson;
        TArray<TSharedPtr<FJsonValue>> TexturesJson;
        TArray<TSharedPtr<FJsonValue>> LandscapesJson;

        TArray<FMeshAsset> MeshAssets;

        TMap<FString, int32> MeshKeyToIndex;
        TMap<FString, FString> MaterialKeyToId;
        TMap<FString, int32> TextureAssetPathToIndex;
    };

    static FString StableId(const FString& Prefix, const FString& Seed)
    {
        return Prefix + TEXT("_") + FMD5::HashAnsiString(*Seed);
    }

    static uint64 StableHash64(const FString& Text)
    {
        FTCHARToUTF8 Conv(*Text);
        return static_cast<uint64>(FCrc::MemCrc32(Conv.Get(), Conv.Length()));
    }

    static FString ShortStableSuffix(const FString& Seed)
    {
        return FMD5::HashAnsiString(*Seed).Left(8);
    }

    static FString SanitizeExportNamePart(const FString& InText)
    {
        FString Out;
        Out.Reserve(InText.Len());
        for (const TCHAR Ch : InText)
        {
            if (FChar::IsAlnum(Ch) || Ch == TEXT('_'))
            {
                Out.AppendChar(Ch);
            }
            else if (Ch == TEXT(' ') || Ch == TEXT('-') || Ch == TEXT('.') || Ch == TEXT(':') || Ch == TEXT('/') || Ch == TEXT('\\'))
            {
                Out.AppendChar(TEXT('_'));
            }
        }

        while (Out.Contains(TEXT("__")))
        {
            Out.ReplaceInline(TEXT("__"), TEXT("_"));
        }
        Out.TrimStartAndEndInline();
        while (Out.StartsWith(TEXT("_")))
        {
            Out.RightChopInline(1, false);
        }
        while (Out.EndsWith(TEXT("_")))
        {
            Out.LeftChopInline(1, false);
        }

        return Out.IsEmpty() ? TEXT("unnamed") : Out;
    }

    static FString GetActorExportName(const AActor* Actor)
    {
        if (!Actor)
        {
            return TEXT("NoActor");
        }

        FString Name = Actor->GetActorNameOrLabel();
        if (Name.IsEmpty())
        {
            Name = Actor->GetName();
        }
        return SanitizeExportNamePart(Name);
    }

    static FString GetComponentReadableExportName(const UActorComponent* Component)
    {
        if (!Component)
        {
            return TEXT("NoComponent");
        }

        FString Name = Component->GetReadableName();
        if (Name.IsEmpty())
        {
            Name = Component->GetName();
        }
        return SanitizeExportNamePart(Name);
    }

    static FString GetObjectExportName(const UObject* Object)
    {
        if (!Object)
        {
            return FString();
        }
        return SanitizeExportNamePart(Object->GetName());
    }

    static FString MakeUniqueExportName(const AActor* Owner, const UActorComponent* Component, const UObject* AssetObject = nullptr, const FString& ExplicitCategory = FString())
    {
        TArray<FString> Parts;
        Parts.Reserve(4);

        const FString ActorName = GetActorExportName(Owner);
        if (!ActorName.IsEmpty())
        {
            Parts.Add(ActorName);
        }

        const FString AssetName = GetObjectExportName(AssetObject);
        if (!AssetName.IsEmpty() && !Parts.Contains(AssetName))
        {
            Parts.Add(AssetName);
        }

        const FString ComponentName = GetComponentReadableExportName(Component);
        if (!ComponentName.IsEmpty() && !Parts.Contains(ComponentName))
        {
            Parts.Add(ComponentName);
        }

        if (!ExplicitCategory.IsEmpty())
        {
            const FString Category = SanitizeExportNamePart(ExplicitCategory);
            if (!Category.IsEmpty() && !Parts.Contains(Category))
            {
                Parts.Add(Category);
            }
        }

        FString Seed;
        if (Component)
        {
            Seed = Component->GetPathName();
        }
        else if (Owner)
        {
            Seed = Owner->GetPathName();
        }
        else if (AssetObject)
        {
            Seed = AssetObject->GetPathName();
        }
        else
        {
            Seed = ExplicitCategory;
        }

        FString Base = FString::Join(Parts, TEXT("__"));
        if (Base.IsEmpty())
        {
            Base = TEXT("SceneNode");
        }

        return Base + TEXT("__") + ShortStableSuffix(Seed);
    }

    static void ExtractCameraView(UCameraComponent* Camera, FVector& OutLocation, FRotator& OutRotation, float& OutFOV)
    {
        OutLocation = Camera ? Camera->GetComponentLocation() : FVector::ZeroVector;
        OutRotation = Camera ? Camera->GetComponentRotation() : FRotator::ZeroRotator;
        OutFOV = Camera ? Camera->FieldOfView : 90.0f;

        if (!Camera)
        {
            return;
        }

        FMinimalViewInfo ViewInfo;
        Camera->GetCameraView(0.0f, ViewInfo);
        OutLocation = ViewInfo.Location;
        OutRotation = ViewInfo.Rotation;
        OutFOV = ViewInfo.FOV;

#if WITH_EDITOR
        FMinimalViewInfo PreviewInfo;
        if (Camera->GetEditorPreviewInfo(0.0f, PreviewInfo))
        {
            OutLocation = PreviewInfo.Location;
            OutRotation = PreviewInfo.Rotation;
            OutFOV = PreviewInfo.FOV;
        }
#endif
    }

    static FVector ConvertLocalPos(const FVector& InPos, const FSettings& Settings)
    {
        return Settings.bMirrorYToRightHanded ? FVector(InPos.X, -InPos.Y, InPos.Z) : InPos;
    }

    static FVector ConvertLocalDir(const FVector& InDir, const FSettings& Settings)
    {
        return Settings.bMirrorYToRightHanded ? FVector(InDir.X, -InDir.Y, InDir.Z) : InDir;
    }

    static FVector ConvertWorldPosScaled(const FVector& InPos, const FSettings& Settings)
    {
        return ConvertLocalPos(InPos * Settings.UnitScale, Settings);
    }

    static TArray<TSharedPtr<FJsonValue>> Vec3ToJson(const FVector& V)
    {
        return {
            MakeShared<FJsonValueNumber>(V.X),
            MakeShared<FJsonValueNumber>(V.Y),
            MakeShared<FJsonValueNumber>(V.Z)
        };
    }

    static TArray<TSharedPtr<FJsonValue>> ColorToJson(const FLinearColor& C)
    {
        return {
            MakeShared<FJsonValueNumber>(C.R),
            MakeShared<FJsonValueNumber>(C.G),
            MakeShared<FJsonValueNumber>(C.B),
            MakeShared<FJsonValueNumber>(C.A)
        };
    }

    static TArray<TSharedPtr<FJsonValue>> NameArrayToJson(const TArray<FName>& Items)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        for (const FName& Item : Items)
        {
            Out.Add(MakeShared<FJsonValueString>(Item.ToString()));
        }
        return Out;
    }

    static TArray<TSharedPtr<FJsonValue>> MatrixToJson(const TArray<double>& M)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        for (double Value : M)
        {
            Out.Add(MakeShared<FJsonValueNumber>(Value));
        }
        return Out;
    }

    static TArray<double> IdentityMatrixJson()
    {
        return {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
    }

    static TArray<double> MakeSceneTransformMatrix(const FTransform& UETransform, const FSettings& Settings)
    {
        const FVector O = ConvertWorldPosScaled(UETransform.TransformPosition(FVector::ZeroVector), Settings);
        const FVector X = ConvertWorldPosScaled(UETransform.TransformPosition(FVector(1.0, 0.0, 0.0)), Settings) - O;
        const FVector Y = Settings.bMirrorYToRightHanded
            ? (ConvertWorldPosScaled(UETransform.TransformPosition(FVector(0.0, -1.0, 0.0)), Settings) - O)
            : (ConvertWorldPosScaled(UETransform.TransformPosition(FVector(0.0, 1.0, 0.0)), Settings) - O);
        const FVector Z = ConvertWorldPosScaled(UETransform.TransformPosition(FVector(0.0, 0.0, 1.0)), Settings) - O;

        return {
            X.X, Y.X, Z.X, O.X,
            X.Y, Y.Y, Z.Y, O.Y,
            X.Z, Y.Z, Z.Z, O.Z,
            0.0, 0.0, 0.0, 1.0
        };
    }

    static FString SafeFileName(const FString& InName)
    {
        FString Safe = InName;
        static const TCHAR* Forbidden = TEXT("\\/:*?\"<>| ");
        for (const TCHAR* C = Forbidden; *C; ++C)
        {
            const FString From = FString::Chr(*C);
            Safe = Safe.Replace(*From, TEXT("_"), ESearchCase::CaseSensitive);
        }
        return Safe;
    }

    static bool ShouldExportActor(const AActor* Actor, const FSettings& Settings, const AActor* ExporterActor)
    {
        if (!Actor || Actor == ExporterActor)
        {
            return false;
        }
        if (!Settings.bExportHiddenActors && Actor->IsHidden())
        {
            return false;
        }
        return true;
    }

    static bool ShouldExportComponent(const UActorComponent* Component, const FSettings& Settings)
    {
        if (!Component)
        {
            return false;
        }
        if (!Settings.bExportOnlyVisibleComponents)
        {
            return true;
        }
        if (const USceneComponent* Scene = Cast<USceneComponent>(Component))
        {
            if (!Scene->IsVisible())
            {
                return false;
            }
        }
        if (const UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Component))
        {
            if (Primitive->bHiddenInGame)
            {
                return false;
            }
        }
        return true;
    }

    static FBoxSphereBounds GetSceneBoundsFromWorldBounds(const FBoxSphereBounds& InBounds, const FSettings& Settings)
    {
        FBoxSphereBounds Out = InBounds;
        Out.Origin = ConvertWorldPosScaled(InBounds.Origin, Settings);
        Out.BoxExtent *= Settings.UnitScale;
        Out.SphereRadius *= Settings.UnitScale;
        return Out;
    }

    static TSharedPtr<FJsonObject> BoundsToJson(const FBoxSphereBounds& Bounds)
    {
        TSharedPtr<FJsonObject> J = MakeShared<FJsonObject>();
        J->SetArrayField(TEXT("origin"), Vec3ToJson(Bounds.Origin));
        J->SetArrayField(TEXT("box_extent"), Vec3ToJson(Bounds.BoxExtent));
        J->SetNumberField(TEXT("sphere_radius"), Bounds.SphereRadius);
        return J;
    }

    static bool SaveTexturePng(UTexture2D* Texture, const FString& FullPath)
    {
#if WITH_EDITOR
        if (!Texture)
        {
            return false;
        }
        FImage SrcImage;
        if (!FImageUtils::GetTexture2DSourceImage(Texture, SrcImage))
        {
            return false;
        }
        IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
        PF.CreateDirectoryTree(*FPaths::GetPath(FullPath));
        return FImageUtils::SaveImageByExtension(*FullPath, FImageView(SrcImage), 100);
#else
        return false;
#endif
    }

    static FString ExportTexture(UTexture2D* Texture, FExportContext& Context)
    {
        if (!Texture)
        {
            return FString();
        }

        const FString AssetPath = Texture->GetPathName();
        if (const int32* Existing = Context.TextureAssetPathToIndex.Find(AssetPath))
        {
            return Context.TexturesJson[*Existing]->AsObject()->GetStringField(TEXT("stable_id"));
        }

        const FString TextureId = StableId(TEXT("tex"), AssetPath);
        const FString FileName = SafeFileName(Texture->GetName()) + TEXT("_") + FString::Printf(TEXT("%08X.png"), FCrc::StrCrc32(*AssetPath));
        const FString FullPath = FPaths::Combine(Context.Settings.TexturesDir, FileName);
        const FString RelativePath = FString::Printf(TEXT("Textures/%s"), *FileName);

        if (Context.Settings.bExportMaterialTextures)
        {
            SaveTexturePng(Texture, FullPath);
        }

        TSharedPtr<FJsonObject> JT = MakeShared<FJsonObject>();
        JT->SetStringField(TEXT("stable_id"), TextureId);
        JT->SetStringField(TEXT("name"), Texture->GetName());
        JT->SetStringField(TEXT("asset_path"), AssetPath);
        JT->SetStringField(TEXT("class"), Texture->GetClass()->GetName());
        JT->SetBoolField(TEXT("srgb"), Texture->SRGB);
        JT->SetNumberField(TEXT("compression"), static_cast<int32>(Texture->CompressionSettings));
        JT->SetNumberField(TEXT("size_x"), Texture->GetSizeX());
        JT->SetNumberField(TEXT("size_y"), Texture->GetSizeY());
        if (Context.Settings.bExportMaterialTextures)
        {
            JT->SetStringField(TEXT("exported_path"), RelativePath);
        }

        Context.TextureAssetPathToIndex.Add(AssetPath, Context.TexturesJson.Num());
        Context.TexturesJson.Add(MakeShared<FJsonValueObject>(JT));
        return TextureId;
    }

    static bool LooksLikeEmissiveTexture(const FString& UpperName)
    {
        return UpperName.Contains(TEXT("EMISS"))
            || UpperName.Contains(TEXT("_EMM"))
            || UpperName.Contains(TEXT("_EMI"))
            || UpperName.Contains(TEXT("GLOW"))
            || UpperName.Contains(TEXT("ILLUM"))
            || UpperName.Contains(TEXT("SELFILLUM"));
    }

    static bool WantsEmissiveParameter(const FString& UpperName)
    {
        return UpperName.Contains(TEXT("EMISS"))
            || UpperName.Contains(TEXT("GLOW"))
            || UpperName.Contains(TEXT("ILLUM"));
    }

    static void ClassifyTextureSlot(
        const FString& ParamUpper,
        const FString& TexUpper,
        UTexture2D* Texture,
        FString& InOutBaseColor,
        FString& InOutNormal,
        FString& InOutOrm,
        FString& InOutRoughness,
        FString& InOutMetallic,
        FString& InOutAO,
        FString& InOutEmissive,
        const FString& TextureId)
    {
        if (!Texture || TextureId.IsEmpty())
        {
            return;
        }

        const bool bSRGB = Texture->SRGB;
        const bool bNormal = ParamUpper.Contains(TEXT("NORMAL")) || TexUpper.Contains(TEXT("NORMAL")) || TexUpper.Contains(TEXT("_N")) || TexUpper.Contains(TEXT("_NRM"));
        const bool bPackedMasks = ParamUpper.Contains(TEXT("ORM")) || ParamUpper.Contains(TEXT("RMA")) || ParamUpper.Contains(TEXT("MASK")) || TexUpper.Contains(TEXT("ORM")) || TexUpper.Contains(TEXT("RMA")) || TexUpper.Contains(TEXT("AO_R_M")) || TexUpper.Contains(TEXT("_MASK"));
        const bool bRough = ParamUpper.Contains(TEXT("ROUGH")) || TexUpper.Contains(TEXT("ROUGH"));
        const bool bMetal = ParamUpper.Contains(TEXT("METAL")) || TexUpper.Contains(TEXT("METAL"));
        const bool bAO = ParamUpper.Contains(TEXT("AO")) || ParamUpper.Contains(TEXT("OCCLUSION")) || TexUpper.Contains(TEXT("AO")) || TexUpper.Contains(TEXT("OCCLUSION"));
        const bool bBase = ParamUpper.Contains(TEXT("BASECOLOR")) || ParamUpper.Contains(TEXT("BASE_COLOR")) || ParamUpper.Contains(TEXT("ALBEDO")) || ParamUpper.Contains(TEXT("DIFFUSE")) || TexUpper.Contains(TEXT("BASECOLOR")) || TexUpper.Contains(TEXT("BASE_COLOR")) || TexUpper.Contains(TEXT("ALBEDO")) || TexUpper.Contains(TEXT("DIFFUSE")) || TexUpper.Contains(TEXT("_ALB"));
        const bool bTextureLooksEmissive = LooksLikeEmissiveTexture(TexUpper);
        const bool bParamWantsEmissive = WantsEmissiveParameter(ParamUpper);
        const bool bValidEmissiveTexture = bTextureLooksEmissive && !bPackedMasks && !bNormal;

        if (bNormal && InOutNormal.IsEmpty()) InOutNormal = TextureId;
        if (bParamWantsEmissive && bValidEmissiveTexture && InOutEmissive.IsEmpty()) InOutEmissive = TextureId;
        if (bPackedMasks && !bSRGB && InOutOrm.IsEmpty() && !bNormal) InOutOrm = TextureId;
        if (bAO && !bSRGB && InOutAO.IsEmpty() && !bPackedMasks) InOutAO = TextureId;
        if (bRough && !bSRGB && InOutRoughness.IsEmpty() && !bPackedMasks) InOutRoughness = TextureId;
        if (bMetal && !bSRGB && InOutMetallic.IsEmpty() && !bPackedMasks) InOutMetallic = TextureId;
        if (InOutBaseColor.IsEmpty())
        {
            if (bBase)
            {
                InOutBaseColor = TextureId;
            }
            else if (bSRGB && !bNormal && !bPackedMasks && !bTextureLooksEmissive && !bAO && !bRough && !bMetal)
            {
                InOutBaseColor = TextureId;
            }
        }
    }

    static FString GetOrCreateMaterialId(UMaterialInterface* Material, FExportContext& Context)
    {
        if (!Material)
        {
            return FString();
        }

        const FString Key = Material->GetPathName();
        if (const FString* Existing = Context.MaterialKeyToId.Find(Key))
        {
            return *Existing;
        }

        const FString MaterialId = StableId(TEXT("mat"), Key);
        TSharedPtr<FJsonObject> JM = MakeShared<FJsonObject>();
        JM->SetStringField(TEXT("stable_id"), MaterialId);
        JM->SetStringField(TEXT("name"), Material->GetName());
        JM->SetStringField(TEXT("asset_path"), Material->GetPathName());
        JM->SetStringField(TEXT("class"), Material->GetClass()->GetPathName());

        FString BaseColorTexId;
        FString NormalTexId;
        FString OrmTexId;
        FString RoughTexId;
        FString MetalTexId;
        FString OcclusionTexId;
        FString EmissiveTexId;

        if (UMaterial* BaseMat = Material->GetMaterial())
        {
            JM->SetBoolField(TEXT("two_sided"), BaseMat->TwoSided);
            JM->SetNumberField(TEXT("blend_mode"), static_cast<int32>(BaseMat->BlendMode));
            JM->SetNumberField(TEXT("material_domain"), static_cast<int32>(BaseMat->MaterialDomain));
            JM->SetStringField(TEXT("base_material_asset_path"), BaseMat->GetPathName());
        }

        TArray<TSharedPtr<FJsonValue>> TextureParamsJson;
        TArray<TSharedPtr<FJsonValue>> ScalarParamsJson;
        TArray<TSharedPtr<FJsonValue>> VectorParamsJson;

        if (UMaterialInstance* MI = Cast<UMaterialInstance>(Material))
        {
            TArray<FMaterialParameterInfo> ParamInfos;
            TArray<FGuid> ParamGuids;

            MI->GetAllTextureParameterInfo(ParamInfos, ParamGuids);
            for (const FMaterialParameterInfo& ParamInfo : ParamInfos)
            {
                UTexture* Texture = nullptr;
                if (!MI->GetTextureParameterValue(ParamInfo, Texture))
                {
                    continue;
                }
                UTexture2D* Texture2D = Cast<UTexture2D>(Texture);
                if (!Texture2D)
                {
                    continue;
                }

                const FString TextureId = ExportTexture(Texture2D, Context);
                TSharedPtr<FJsonObject> JT = MakeShared<FJsonObject>();
                JT->SetStringField(TEXT("name"), ParamInfo.Name.ToString());
                JT->SetStringField(TEXT("texture_id"), TextureId);
                JT->SetStringField(TEXT("texture_name"), Texture2D->GetName());
                JT->SetStringField(TEXT("texture_asset_path"), Texture2D->GetPathName());
                TextureParamsJson.Add(MakeShared<FJsonValueObject>(JT));

                ClassifyTextureSlot(ParamInfo.Name.ToString().ToUpper(), Texture2D->GetName().ToUpper(), Texture2D,
                    BaseColorTexId, NormalTexId, OrmTexId, RoughTexId, MetalTexId, OcclusionTexId, EmissiveTexId, TextureId);
            }

            ParamInfos.Reset();
            ParamGuids.Reset();
            MI->GetAllScalarParameterInfo(ParamInfos, ParamGuids);
            for (const FMaterialParameterInfo& ParamInfo : ParamInfos)
            {
                float Value = 0.0f;
                if (!MI->GetScalarParameterValue(ParamInfo, Value))
                {
                    continue;
                }
                TSharedPtr<FJsonObject> JS = MakeShared<FJsonObject>();
                JS->SetStringField(TEXT("name"), ParamInfo.Name.ToString());
                JS->SetNumberField(TEXT("value"), Value);
                ScalarParamsJson.Add(MakeShared<FJsonValueObject>(JS));
            }

            ParamInfos.Reset();
            ParamGuids.Reset();
            MI->GetAllVectorParameterInfo(ParamInfos, ParamGuids);
            for (const FMaterialParameterInfo& ParamInfo : ParamInfos)
            {
                FLinearColor Value(0, 0, 0, 0);
                if (!MI->GetVectorParameterValue(ParamInfo, Value))
                {
                    continue;
                }
                TSharedPtr<FJsonObject> JV = MakeShared<FJsonObject>();
                JV->SetStringField(TEXT("name"), ParamInfo.Name.ToString());
                JV->SetArrayField(TEXT("value"), ColorToJson(Value));
                VectorParamsJson.Add(MakeShared<FJsonValueObject>(JV));
            }
        }

        TArray<UTexture*> UsedTextures;
        Material->GetUsedTextures(UsedTextures, EMaterialQualityLevel::High, true, ERHIFeatureLevel::SM5, true);
        int32 UsedIndex = 0;
        for (UTexture* Texture : UsedTextures)
        {
            UTexture2D* Texture2D = Cast<UTexture2D>(Texture);
            if (!Texture2D)
            {
                continue;
            }
            const FString TextureId = ExportTexture(Texture2D, Context);
            TSharedPtr<FJsonObject> JT = MakeShared<FJsonObject>();
            JT->SetStringField(TEXT("name"), FString::Printf(TEXT("USED_%03d"), UsedIndex));
            JT->SetStringField(TEXT("texture_id"), TextureId);
            JT->SetStringField(TEXT("texture_name"), Texture2D->GetName());
            JT->SetStringField(TEXT("texture_asset_path"), Texture2D->GetPathName());
            TextureParamsJson.Add(MakeShared<FJsonValueObject>(JT));
            ClassifyTextureSlot(Texture2D->GetName().ToUpper(), Texture2D->GetName().ToUpper(), Texture2D,
                BaseColorTexId, NormalTexId, OrmTexId, RoughTexId, MetalTexId, OcclusionTexId, EmissiveTexId, TextureId);
            ++UsedIndex;
        }

        const FString MaterialNameUpper = Material->GetName().ToUpper();
        const FString MaterialPathUpper = Material->GetPathName().ToUpper();
        const FString BaseMaterialPathUpper = Material->GetMaterial() ? Material->GetMaterial()->GetPathName().ToUpper() : FString();
        const bool bMaterialLooksExplicitlyEmissive = MaterialNameUpper.Contains(TEXT("EMISS"))
            || MaterialPathUpper.Contains(TEXT("EMISS"))
            || BaseMaterialPathUpper.Contains(TEXT("EMISS"));
        if (!EmissiveTexId.IsEmpty())
        {
            int32 TextureIndex = INDEX_NONE;
            for (int32 TextureIt = 0; TextureIt < Context.TexturesJson.Num(); ++TextureIt)
            {
                const TSharedPtr<FJsonObject> TextureJson = Context.TexturesJson[TextureIt]->AsObject();
                if (TextureJson.IsValid() && TextureJson->GetStringField(TEXT("stable_id")) == EmissiveTexId)
                {
                    TextureIndex = TextureIt;
                    break;
                }
            }

            bool bKeepEmissive = bMaterialLooksExplicitlyEmissive;
            if (Context.TexturesJson.IsValidIndex(TextureIndex))
            {
                const TSharedPtr<FJsonObject> EmissiveTextureJson = Context.TexturesJson[TextureIndex]->AsObject();
                if (EmissiveTextureJson.IsValid())
                {
                    const FString EmissiveNameUpper = EmissiveTextureJson->GetStringField(TEXT("name")).ToUpper();
                    const FString EmissivePathUpper = EmissiveTextureJson->GetStringField(TEXT("asset_path")).ToUpper();
                    bKeepEmissive = bKeepEmissive || LooksLikeEmissiveTexture(EmissiveNameUpper) || LooksLikeEmissiveTexture(EmissivePathUpper);
                }
            }
            if (!bKeepEmissive)
            {
                EmissiveTexId.Reset();
            }
        }

        if (!BaseColorTexId.IsEmpty()) JM->SetStringField(TEXT("base_color_texture_id"), BaseColorTexId);
        if (!NormalTexId.IsEmpty()) JM->SetStringField(TEXT("normal_texture_id"), NormalTexId);
        if (!OrmTexId.IsEmpty()) JM->SetStringField(TEXT("orm_texture_id"), OrmTexId);
        if (!RoughTexId.IsEmpty()) JM->SetStringField(TEXT("roughness_texture_id"), RoughTexId);
        if (!MetalTexId.IsEmpty()) JM->SetStringField(TEXT("metallic_texture_id"), MetalTexId);
        if (!OcclusionTexId.IsEmpty()) JM->SetStringField(TEXT("occlusion_texture_id"), OcclusionTexId);
        if (!EmissiveTexId.IsEmpty()) JM->SetStringField(TEXT("emissive_texture_id"), EmissiveTexId);

        JM->SetArrayField(TEXT("texture_parameters"), TextureParamsJson);
        JM->SetArrayField(TEXT("scalar_parameters"), ScalarParamsJson);
        JM->SetArrayField(TEXT("vector_parameters"), VectorParamsJson);

        Context.MaterialKeyToId.Add(Key, MaterialId);
        Context.MaterialsJson.Add(MakeShared<FJsonValueObject>(JM));
        return MaterialId;
    }

    static FString GetMaterialSlotName(const UStaticMesh* Mesh, int32 MaterialSlotIndex)
    {
        if (!Mesh)
        {
            return FString();
        }
        const TArray<FStaticMaterial>& Mats = Mesh->GetStaticMaterials();
        return Mats.IsValidIndex(MaterialSlotIndex) ? Mats[MaterialSlotIndex].MaterialSlotName.ToString() : FString();
    }

    static bool HasVertexColors(const FColorVertexBuffer* ColorBuffer)
    {
        return ColorBuffer && ColorBuffer->GetNumVertices() > 0;
    }

    static const FColorVertexBuffer* GetComponentOverrideVertexColors(const UStaticMeshComponent* Component, int32 LODIndex)
    {
        if (!Component || !Component->LODData.IsValidIndex(LODIndex))
        {
            return nullptr;
        }

        const FStaticMeshComponentLODInfo& LODInfo = Component->LODData[LODIndex];
        return HasVertexColors(LODInfo.OverrideVertexColors) ? LODInfo.OverrideVertexColors : nullptr;
    }

    static bool ComponentHasVertexColorOverrides(const UStaticMeshComponent* Component, int32 LODCount)
    {
        if (!Component)
        {
            return false;
        }

        for (int32 LODIndex = 0; LODIndex < LODCount; ++LODIndex)
        {
            if (GetComponentOverrideVertexColors(Component, LODIndex))
            {
                return true;
            }
        }
        return false;
    }

    static FColor ResolveVertexColor(const FColorVertexBuffer* OverrideColorBuffer, const FColorVertexBuffer& AssetColorBuffer, uint32 VertexIndex)
    {
        if (HasVertexColors(OverrideColorBuffer) && VertexIndex < static_cast<uint32>(OverrideColorBuffer->GetNumVertices()))
        {
            return OverrideColorBuffer->VertexColor(VertexIndex);
        }
        return (AssetColorBuffer.GetNumVertices() > 0 && VertexIndex < static_cast<uint32>(AssetColorBuffer.GetNumVertices()))
            ? AssetColorBuffer.VertexColor(VertexIndex)
            : FColor::White;
    }

    static FString GetVertexColorSourceString(bool bHasVertexColors, bool bUsesComponentVertexColors)
    {
        if (bUsesComponentVertexColors)
        {
            return TEXT("component_override");
        }
        return bHasVertexColors ? TEXT("asset") : TEXT("none");
    }

    static bool MeshAssetHasVertexColors(const FMeshAsset& Mesh)
    {
        for (const FMeshLODData& LOD : Mesh.LODs)
        {
            if (LOD.bHasVertexColors)
            {
                return true;
            }
        }
        return false;
    }

    static FString GetMeshAssetVertexColorSource(const FMeshAsset& Mesh)
    {
        bool bHasAnyVertexColors = false;
        for (const FMeshLODData& LOD : Mesh.LODs)
        {
            if (LOD.bUsesComponentVertexColors)
            {
                return TEXT("component_override");
            }
            bHasAnyVertexColors |= LOD.bHasVertexColors;
        }
        return bHasAnyVertexColors ? TEXT("asset") : TEXT("none");
    }

    static void BuildStaticMeshLODData(UStaticMesh* Mesh, const UStaticMeshComponent* SourceComponent, int32 LODIndex, const FSettings& Settings, FMeshLODData& OutLOD)
    {
        const FStaticMeshRenderData* RenderData = Mesh->GetRenderData();
        const FStaticMeshLODResources& LOD = RenderData->LODResources[LODIndex];
        const FPositionVertexBuffer& Positions = LOD.VertexBuffers.PositionVertexBuffer;
        const FStaticMeshVertexBuffer& VertexBuffer = LOD.VertexBuffers.StaticMeshVertexBuffer;
        const FColorVertexBuffer& AssetColorBuffer = LOD.VertexBuffers.ColorVertexBuffer;
        const FColorVertexBuffer* OverrideColorBuffer = GetComponentOverrideVertexColors(SourceComponent, LODIndex);
        const FIndexArrayView IndexBuffer = LOD.IndexBuffer.GetArrayView();

        OutLOD.LODIndex = LODIndex;
        OutLOD.bUsesComponentVertexColors = HasVertexColors(OverrideColorBuffer);
        OutLOD.bHasVertexColors = OutLOD.bUsesComponentVertexColors || AssetColorBuffer.GetNumVertices() > 0;
        const uint32 VertexCount = Positions.GetNumVertices();
        const uint32 UVSetCount = VertexBuffer.GetNumTexCoords();

        OutLOD.Positions.Reserve(VertexCount);
        OutLOD.Normals.Reserve(VertexCount);
        OutLOD.Tangents.Reserve(VertexCount);
        OutLOD.Colors.Reserve(VertexCount);
        OutLOD.UVSets.SetNum(UVSetCount);
        for (uint32 UVIndex = 0; UVIndex < UVSetCount; ++UVIndex)
        {
            OutLOD.UVSets[UVIndex].Reserve(VertexCount);
        }

        for (uint32 VertexIndex = 0; VertexIndex < VertexCount; ++VertexIndex)
        {
            const FVector Position = ConvertLocalPos(FVector(Positions.VertexPosition(VertexIndex)), Settings);
            const FVector Normal = ConvertLocalDir(FVector(VertexBuffer.VertexTangentZ(VertexIndex)), Settings);
            const FVector TangentX = ConvertLocalDir(FVector(VertexBuffer.VertexTangentX(VertexIndex)), Settings);

            OutLOD.Positions.Add(FVector3f(Position));
            OutLOD.Normals.Add(FVector3f(Normal.GetSafeNormal()));
            OutLOD.Tangents.Add(FVector4f(FVector3f(TangentX.GetSafeNormal()), 1.0f));
            OutLOD.Colors.Add(ResolveVertexColor(OverrideColorBuffer, AssetColorBuffer, VertexIndex));

            for (uint32 UVIndex = 0; UVIndex < UVSetCount; ++UVIndex)
            {
                OutLOD.UVSets[UVIndex].Add(VertexBuffer.GetVertexUV(VertexIndex, UVIndex));
            }
        }

        for (int32 SectionIndex = 0; SectionIndex < LOD.Sections.Num(); ++SectionIndex)
        {
            const FStaticMeshSection& Section = LOD.Sections[SectionIndex];
            FMeshSection NewSection;
            NewSection.SectionIndex = SectionIndex;
            NewSection.MaterialSlotIndex = Section.MaterialIndex;
            NewSection.MaterialSlotName = GetMaterialSlotName(Mesh, Section.MaterialIndex);
            NewSection.FirstIndex = OutLOD.Indices.Num();

            for (uint32 Tri = 0; Tri < Section.NumTriangles; ++Tri)
            {
                const uint32 BaseIndex = Section.FirstIndex + Tri * 3;
                uint32 I0 = IndexBuffer[BaseIndex + 0];
                uint32 I1 = IndexBuffer[BaseIndex + 1];
                uint32 I2 = IndexBuffer[BaseIndex + 2];
                if (Settings.bMirrorYToRightHanded)
                {
                    Swap(I1, I2);
                }
                OutLOD.Indices.Add(I0);
                OutLOD.Indices.Add(I1);
                OutLOD.Indices.Add(I2);
            }

            NewSection.IndexCount = OutLOD.Indices.Num() - NewSection.FirstIndex;
            OutLOD.Sections.Add(NewSection);
        }
    }

    static int32 GetOrCreateStaticMeshAsset(UStaticMesh* Mesh, const FString& PrimitiveType, UStaticMeshComponent* SourceComponent, FExportContext& Context)
    {
        if (!Mesh)
        {
            return INDEX_NONE;
        }

        const FStaticMeshRenderData* RenderData = Mesh->GetRenderData();
        if (!RenderData || RenderData->LODResources.Num() == 0)
        {
            return INDEX_NONE;
        }

        const int32 LODCount = Context.Settings.bExportAllStaticMeshLODs ? RenderData->LODResources.Num() : 1;
        const bool bHasComponentVertexColorOverrides = ComponentHasVertexColorOverrides(SourceComponent, LODCount);
        const FString Key = bHasComponentVertexColorOverrides
            ? FString::Printf(TEXT("static_mesh|%s|alllods=%d|vertexcolors=component|component=%s"), *Mesh->GetPathName(), Context.Settings.bExportAllStaticMeshLODs ? 1 : 0, SourceComponent ? *SourceComponent->GetPathName() : TEXT("null"))
            : FString::Printf(TEXT("static_mesh|%s|alllods=%d|vertexcolors=asset"), *Mesh->GetPathName(), Context.Settings.bExportAllStaticMeshLODs ? 1 : 0);
        if (const int32* Existing = Context.MeshKeyToIndex.Find(Key))
        {
            return *Existing;
        }

        FMeshAsset Asset;
        Asset.Key = Key;
        Asset.StableId = StableId(TEXT("mesh"), Key);
        Asset.Name = Mesh->GetName();
        Asset.PrimitiveType = PrimitiveType;
        Asset.SourceAssetPath = Mesh->GetPathName();
        Asset.MeshSpace = TEXT("scene_local");

        for (int32 LODIndex = 0; LODIndex < LODCount; ++LODIndex)
        {
            FMeshLODData LODData;
            BuildStaticMeshLODData(Mesh, SourceComponent, LODIndex, Context.Settings, LODData);
            Asset.LODs.Add(MoveTemp(LODData));
        }

        const int32 NewIndex = Context.MeshAssets.Num();
        Context.MeshAssets.Add(MoveTemp(Asset));
        Context.MeshKeyToIndex.Add(Key, NewIndex);
        return NewIndex;
    }

    static void BuildSplineMeshGeometry(USplineMeshComponent* SplineComponent, const FSettings& Settings, FMeshLODData& OutLOD)
    {
        OutLOD = FMeshLODData{};
        if (!SplineComponent || !SplineComponent->GetStaticMesh())
        {
            return;
        }

        const FStaticMeshRenderData* RenderData = SplineComponent->GetStaticMesh()->GetRenderData();
        if (!RenderData || RenderData->LODResources.Num() == 0)
        {
            return;
        }

        const FStaticMeshLODResources& LOD = RenderData->LODResources[0];
        const FPositionVertexBuffer& PositionBuffer = LOD.VertexBuffers.PositionVertexBuffer;
        const FStaticMeshVertexBuffer& VertexBuffer = LOD.VertexBuffers.StaticMeshVertexBuffer;
        const FColorVertexBuffer& AssetColorBuffer = LOD.VertexBuffers.ColorVertexBuffer;
        const FColorVertexBuffer* OverrideColorBuffer = GetComponentOverrideVertexColors(SplineComponent, 0);
        const FIndexArrayView IndexBuffer = LOD.IndexBuffer.GetArrayView();
        const uint32 VertexCount = PositionBuffer.GetNumVertices();
        if (VertexCount == 0)
        {
            return;
        }

        ESplineMeshAxis::Type ForwardAxis = SplineComponent->GetForwardAxis();
        int32 AxisIndex = 0;
        switch (ForwardAxis)
        {
            case ESplineMeshAxis::Y: AxisIndex = 1; break;
            case ESplineMeshAxis::Z: AxisIndex = 2; break;
            default: AxisIndex = 0; break;
        }

        TArray<float> AxisValues;
        AxisValues.SetNum(VertexCount);
        float MinAxis = TNumericLimits<float>::Max();
        float MaxAxis = TNumericLimits<float>::Lowest();
        for (uint32 VertexIndex = 0; VertexIndex < VertexCount; ++VertexIndex)
        {
            const FVector3f P = PositionBuffer.VertexPosition(VertexIndex);
            const float AxisValue = (AxisIndex == 0) ? P.X : ((AxisIndex == 1) ? P.Y : P.Z);
            AxisValues[VertexIndex] = AxisValue;
            MinAxis = FMath::Min(MinAxis, AxisValue);
            MaxAxis = FMath::Max(MaxAxis, AxisValue);
        }

        const float AxisLength = FMath::Max(MaxAxis - MinAxis, KINDA_SMALL_NUMBER);

        FVector StartPos = SplineComponent->GetStartPosition();
        FVector EndPos = SplineComponent->GetEndPosition();
        FVector StartTangent = SplineComponent->GetStartTangent();
        FVector EndTangent = SplineComponent->GetEndTangent();
        const FTransform& C2W = SplineComponent->GetComponentTransform();
        StartPos = C2W.TransformPosition(StartPos);
        EndPos = C2W.TransformPosition(EndPos);
        StartTangent = C2W.TransformVectorNoScale(StartTangent);
        EndTangent = C2W.TransformVectorNoScale(EndTangent);

        FVector UpDir = SplineComponent->SplineUpDir;
        if (UpDir.IsNearlyZero())
        {
            UpDir = FVector::UpVector;
        }
        UpDir.Normalize();

        auto EvalPos = [&](float Alpha) -> FVector
        {
            Alpha = FMath::Clamp(Alpha, 0.0f, 1.0f);
            const float A2 = Alpha * Alpha;
            const float A3 = A2 * Alpha;
            return (2.0f * A3 - 3.0f * A2 + 1.0f) * StartPos
                + (A3 - 2.0f * A2 + Alpha) * StartTangent
                + (-2.0f * A3 + 3.0f * A2) * EndPos
                + (A3 - A2) * EndTangent;
        };

        auto EvalTangent = [&](float Alpha) -> FVector
        {
            Alpha = FMath::Clamp(Alpha, 0.0f, 1.0f);
            const float A2 = Alpha * Alpha;
            FVector T = (6.0f * A2 - 6.0f * Alpha) * StartPos
                + (3.0f * A2 - 4.0f * Alpha + 1.0f) * StartTangent
                + (-6.0f * A2 + 6.0f * Alpha) * EndPos
                + (3.0f * A2 - 2.0f * Alpha) * EndTangent;
            return T.GetSafeNormal();
        };

        auto TransformLocalSplineVector = [&](const FVector& LocalVector, const FVector& TangentDir, const FVector& Right, const FVector& TrueUp) -> FVector
        {
            double Along = 0.0;
            double CrossU = 0.0;
            double CrossV = 0.0;
            switch (AxisIndex)
            {
                case 0: Along = LocalVector.X; CrossU = LocalVector.Y; CrossV = LocalVector.Z; break;
                case 1: Along = LocalVector.Y; CrossU = LocalVector.X; CrossV = LocalVector.Z; break;
                default: Along = LocalVector.Z; CrossU = LocalVector.X; CrossV = LocalVector.Y; break;
            }
            return Along * TangentDir + CrossU * Right + CrossV * TrueUp;
        };

        OutLOD.LODIndex = 0;
        OutLOD.bUsesComponentVertexColors = HasVertexColors(OverrideColorBuffer);
        OutLOD.bHasVertexColors = OutLOD.bUsesComponentVertexColors || AssetColorBuffer.GetNumVertices() > 0;
        OutLOD.Positions.Reserve(VertexCount);
        OutLOD.Normals.Reserve(VertexCount);
        OutLOD.Tangents.Reserve(VertexCount);
        OutLOD.Colors.Reserve(VertexCount);
        const uint32 UVSetCount = VertexBuffer.GetNumTexCoords();
        OutLOD.UVSets.SetNum(UVSetCount);
        for (uint32 UVIndex = 0; UVIndex < UVSetCount; ++UVIndex)
        {
            OutLOD.UVSets[UVIndex].Reserve(VertexCount);
        }

        for (uint32 VertexIndex = 0; VertexIndex < VertexCount; ++VertexIndex)
        {
            const FVector LocalPos = FVector(PositionBuffer.VertexPosition(VertexIndex));
            float AxisValue = 0.0f;
            float CrossU = 0.0f;
            float CrossV = 0.0f;
            switch (AxisIndex)
            {
                case 0: AxisValue = LocalPos.X; CrossU = LocalPos.Y; CrossV = LocalPos.Z; break;
                case 1: AxisValue = LocalPos.Y; CrossU = LocalPos.X; CrossV = LocalPos.Z; break;
                default: AxisValue = LocalPos.Z; CrossU = LocalPos.X; CrossV = LocalPos.Y; break;
            }

            const float Alpha = (AxisValue - MinAxis) / AxisLength;
            const FVector Center = EvalPos(Alpha);
            const FVector TangentDir = EvalTangent(Alpha);
            FVector Right = FVector::CrossProduct(UpDir, TangentDir);
            if (Right.IsNearlyZero())
            {
                Right = FVector::CrossProduct(FVector::UpVector, TangentDir);
            }
            Right.Normalize();
            const FVector TrueUp = FVector::CrossProduct(TangentDir, Right).GetSafeNormal();

            const FVector WorldPosUE = Center + CrossU * Right + CrossV * TrueUp;
            const FVector WorldPosScene = ConvertWorldPosScaled(WorldPosUE, Settings);
            OutLOD.Positions.Add(FVector3f(WorldPosScene));

            const FVector WorldNormalUE = TransformLocalSplineVector(FVector(VertexBuffer.VertexTangentZ(VertexIndex)), TangentDir, Right, TrueUp).GetSafeNormal();
            const FVector WorldTangentUE = TransformLocalSplineVector(FVector(VertexBuffer.VertexTangentX(VertexIndex)), TangentDir, Right, TrueUp).GetSafeNormal();
            OutLOD.Normals.Add(FVector3f(ConvertLocalDir(WorldNormalUE, Settings).GetSafeNormal()));
            OutLOD.Tangents.Add(FVector4f(FVector3f(ConvertLocalDir(WorldTangentUE, Settings).GetSafeNormal()), 1.0f));
            OutLOD.Colors.Add(ResolveVertexColor(OverrideColorBuffer, AssetColorBuffer, VertexIndex));

            for (uint32 UVIndex = 0; UVIndex < UVSetCount; ++UVIndex)
            {
                OutLOD.UVSets[UVIndex].Add(VertexBuffer.GetVertexUV(VertexIndex, UVIndex));
            }
        }

        for (int32 SectionIndex = 0; SectionIndex < LOD.Sections.Num(); ++SectionIndex)
        {
            const FStaticMeshSection& Section = LOD.Sections[SectionIndex];
            FMeshSection NewSection;
            NewSection.SectionIndex = SectionIndex;
            NewSection.MaterialSlotIndex = Section.MaterialIndex;
            NewSection.MaterialSlotName = GetMaterialSlotName(SplineComponent->GetStaticMesh(), Section.MaterialIndex);
            NewSection.FirstIndex = OutLOD.Indices.Num();

            for (uint32 Tri = 0; Tri < Section.NumTriangles; ++Tri)
            {
                const uint32 BaseIndex = Section.FirstIndex + Tri * 3;
                uint32 I0 = IndexBuffer[BaseIndex + 0];
                uint32 I1 = IndexBuffer[BaseIndex + 1];
                uint32 I2 = IndexBuffer[BaseIndex + 2];
                if (Settings.bMirrorYToRightHanded)
                {
                    Swap(I1, I2);
                }
                OutLOD.Indices.Add(I0);
                OutLOD.Indices.Add(I1);
                OutLOD.Indices.Add(I2);
            }

            NewSection.IndexCount = OutLOD.Indices.Num() - NewSection.FirstIndex;
            OutLOD.Sections.Add(NewSection);
        }
    }

    static int32 CreateSplineMeshAsset(USplineMeshComponent* SplineComponent, FExportContext& Context)
    {
        const FString Key = FString::Printf(TEXT("spline_mesh|%s"), *SplineComponent->GetPathName());
        if (const int32* Existing = Context.MeshKeyToIndex.Find(Key))
        {
            return *Existing;
        }

        FMeshLODData LOD;
        BuildSplineMeshGeometry(SplineComponent, Context.Settings, LOD);
        if (LOD.Positions.Num() == 0)
        {
            return INDEX_NONE;
        }

        FMeshAsset Asset;
        Asset.Key = Key;
        Asset.StableId = StableId(TEXT("mesh"), Key);
        Asset.Name = SplineComponent->GetName();
        Asset.PrimitiveType = TEXT("spline_mesh");
        Asset.SourceAssetPath = SplineComponent->GetStaticMesh() ? SplineComponent->GetStaticMesh()->GetPathName() : FString();
        Asset.MeshSpace = TEXT("scene_world_baked");
        Asset.LODs.Add(MoveTemp(LOD));

        const int32 NewIndex = Context.MeshAssets.Num();
        Context.MeshAssets.Add(MoveTemp(Asset));
        Context.MeshKeyToIndex.Add(Key, NewIndex);
        return NewIndex;
    }

    static int32 CreateLandscapeMeshAsset(ULandscapeComponent* LandscapeComponent, FExportContext& Context)
    {
        const FString Key = FString::Printf(TEXT("landscape_component|%s"), *LandscapeComponent->GetPathName());
        if (const int32* Existing = Context.MeshKeyToIndex.Find(Key))
        {
            return *Existing;
        }

        const int32 VertsPerSide = LandscapeComponent->ComponentSizeQuads + 1;
        if (VertsPerSide < 2)
        {
            return INDEX_NONE;
        }

        FLandscapeComponentDataInterface DataInterface(LandscapeComponent, 0);
        FMeshLODData LOD;
        LOD.LODIndex = 0;
        LOD.Positions.SetNum(VertsPerSide * VertsPerSide);
        LOD.Normals.SetNum(VertsPerSide * VertsPerSide);
        LOD.Tangents.SetNum(VertsPerSide * VertsPerSide);
        LOD.Colors.Init(FColor::White, VertsPerSide * VertsPerSide);
        LOD.bHasVertexColors = false;
        LOD.bUsesComponentVertexColors = false;
        LOD.UVSets.SetNum(1);
        LOD.UVSets[0].SetNum(VertsPerSide * VertsPerSide);

        auto IndexOf = [VertsPerSide](int32 X, int32 Y) -> int32
        {
            return Y * VertsPerSide + X;
        };

        for (int32 Y = 0; Y < VertsPerSide; ++Y)
        {
            for (int32 X = 0; X < VertsPerSide; ++X)
            {
                const int32 Index = IndexOf(X, Y);
                const FVector WorldPosScene = ConvertWorldPosScaled(DataInterface.GetWorldVertex(X, Y), Context.Settings);
                LOD.Positions[Index] = FVector3f(WorldPosScene);
                LOD.UVSets[0][Index] = FVector2f(static_cast<float>(X) / static_cast<float>(VertsPerSide - 1), static_cast<float>(Y) / static_cast<float>(VertsPerSide - 1));
            }
        }

        for (int32 Y = 0; Y < VertsPerSide; ++Y)
        {
            for (int32 X = 0; X < VertsPerSide; ++X)
            {
                const int32 Index = IndexOf(X, Y);
                const int32 X0 = FMath::Max(0, X - 1);
                const int32 X1 = FMath::Min(VertsPerSide - 1, X + 1);
                const int32 Y0 = FMath::Max(0, Y - 1);
                const int32 Y1 = FMath::Min(VertsPerSide - 1, Y + 1);

                const FVector PX0 = FVector(LOD.Positions[IndexOf(X0, Y)]);
                const FVector PX1 = FVector(LOD.Positions[IndexOf(X1, Y)]);
                const FVector PY0 = FVector(LOD.Positions[IndexOf(X, Y0)]);
                const FVector PY1 = FVector(LOD.Positions[IndexOf(X, Y1)]);
                const FVector TangentX = (PX1 - PX0).GetSafeNormal();
                const FVector TangentY = (PY1 - PY0).GetSafeNormal();
                FVector Normal = FVector::CrossProduct(TangentY, TangentX).GetSafeNormal();
                if (Normal.IsNearlyZero())
                {
                    Normal = FVector::UpVector;
                }
                LOD.Normals[Index] = FVector3f(Normal);
                LOD.Tangents[Index] = FVector4f(FVector3f(TangentX), 1.0f);
            }
        }

        FMeshSection Section;
        Section.SectionIndex = 0;
        Section.MaterialSlotIndex = 0;
        Section.FirstIndex = 0;
        Section.MaterialSlotName = TEXT("Landscape");

        for (int32 Y = 0; Y < LandscapeComponent->ComponentSizeQuads; ++Y)
        {
            for (int32 X = 0; X < LandscapeComponent->ComponentSizeQuads; ++X)
            {
                const uint32 I00 = IndexOf(X, Y);
                const uint32 I10 = IndexOf(X + 1, Y);
                const uint32 I01 = IndexOf(X, Y + 1);
                const uint32 I11 = IndexOf(X + 1, Y + 1);
                if (Context.Settings.bMirrorYToRightHanded)
                {
                    LOD.Indices.Add(I00); LOD.Indices.Add(I01); LOD.Indices.Add(I10);
                    LOD.Indices.Add(I10); LOD.Indices.Add(I01); LOD.Indices.Add(I11);
                }
                else
                {
                    LOD.Indices.Add(I00); LOD.Indices.Add(I10); LOD.Indices.Add(I01);
                    LOD.Indices.Add(I10); LOD.Indices.Add(I11); LOD.Indices.Add(I01);
                }
            }
        }
        Section.IndexCount = LOD.Indices.Num();
        LOD.Sections.Add(Section);

        FMeshAsset Asset;
        Asset.Key = Key;
        Asset.StableId = StableId(TEXT("mesh"), Key);
        Asset.Name = LandscapeComponent->GetName();
        Asset.PrimitiveType = TEXT("landscape_component");
        Asset.SourceAssetPath = LandscapeComponent->GetLandscapeProxy() ? LandscapeComponent->GetLandscapeProxy()->GetPathName() : FString();
        Asset.MeshSpace = TEXT("scene_world_baked");
        Asset.LODs.Add(MoveTemp(LOD));

        const int32 NewIndex = Context.MeshAssets.Num();
        Context.MeshAssets.Add(MoveTemp(Asset));
        Context.MeshKeyToIndex.Add(Key, NewIndex);
        return NewIndex;
    }

    static FLinearColor GetFogInscatteringColorCompat(const UExponentialHeightFogComponent* Fog)
    {
        if (!Fog)
        {
            return FLinearColor::White;
        }

        if (const FStructProperty* Prop = FindFProperty<FStructProperty>(Fog->GetClass(), TEXT("FogInscatteringLuminance")))
        {
            if (Prop->Struct == TBaseStructure<FLinearColor>::Get())
            {
                if (const FLinearColor* Value = Prop->ContainerPtrToValuePtr<FLinearColor>(Fog))
                {
                    return *Value;
                }
            }
        }

        if (const FStructProperty* Prop = FindFProperty<FStructProperty>(Fog->GetClass(), TEXT("FogInscatteringColor")))
        {
            if (Prop->Struct == TBaseStructure<FLinearColor>::Get())
            {
                if (const FLinearColor* Value = Prop->ContainerPtrToValuePtr<FLinearColor>(Fog))
                {
                    return *Value;
                }
            }
        }

        return FLinearColor::White;
    }

    static TArray<TSharedPtr<FJsonValue>> BuildSectionMaterialMapJson(const TArray<FMeshSection>& Sections, const UMeshComponent* MeshComponent, FExportContext& Context)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        for (const FMeshSection& Section : Sections)
        {
            TSharedPtr<FJsonObject> JS = MakeShared<FJsonObject>();
            JS->SetNumberField(TEXT("section_index"), Section.SectionIndex);
            JS->SetNumberField(TEXT("material_slot_index"), Section.MaterialSlotIndex);
            JS->SetNumberField(TEXT("first_index"), Section.FirstIndex);
            JS->SetNumberField(TEXT("index_count"), Section.IndexCount);
            if (!Section.MaterialSlotName.IsEmpty())
            {
                JS->SetStringField(TEXT("material_slot_name"), Section.MaterialSlotName);
            }
            if (MeshComponent)
            {
                if (UMaterialInterface* Material = MeshComponent->GetMaterial(Section.MaterialSlotIndex))
                {
                    JS->SetStringField(TEXT("material_id"), GetOrCreateMaterialId(Material, Context));
                    JS->SetStringField(TEXT("material_name"), Material->GetName());
                    JS->SetStringField(TEXT("material_asset_path"), Material->GetPathName());
                }
            }
            Out.Add(MakeShared<FJsonValueObject>(JS));
        }
        return Out;
    }

    static TArray<TSharedPtr<FJsonValue>> BuildSectionMaterialMapJson(const TArray<FMeshSection>& Sections, const ULandscapeComponent* LandscapeComponent, FExportContext& Context)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        for (const FMeshSection& Section : Sections)
        {
            TSharedPtr<FJsonObject> JS = MakeShared<FJsonObject>();
            JS->SetNumberField(TEXT("section_index"), Section.SectionIndex);
            JS->SetNumberField(TEXT("material_slot_index"), Section.MaterialSlotIndex);
            JS->SetNumberField(TEXT("first_index"), Section.FirstIndex);
            JS->SetNumberField(TEXT("index_count"), Section.IndexCount);
            if (!Section.MaterialSlotName.IsEmpty())
            {
                JS->SetStringField(TEXT("material_slot_name"), Section.MaterialSlotName);
            }
            if (LandscapeComponent)
            {
                if (UMaterialInterface* Material = LandscapeComponent->GetMaterial(Section.MaterialSlotIndex))
                {
                    JS->SetStringField(TEXT("material_id"), GetOrCreateMaterialId(Material, Context));
                    JS->SetStringField(TEXT("material_name"), Material->GetName());
                    JS->SetStringField(TEXT("material_asset_path"), Material->GetPathName());
                }
            }
            Out.Add(MakeShared<FJsonValueObject>(JS));
        }
        return Out;
    }

    static void AddActorJson(AActor* Actor, FExportContext& Context)
    {
        TSharedPtr<FJsonObject> JA = MakeShared<FJsonObject>();
        JA->SetStringField(TEXT("stable_id"), StableId(TEXT("actor"), Actor->GetPathName()));
        JA->SetStringField(TEXT("name"), Actor->GetName());
        JA->SetStringField(TEXT("actor_label"), Actor->GetActorNameOrLabel());
        JA->SetStringField(TEXT("actor_path"), Actor->GetPathName());
        JA->SetStringField(TEXT("class"), Actor->GetClass()->GetPathName());
        JA->SetBoolField(TEXT("hidden"), Actor->IsHidden());
        JA->SetStringField(TEXT("level_path"), Actor->GetLevel() ? Actor->GetLevel()->GetPathName() : FString());
        JA->SetStringField(TEXT("folder_path"), Actor->GetFolderPath().ToString());
        if (AActor* AttachParent = Actor->GetAttachParentActor())
        {
            JA->SetStringField(TEXT("attach_parent_actor_path"), AttachParent->GetPathName());
        }

        const FBoxSphereBounds EmptyBounds(FVector::ZeroVector, FVector::ZeroVector, 0.0f);
        const FBoxSphereBounds Bounds = GetSceneBoundsFromWorldBounds(Actor->GetRootComponent() ? Actor->GetRootComponent()->Bounds : EmptyBounds, Context.Settings);
        JA->SetObjectField(TEXT("bounds"), BoundsToJson(Bounds));
        if (Context.Settings.bExportActorTags)
        {
            JA->SetArrayField(TEXT("tags"), NameArrayToJson(Actor->Tags));
        }
#if WITH_EDITOR
        if (Context.Settings.bExportDataLayers)
        {
            JA->SetArrayField(TEXT("data_layers"), NameArrayToJson(Actor->GetDataLayerInstanceNames()));
        }
#endif

        TArray<UActorComponent*> Components;
        Actor->GetComponents(Components);
        TArray<TSharedPtr<FJsonValue>> ComponentsJson;
        for (UActorComponent* Component : Components)
        {
            if (!Component)
            {
                continue;
            }
            TSharedPtr<FJsonObject> JC = MakeShared<FJsonObject>();
            JC->SetStringField(TEXT("name"), Component->GetName());
            JC->SetStringField(TEXT("readable_name"), GetComponentReadableExportName(Component));
            JC->SetStringField(TEXT("component_path"), Component->GetPathName());
            JC->SetStringField(TEXT("class"), Component->GetClass()->GetPathName());
            if (const USceneComponent* Scene = Cast<USceneComponent>(Component))
            {
                JC->SetBoolField(TEXT("visible"), Scene->IsVisible());
                JC->SetArrayField(TEXT("relative_location"), Vec3ToJson(Scene->GetRelativeLocation()));
                JC->SetArrayField(TEXT("relative_rotation_euler"), Vec3ToJson(Scene->GetRelativeRotation().Euler()));
                JC->SetArrayField(TEXT("relative_scale"), Vec3ToJson(Scene->GetRelativeScale3D()));
                JC->SetArrayField(TEXT("transform_matrix"), MatrixToJson(MakeSceneTransformMatrix(Scene->GetComponentTransform(), Context.Settings)));
            }
            if (const UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Component))
            {
                JC->SetBoolField(TEXT("hidden_in_game"), Primitive->bHiddenInGame);
                JC->SetBoolField(TEXT("cast_shadow"), Primitive->CastShadow);
                JC->SetNumberField(TEXT("mobility"), static_cast<int32>(Primitive->Mobility));
                JC->SetNumberField(TEXT("collision_enabled"), static_cast<int32>(Primitive->GetCollisionEnabled()));
            }
            ComponentsJson.Add(MakeShared<FJsonValueObject>(JC));
        }
        JA->SetArrayField(TEXT("components"), ComponentsJson);
        Context.ActorsJson.Add(MakeShared<FJsonValueObject>(JA));
    }

    static void AddPrimitiveJsonCommon(
        UPrimitiveComponent* Primitive,
        const FString& PrimitiveType,
        const FString& SourceAssetPath,
        const FString& MeshAssetId,
        const TArray<TArray<double>>& InstanceTransforms,
        const TArray<TSharedPtr<FJsonValue>>& SectionsJson,
        bool bGeometryExported,
        FExportContext& Context,
        const FString& MeshSpace = FString(),
        bool bHasVertexColors = false,
        FString VertexColorSource = TEXT("none"))
    {
        TSharedPtr<FJsonObject> JP = MakeShared<FJsonObject>();
        JP->SetStringField(TEXT("stable_id"), StableId(TEXT("prim"), Primitive->GetPathName()));
        JP->SetStringField(TEXT("name"), MakeUniqueExportName(Primitive->GetOwner(), Primitive));
        JP->SetStringField(TEXT("raw_name"), Primitive->GetName());
        JP->SetStringField(TEXT("primitive_type"), PrimitiveType);
        if (Primitive->GetOwner())
        {
            JP->SetStringField(TEXT("actor_name"), Primitive->GetOwner()->GetName());
            JP->SetStringField(TEXT("actor_label"), Primitive->GetOwner()->GetActorNameOrLabel());
        }
        JP->SetStringField(TEXT("actor_path"), Primitive->GetOwner() ? Primitive->GetOwner()->GetPathName() : FString());
        JP->SetStringField(TEXT("component_path"), Primitive->GetPathName());
        JP->SetStringField(TEXT("readable_name"), Primitive->GetReadableName());
        JP->SetStringField(TEXT("component_class"), Primitive->GetClass()->GetPathName());
        JP->SetStringField(TEXT("source_asset_path"), SourceAssetPath);
        JP->SetBoolField(TEXT("geometry_exported"), bGeometryExported);
        JP->SetBoolField(TEXT("visible"), Primitive->IsVisible());
        JP->SetBoolField(TEXT("hidden_in_game"), Primitive->bHiddenInGame);
        JP->SetBoolField(TEXT("cast_shadow"), Primitive->CastShadow);
        JP->SetNumberField(TEXT("mobility"), static_cast<int32>(Primitive->Mobility));
        JP->SetNumberField(TEXT("collision_enabled"), static_cast<int32>(Primitive->GetCollisionEnabled()));
        JP->SetObjectField(TEXT("bounds"), BoundsToJson(GetSceneBoundsFromWorldBounds(Primitive->Bounds, Context.Settings)));
        if (!MeshAssetId.IsEmpty())
        {
            JP->SetStringField(TEXT("mesh_asset_id"), MeshAssetId);
        }
        if (!MeshSpace.IsEmpty())
        {
            JP->SetStringField(TEXT("mesh_space"), MeshSpace);
        }
        JP->SetBoolField(TEXT("has_vertex_colors"), bHasVertexColors);
        JP->SetStringField(TEXT("vertex_color_source"), VertexColorSource.IsEmpty() ? TEXT("none") : VertexColorSource);

        TArray<TSharedPtr<FJsonValue>> InstancesJson;
        for (const TArray<double>& Matrix : InstanceTransforms)
        {
            TSharedPtr<FJsonObject> JI = MakeShared<FJsonObject>();
            JI->SetArrayField(TEXT("matrix"), MatrixToJson(Matrix));
            InstancesJson.Add(MakeShared<FJsonValueObject>(JI));
        }
        JP->SetArrayField(TEXT("instance_transforms"), InstancesJson);
        JP->SetNumberField(TEXT("instance_count"), InstanceTransforms.Num());
        JP->SetArrayField(TEXT("sections"), SectionsJson);
        if (Primitive->GetOwner() && Context.Settings.bExportActorTags)
        {
            JP->SetArrayField(TEXT("actor_tags"), NameArrayToJson(Primitive->GetOwner()->Tags));
        }
#if WITH_EDITOR
        if (Primitive->GetOwner() && Context.Settings.bExportDataLayers)
        {
            JP->SetArrayField(TEXT("data_layers"), NameArrayToJson(Primitive->GetOwner()->GetDataLayerInstanceNames()));
        }
#endif
        Context.PrimitivesJson.Add(MakeShared<FJsonValueObject>(JP));
    }

    static void AddStaticMeshPrimitive(UStaticMeshComponent* Component, FExportContext& Context)
    {
        if (!Component || !Component->GetStaticMesh())
        {
            return;
        }

        if (USplineMeshComponent* SplineComponent = Cast<USplineMeshComponent>(Component))
        {
            if (!Context.Settings.bExportSplineMeshGeometry)
            {
                AddPrimitiveJsonCommon(Component, TEXT("spline_mesh"), Component->GetStaticMesh()->GetPathName(), FString(),
                    { MakeSceneTransformMatrix(Component->GetComponentTransform(), Context.Settings) }, {}, false, Context,
                    FString(), false, TEXT("none"));
                return;
            }

            const int32 MeshIndex = CreateSplineMeshAsset(SplineComponent, Context);
            const FString MeshAssetId = MeshIndex != INDEX_NONE ? Context.MeshAssets[MeshIndex].StableId : FString();
            const TArray<TSharedPtr<FJsonValue>> SectionsJson = MeshIndex != INDEX_NONE
                ? BuildSectionMaterialMapJson(Context.MeshAssets[MeshIndex].LODs[0].Sections, Component, Context)
                : TArray<TSharedPtr<FJsonValue>>();

            const bool bHasVertexColors = MeshIndex != INDEX_NONE ? MeshAssetHasVertexColors(Context.MeshAssets[MeshIndex]) : false;
            const FString VertexColorSource = MeshIndex != INDEX_NONE ? GetMeshAssetVertexColorSource(Context.MeshAssets[MeshIndex]) : TEXT("none");
            AddPrimitiveJsonCommon(Component, TEXT("spline_mesh"), Component->GetStaticMesh()->GetPathName(), MeshAssetId,
                { IdentityMatrixJson() }, SectionsJson, MeshIndex != INDEX_NONE, Context, TEXT("scene_world_baked"), bHasVertexColors, VertexColorSource);
            return;
        }

        const FString PrimitiveType = Cast<UHierarchicalInstancedStaticMeshComponent>(Component) ? TEXT("hierarchical_instanced_static_mesh")
            : Cast<UInstancedStaticMeshComponent>(Component) ? TEXT("instanced_static_mesh")
            : TEXT("static_mesh");

        const int32 MeshIndex = GetOrCreateStaticMeshAsset(Component->GetStaticMesh(), PrimitiveType, Component, Context);
        if (MeshIndex == INDEX_NONE)
        {
            return;
        }

        TArray<TArray<double>> InstanceTransforms;
        if (UInstancedStaticMeshComponent* ISM = Cast<UInstancedStaticMeshComponent>(Component))
        {
            const int32 Count = ISM->GetInstanceCount();
            InstanceTransforms.Reserve(Count);
            for (int32 InstanceIndex = 0; InstanceIndex < Count; ++InstanceIndex)
            {
                FTransform InstanceTransform;
                ISM->GetInstanceTransform(InstanceIndex, InstanceTransform, true);
                InstanceTransforms.Add(MakeSceneTransformMatrix(InstanceTransform, Context.Settings));
            }
        }
        else
        {
            InstanceTransforms.Add(MakeSceneTransformMatrix(Component->GetComponentTransform(), Context.Settings));
        }

        const TArray<TSharedPtr<FJsonValue>> SectionsJson = Context.MeshAssets[MeshIndex].LODs.Num() > 0
            ? BuildSectionMaterialMapJson(Context.MeshAssets[MeshIndex].LODs[0].Sections, Component, Context)
            : TArray<TSharedPtr<FJsonValue>>();

        const bool bHasVertexColors = MeshAssetHasVertexColors(Context.MeshAssets[MeshIndex]);
        const FString VertexColorSource = GetMeshAssetVertexColorSource(Context.MeshAssets[MeshIndex]);
        AddPrimitiveJsonCommon(Component, PrimitiveType, Component->GetStaticMesh()->GetPathName(), Context.MeshAssets[MeshIndex].StableId,
            InstanceTransforms, SectionsJson, true, Context, Context.MeshAssets[MeshIndex].MeshSpace, bHasVertexColors, VertexColorSource);
    }

    static void AddSkeletalPrimitive(USkeletalMeshComponent* Component, FExportContext& Context)
    {
        if (!Component || !Context.Settings.bExportSkeletalMeshMetadata)
        {
            return;
        }

        TArray<TSharedPtr<FJsonValue>> SectionsJson;
        const int32 MaterialCount = Component->GetNumMaterials();
        for (int32 MaterialIndex = 0; MaterialIndex < MaterialCount; ++MaterialIndex)
        {
            TSharedPtr<FJsonObject> JS = MakeShared<FJsonObject>();
            JS->SetNumberField(TEXT("section_index"), MaterialIndex);
            JS->SetNumberField(TEXT("material_slot_index"), MaterialIndex);
            if (UMaterialInterface* Material = Component->GetMaterial(MaterialIndex))
            {
                JS->SetStringField(TEXT("material_id"), GetOrCreateMaterialId(Material, Context));
                JS->SetStringField(TEXT("material_name"), Material->GetName());
                JS->SetStringField(TEXT("material_asset_path"), Material->GetPathName());
            }
            SectionsJson.Add(MakeShared<FJsonValueObject>(JS));
        }

        AddPrimitiveJsonCommon(Component, TEXT("skeletal_mesh"),
            Component->GetSkeletalMeshAsset() ? Component->GetSkeletalMeshAsset()->GetPathName() : FString(),
            FString(), { MakeSceneTransformMatrix(Component->GetComponentTransform(), Context.Settings) }, SectionsJson, false, Context,
            FString(), false, TEXT("none"));

        if (Component->GetSkeletalMeshAsset())
        {
            TSharedPtr<FJsonObject> JSkel = Context.PrimitivesJson.Last()->AsObject();
            JSkel->SetNumberField(TEXT("bone_count"), Component->GetSkeletalMeshAsset()->GetRefSkeleton().GetNum());
            JSkel->SetNumberField(TEXT("socket_count"), Component->GetSkeletalMeshAsset()->GetActiveSocketList().Num());
        }
    }

    static void AddLandscapePrimitive(ULandscapeComponent* Component, FExportContext& Context)
    {
        if (!Component)
        {
            return;
        }

        const int32 MeshIndex = Context.Settings.bExportLandscapeGeometry ? CreateLandscapeMeshAsset(Component, Context) : INDEX_NONE;
        const FString MeshAssetId = MeshIndex != INDEX_NONE ? Context.MeshAssets[MeshIndex].StableId : FString();
        const TArray<TSharedPtr<FJsonValue>> SectionsJson = MeshIndex != INDEX_NONE
            ? BuildSectionMaterialMapJson(Context.MeshAssets[MeshIndex].LODs[0].Sections, Component, Context)
            : TArray<TSharedPtr<FJsonValue>>();

        AddPrimitiveJsonCommon(Component, TEXT("landscape_component"),
            Component->GetLandscapeProxy() ? Component->GetLandscapeProxy()->GetPathName() : FString(),
            MeshAssetId, { IdentityMatrixJson() }, SectionsJson, MeshIndex != INDEX_NONE, Context, TEXT("scene_world_baked"),
            false, TEXT("none"));

        TSharedPtr<FJsonObject> JL = MakeShared<FJsonObject>();
        JL->SetStringField(TEXT("stable_id"), StableId(TEXT("landscape"), Component->GetPathName()));
        JL->SetStringField(TEXT("name"), Component->GetName());
        JL->SetStringField(TEXT("component_path"), Component->GetPathName());
        JL->SetStringField(TEXT("landscape_actor_path"), Component->GetLandscapeProxy() ? Component->GetLandscapeProxy()->GetPathName() : FString());
        JL->SetNumberField(TEXT("component_size_quads"), Component->ComponentSizeQuads);
        JL->SetNumberField(TEXT("subsection_size_quads"), Component->SubsectionSizeQuads);
        JL->SetNumberField(TEXT("num_subsections"), Component->NumSubsections);
        JL->SetNumberField(TEXT("section_base_x"), Component->SectionBaseX);
        JL->SetNumberField(TEXT("section_base_y"), Component->SectionBaseY);
        UTexture2D* HeightmapTexture = Component->GetHeightmap(false);
        JL->SetStringField(TEXT("heightmap_texture_path"), HeightmapTexture ? HeightmapTexture->GetPathName() : FString());
        TArray<TSharedPtr<FJsonValue>> Weightmaps;
        const TArray<UTexture2D*>& WeightmapTextures = Component->GetWeightmapTextures(false);
        for (UTexture2D* Weightmap : WeightmapTextures)
        {
            if (!Weightmap)
            {
                continue;
            }
            TSharedPtr<FJsonObject> JW = MakeShared<FJsonObject>();
            JW->SetStringField(TEXT("texture_id"), ExportTexture(Weightmap, Context));
            JW->SetStringField(TEXT("asset_path"), Weightmap->GetPathName());
            Weightmaps.Add(MakeShared<FJsonValueObject>(JW));
        }
        JL->SetArrayField(TEXT("weightmaps"), Weightmaps);
        if (UMaterialInterface* Material = Component->GetMaterial(0))
        {
            JL->SetStringField(TEXT("material_id"), GetOrCreateMaterialId(Material, Context));
            JL->SetStringField(TEXT("material_asset_path"), Material->GetPathName());
        }
        Context.LandscapesJson.Add(MakeShared<FJsonValueObject>(JL));
    }

    static void AddLightJson(ULightComponent* Light, FExportContext& Context)
    {
        if (!Light || !Context.Settings.bExportLights)
        {
            return;
        }
        TSharedPtr<FJsonObject> JL = MakeShared<FJsonObject>();
        JL->SetStringField(TEXT("stable_id"), StableId(TEXT("light"), Light->GetPathName()));
        JL->SetStringField(TEXT("name"), Light->GetName());
        JL->SetStringField(TEXT("actor_path"), Light->GetOwner() ? Light->GetOwner()->GetPathName() : FString());
        JL->SetStringField(TEXT("component_path"), Light->GetPathName());
        JL->SetStringField(TEXT("class"), Light->GetClass()->GetPathName());
        JL->SetArrayField(TEXT("transform_matrix"), MatrixToJson(MakeSceneTransformMatrix(Light->GetComponentTransform(), Context.Settings)));
        JL->SetArrayField(TEXT("color"), ColorToJson(Light->GetLightColor()));
        JL->SetNumberField(TEXT("intensity"), Light->Intensity);
        JL->SetBoolField(TEXT("cast_shadows"), Light->CastShadows);
        JL->SetNumberField(TEXT("mobility"), static_cast<int32>(Light->Mobility));
        JL->SetBoolField(TEXT("use_temperature"), Light->bUseTemperature);
        JL->SetNumberField(TEXT("temperature"), Light->Temperature);

        FString LightType = TEXT("light");
        if (Cast<UDirectionalLightComponent>(Light)) LightType = TEXT("directional");
        else if (Cast<USpotLightComponent>(Light)) LightType = TEXT("spot");
        else if (Cast<URectLightComponent>(Light)) LightType = TEXT("rect");
        else if (Cast<USkyLightComponent>(Light)) LightType = TEXT("sky");
        else if (Cast<UPointLightComponent>(Light)) LightType = TEXT("point");
        JL->SetStringField(TEXT("light_type"), LightType);

        if (const UPointLightComponent* Point = Cast<UPointLightComponent>(Light))
        {
            JL->SetNumberField(TEXT("attenuation_radius"), Point->AttenuationRadius * Context.Settings.UnitScale);
            JL->SetNumberField(TEXT("source_radius"), Point->SourceRadius * Context.Settings.UnitScale);
            JL->SetNumberField(TEXT("soft_source_radius"), Point->SoftSourceRadius * Context.Settings.UnitScale);
            JL->SetNumberField(TEXT("source_length"), Point->SourceLength * Context.Settings.UnitScale);
        }
        if (const USpotLightComponent* Spot = Cast<USpotLightComponent>(Light))
        {
            JL->SetNumberField(TEXT("inner_cone_angle"), Spot->InnerConeAngle);
            JL->SetNumberField(TEXT("outer_cone_angle"), Spot->OuterConeAngle);
        }
        if (const URectLightComponent* Rect = Cast<URectLightComponent>(Light))
        {
            JL->SetNumberField(TEXT("source_width"), Rect->SourceWidth * Context.Settings.UnitScale);
            JL->SetNumberField(TEXT("source_height"), Rect->SourceHeight * Context.Settings.UnitScale);
        }
        if (const UDirectionalLightComponent* Dir = Cast<UDirectionalLightComponent>(Light))
        {
            JL->SetNumberField(TEXT("source_angle"), Dir->LightSourceAngle);
            JL->SetNumberField(TEXT("source_soft_angle"), Dir->LightSourceSoftAngle);
        }
        if (const USkyLightComponent* Sky = Cast<USkyLightComponent>(Light))
        {
            JL->SetBoolField(TEXT("real_time_capture"), Sky->bRealTimeCapture);
            if (Sky->Cubemap)
            {
                JL->SetStringField(TEXT("cubemap_path"), Sky->Cubemap->GetPathName());
            }
        }
        Context.LightsJson.Add(MakeShared<FJsonValueObject>(JL));
    }

    static void AddCameraJson(UCameraComponent* Camera, FExportContext& Context)
    {
        if (!Camera || !Context.Settings.bExportCameras)
        {
            return;
        }

        FVector ViewLocation = FVector::ZeroVector;
        FRotator ViewRotation = FRotator::ZeroRotator;
        float ViewFOV = Camera->FieldOfView;
        ExtractCameraView(Camera, ViewLocation, ViewRotation, ViewFOV);

        const FTransform ViewTransform(ViewRotation, ViewLocation, FVector::OneVector);
        const FVector Position = ConvertWorldPosScaled(ViewLocation, Context.Settings);
        const FVector Forward = ConvertLocalDir(ViewRotation.Quaternion().GetForwardVector(), Context.Settings);
        const FVector Up = ConvertLocalDir(ViewRotation.Quaternion().GetUpVector(), Context.Settings);
        const FVector Right = ConvertLocalDir(ViewRotation.Quaternion().GetRightVector(), Context.Settings);

        TSharedPtr<FJsonObject> JC = MakeShared<FJsonObject>();
        JC->SetStringField(TEXT("stable_id"), StableId(TEXT("cam"), Camera->GetPathName()));
        const FString CameraActorName = Camera->GetOwner() ? Camera->GetOwner()->GetName() : Camera->GetName();
        const FString CameraActorLabel = Camera->GetOwner() ? Camera->GetOwner()->GetActorNameOrLabel() : FString();
        JC->SetStringField(TEXT("name"), CameraActorName);
        JC->SetStringField(TEXT("unique_name"), MakeUniqueExportName(Camera->GetOwner(), Camera));
        JC->SetStringField(TEXT("raw_name"), Camera->GetName());
        if (Camera->GetOwner())
        {
            JC->SetStringField(TEXT("actor_name"), Camera->GetOwner()->GetName());
            JC->SetStringField(TEXT("actor_label"), CameraActorLabel);
        }
        JC->SetStringField(TEXT("actor_path"), Camera->GetOwner() ? Camera->GetOwner()->GetPathName() : FString());
        JC->SetStringField(TEXT("component_path"), Camera->GetPathName());
        JC->SetStringField(TEXT("class"), Camera->GetClass()->GetPathName());
        JC->SetArrayField(TEXT("transform_matrix"), MatrixToJson(MakeSceneTransformMatrix(ViewTransform, Context.Settings)));
        JC->SetArrayField(TEXT("position"), Vec3ToJson(Position));
        JC->SetArrayField(TEXT("forward"), Vec3ToJson(Forward));
        JC->SetArrayField(TEXT("up"), Vec3ToJson(Up));
        JC->SetArrayField(TEXT("right"), Vec3ToJson(Right));

        float AspectRatio = Camera->AspectRatio;
        if (!(AspectRatio > KINDA_SMALL_NUMBER))
        {
            if (const UCineCameraComponent* CineForAspect = Cast<UCineCameraComponent>(Camera))
            {
                if (CineForAspect->Filmback.SensorHeight > KINDA_SMALL_NUMBER)
                {
                    AspectRatio = CineForAspect->Filmback.SensorWidth / CineForAspect->Filmback.SensorHeight;
                }
            }
        }
        if (!(AspectRatio > KINDA_SMALL_NUMBER))
        {
            AspectRatio = 16.0f / 9.0f;
        }

        const float FovXDeg = ViewFOV;
        const float FovXRad = FMath::DegreesToRadians(FovXDeg);
        const float FovYRad = 2.0f * FMath::Atan(FMath::Tan(FovXRad * 0.5f) / AspectRatio);
        const float FovYDeg = FMath::RadiansToDegrees(FovYRad);

        JC->SetNumberField(TEXT("fov"), FovXDeg);
        JC->SetNumberField(TEXT("fov_x"), FovXRad);
        JC->SetNumberField(TEXT("fov_x_deg"), FovXDeg);
        JC->SetNumberField(TEXT("fov_y"), FovYRad);
        JC->SetNumberField(TEXT("fov_y_deg"), FovYDeg);
        JC->SetNumberField(TEXT("fov_deg"), FovXDeg);
        JC->SetNumberField(TEXT("aspect_ratio"), AspectRatio);
        JC->SetBoolField(TEXT("constrain_aspect_ratio"), Camera->bConstrainAspectRatio);
        JC->SetNumberField(TEXT("projection_mode"), static_cast<int32>(Camera->ProjectionMode));
        JC->SetNumberField(TEXT("ortho_width"), Camera->OrthoWidth * Context.Settings.UnitScale);
        JC->SetNumberField(TEXT("clip_start"), 10.0 * Context.Settings.UnitScale);
        JC->SetNumberField(TEXT("clip_end"), 100000.0 * Context.Settings.UnitScale);

        float FocusDistance = 1000.0f * Context.Settings.UnitScale;
        if (UCineCameraComponent* Cine = Cast<UCineCameraComponent>(Camera))
        {
            FocusDistance = Cine->FocusSettings.ManualFocusDistance * Context.Settings.UnitScale;
            JC->SetNumberField(TEXT("focal_length"), Cine->CurrentFocalLength);
            JC->SetNumberField(TEXT("aperture"), Cine->CurrentAperture);
            JC->SetNumberField(TEXT("sensor_width"), Cine->Filmback.SensorWidth);
            JC->SetNumberField(TEXT("sensor_height"), Cine->Filmback.SensorHeight);
        }
        JC->SetNumberField(TEXT("focus_distance"), FocusDistance);

        Context.CamerasJson.Add(MakeShared<FJsonValueObject>(JC));
    }

    static void AddDecalJson(UDecalComponent* Decal, FExportContext& Context)
    {
        if (!Decal || !Context.Settings.bExportDecals)
        {
            return;
        }
        TSharedPtr<FJsonObject> JD = MakeShared<FJsonObject>();
        JD->SetStringField(TEXT("stable_id"), StableId(TEXT("decal"), Decal->GetPathName()));
        JD->SetStringField(TEXT("name"), Decal->GetName());
        JD->SetStringField(TEXT("actor_path"), Decal->GetOwner() ? Decal->GetOwner()->GetPathName() : FString());
        JD->SetStringField(TEXT("component_path"), Decal->GetPathName());
        JD->SetStringField(TEXT("class"), Decal->GetClass()->GetPathName());

        const TArray<double> SceneM = MakeSceneTransformMatrix(Decal->GetComponentTransform(), Context.Settings);
        JD->SetArrayField(TEXT("transform_matrix"), MatrixToJson(SceneM));

        const FVector SceneAxisX(SceneM[0], SceneM[4], SceneM[8]);
        const FVector SceneAxisY(SceneM[1], SceneM[5], SceneM[9]);
        const FVector SceneAxisZ(SceneM[2], SceneM[6], SceneM[10]);

        const FVector Position(SceneM[3], SceneM[7], SceneM[11]);
        const FVector AxisX = SceneAxisX.GetSafeNormal();
        const FVector AxisZ = SceneAxisZ.GetSafeNormal();
        FVector AxisY = FVector::CrossProduct(AxisX, AxisZ).GetSafeNormal();
        if (AxisY.IsNearlyZero())
        {
            AxisY = SceneAxisY.GetSafeNormal();
        }

        const FVector RawDecalSize = Decal->DecalSize * Context.Settings.UnitScale;
        const FVector ComponentScale3D(FMath::Abs(SceneAxisX.Size()), FMath::Abs(SceneAxisY.Size()), FMath::Abs(SceneAxisZ.Size()));
        const FVector ScaledDecalSize(RawDecalSize.X * ComponentScale3D.X,
                                      RawDecalSize.Y * ComponentScale3D.Y,
                                      RawDecalSize.Z * ComponentScale3D.Z);

        JD->SetArrayField(TEXT("position"), Vec3ToJson(Position));
        JD->SetArrayField(TEXT("axis_x"), Vec3ToJson(AxisX));
        JD->SetArrayField(TEXT("axis_y"), Vec3ToJson(AxisY));
        JD->SetArrayField(TEXT("axis_z"), Vec3ToJson(AxisZ));
        JD->SetArrayField(TEXT("decal_size"), Vec3ToJson(RawDecalSize));
        JD->SetArrayField(TEXT("size"), Vec3ToJson(ScaledDecalSize));
        JD->SetArrayField(TEXT("component_scale"), Vec3ToJson(ComponentScale3D));
        JD->SetBoolField(TEXT("size_includes_component_scale"), true);
        JD->SetNumberField(TEXT("fade_screen_size"), Decal->FadeScreenSize);
        JD->SetNumberField(TEXT("sort_order"), Decal->SortOrder);
        if (UMaterialInterface* Material = Decal->GetDecalMaterial())
        {
            JD->SetStringField(TEXT("material_id"), GetOrCreateMaterialId(Material, Context));
            JD->SetStringField(TEXT("material"), Material->GetName());
            JD->SetStringField(TEXT("material_name"), Material->GetName());
            JD->SetStringField(TEXT("material_asset_path"), Material->GetPathName());
        }
        Context.DecalsJson.Add(MakeShared<FJsonValueObject>(JD));
    }

    static void AddFogJson(UExponentialHeightFogComponent* Fog, FExportContext& Context)
    {
        if (!Fog || !Context.Settings.bExportFog)
        {
            return;
        }
        TSharedPtr<FJsonObject> JF = MakeShared<FJsonObject>();
        JF->SetStringField(TEXT("stable_id"), StableId(TEXT("fog"), Fog->GetPathName()));
        JF->SetStringField(TEXT("name"), Fog->GetName());
        JF->SetStringField(TEXT("actor_path"), Fog->GetOwner() ? Fog->GetOwner()->GetPathName() : FString());
        JF->SetStringField(TEXT("component_path"), Fog->GetPathName());
        JF->SetNumberField(TEXT("fog_density"), Fog->FogDensity);
        JF->SetNumberField(TEXT("fog_height_falloff"), Fog->FogHeightFalloff);
        JF->SetArrayField(TEXT("fog_inscattering_color"), ColorToJson(GetFogInscatteringColorCompat(Fog)));
        JF->SetNumberField(TEXT("fog_max_opacity"), Fog->FogMaxOpacity);
        JF->SetNumberField(TEXT("start_distance"), Fog->StartDistance * Context.Settings.UnitScale);
        JF->SetBoolField(TEXT("volumetric_fog"), Fog->bEnableVolumetricFog);
        JF->SetNumberField(TEXT("volumetric_scattering_distribution"), Fog->VolumetricFogScatteringDistribution);
        Context.FogJson.Add(MakeShared<FJsonValueObject>(JF));
    }

    static void AddPostProcessJson(UPostProcessComponent* PP, FExportContext& Context)
    {
        if (!PP || !Context.Settings.bExportPostProcess)
        {
            return;
        }
        TSharedPtr<FJsonObject> JP = MakeShared<FJsonObject>();
        JP->SetStringField(TEXT("stable_id"), StableId(TEXT("pp"), PP->GetPathName()));
        JP->SetStringField(TEXT("name"), PP->GetName());
        JP->SetStringField(TEXT("actor_path"), PP->GetOwner() ? PP->GetOwner()->GetPathName() : FString());
        JP->SetStringField(TEXT("component_path"), PP->GetPathName());
        JP->SetBoolField(TEXT("unbound"), PP->bUnbound);
        JP->SetNumberField(TEXT("priority"), PP->Priority);
        JP->SetNumberField(TEXT("blend_radius"), PP->BlendRadius * Context.Settings.UnitScale);
        JP->SetNumberField(TEXT("blend_weight"), PP->BlendWeight);
        const FPostProcessSettings& S = PP->Settings;
        JP->SetNumberField(TEXT("bloom_intensity"), S.BloomIntensity);
        JP->SetNumberField(TEXT("vignette_intensity"), S.VignetteIntensity);
        JP->SetNumberField(TEXT("grain_intensity"), S.FilmGrainIntensity);
        JP->SetNumberField(TEXT("scene_fringe_intensity"), S.SceneFringeIntensity);
        JP->SetNumberField(TEXT("auto_exposure_bias"), S.AutoExposureBias);
        JP->SetNumberField(TEXT("auto_exposure_min_brightness"), S.AutoExposureMinBrightness);
        JP->SetNumberField(TEXT("auto_exposure_max_brightness"), S.AutoExposureMaxBrightness);
        JP->SetNumberField(TEXT("ambient_occlusion_intensity"), S.AmbientOcclusionIntensity);
        JP->SetNumberField(TEXT("motion_blur_amount"), S.MotionBlurAmount);
        Context.PostProcessJson.Add(MakeShared<FJsonValueObject>(JP));
    }

    static void AddAtmosphereJson(USceneComponent* Component, const FString& Type, FExportContext& Context)
    {
        if (!Component || !Context.Settings.bExportAtmosphere)
        {
            return;
        }
        TSharedPtr<FJsonObject> JA = MakeShared<FJsonObject>();
        JA->SetStringField(TEXT("stable_id"), StableId(TEXT("atm"), Component->GetPathName()));
        JA->SetStringField(TEXT("name"), Component->GetName());
        JA->SetStringField(TEXT("actor_path"), Component->GetOwner() ? Component->GetOwner()->GetPathName() : FString());
        JA->SetStringField(TEXT("component_path"), Component->GetPathName());
        JA->SetStringField(TEXT("class"), Component->GetClass()->GetPathName());
        JA->SetStringField(TEXT("type"), Type);
        JA->SetArrayField(TEXT("transform_matrix"), MatrixToJson(MakeSceneTransformMatrix(Component->GetComponentTransform(), Context.Settings)));
        Context.AtmosphereJson.Add(MakeShared<FJsonValueObject>(JA));
    }

    static void AddUnsupportedPrimitiveMetadata(UPrimitiveComponent* Primitive, FExportContext& Context)
    {
        AddPrimitiveJsonCommon(Primitive, Primitive->GetClass()->GetName(), FString(), FString(),
            { MakeSceneTransformMatrix(Primitive->GetComponentTransform(), Context.Settings) }, {}, false, Context,
            FString(), false, TEXT("none"));
    }

    template <typename T>
    static int32 AppendPlaceholder(TArray<uint8>& Buffer)
    {
        const int32 Offset = Buffer.AddUninitialized(sizeof(T));
        FMemory::Memzero(Buffer.GetData() + Offset, sizeof(T));
        return Offset;
    }

    template <typename T>
    static void PatchPod(TArray<uint8>& Buffer, int32 Offset, const T& Value)
    {
        FMemory::Memcpy(Buffer.GetData() + Offset, &Value, sizeof(T));
    }

    template <typename T>
    static void AppendArray(TArray<uint8>& Buffer, const TArray<T>& Array)
    {
        if (Array.Num() == 0)
        {
            return;
        }
        const int32 Bytes = Array.Num() * sizeof(T);
        const int32 Offset = Buffer.AddUninitialized(Bytes);
        FMemory::Memcpy(Buffer.GetData() + Offset, Array.GetData(), Bytes);
    }

    static void WriteMeshesBinary(FExportContext& Context)
    {
        TArray<uint8> Buffer;
        Buffer.Reserve(1024 * 1024);
        FMeshFileHeader FileHeader;
        FMemory::Memcpy(FileHeader.Magic, MeshMagic, sizeof(FileHeader.Magic));
        const int32 FileHeaderOffset = AppendPlaceholder<FMeshFileHeader>(Buffer);

        for (FMeshAsset& Mesh : Context.MeshAssets)
        {
            Mesh.BinaryOffset = Buffer.Num();
            FMeshBlockHeader BlockHeader;
            BlockHeader.LodCount = Mesh.LODs.Num();
            BlockHeader.StableIdHash = StableHash64(Mesh.StableId);
            const int32 BlockHeaderOffset = AppendPlaceholder<FMeshBlockHeader>(Buffer);

            TArray<int32> LODHeaderOffsets;
            for (int32 LODIndex = 0; LODIndex < Mesh.LODs.Num(); ++LODIndex)
            {
                LODHeaderOffsets.Add(AppendPlaceholder<FLODHeader>(Buffer));
            }

            for (int32 LODArrayIndex = 0; LODArrayIndex < Mesh.LODs.Num(); ++LODArrayIndex)
            {
                FMeshLODData& LOD = Mesh.LODs[LODArrayIndex];
                FLODHeader Header;
                Header.LodIndex = LOD.LODIndex;
                Header.VertexCount = LOD.Positions.Num();
                Header.IndexCount = LOD.Indices.Num();
                Header.UVSetCount = LOD.UVSets.Num();
                Header.SectionCount = LOD.Sections.Num();

                Header.PositionsOffset = Buffer.Num(); AppendArray(Buffer, LOD.Positions);
                Header.NormalsOffset = Buffer.Num(); AppendArray(Buffer, LOD.Normals);
                Header.TangentsOffset = Buffer.Num(); AppendArray(Buffer, LOD.Tangents);
                Header.ColorsOffset = Buffer.Num(); AppendArray(Buffer, LOD.Colors);
                Header.UVsOffset = Buffer.Num();
                for (const TArray<FVector2f>& UVSet : LOD.UVSets)
                {
                    AppendArray(Buffer, UVSet);
                }
                Header.IndicesOffset = Buffer.Num(); AppendArray(Buffer, LOD.Indices);

                TArray<FSectionRange> Ranges;
                for (const FMeshSection& Section : LOD.Sections)
                {
                    FSectionRange Range;
                    Range.SectionIndex = Section.SectionIndex;
                    Range.MaterialSlotIndex = Section.MaterialSlotIndex;
                    Range.FirstIndex = Section.FirstIndex;
                    Range.IndexCount = Section.IndexCount;
                    Ranges.Add(Range);
                }
                Header.SectionsOffset = Buffer.Num(); AppendArray(Buffer, Ranges);

                LOD.PositionsOffset = Header.PositionsOffset;
                LOD.NormalsOffset = Header.NormalsOffset;
                LOD.TangentsOffset = Header.TangentsOffset;
                LOD.ColorsOffset = Header.ColorsOffset;
                LOD.UVsOffset = Header.UVsOffset;
                LOD.IndicesOffset = Header.IndicesOffset;
                LOD.SectionsOffset = Header.SectionsOffset;

                PatchPod(Buffer, LODHeaderOffsets[LODArrayIndex], Header);
            }

            BlockHeader.PayloadSize = Buffer.Num() - Mesh.BinaryOffset;
            Mesh.BinarySize = BlockHeader.PayloadSize;
            PatchPod(Buffer, BlockHeaderOffset, BlockHeader);
        }

        FileHeader.MeshCount = Context.MeshAssets.Num();
        PatchPod(Buffer, FileHeaderOffset, FileHeader);
        FFileHelper::SaveArrayToFile(Buffer, *Context.Settings.MeshesBinaryPath);
    }

    static TSharedPtr<FJsonObject> BuildMeshAssetJson(const FMeshAsset& Mesh)
    {
        TSharedPtr<FJsonObject> JM = MakeShared<FJsonObject>();
        JM->SetStringField(TEXT("stable_id"), Mesh.StableId);
        JM->SetStringField(TEXT("name"), Mesh.Name);
        JM->SetStringField(TEXT("primitive_type"), Mesh.PrimitiveType);
        JM->SetStringField(TEXT("source_asset_path"), Mesh.SourceAssetPath);
        JM->SetStringField(TEXT("mesh_space"), Mesh.MeshSpace);
        JM->SetBoolField(TEXT("has_vertex_colors"), MeshAssetHasVertexColors(Mesh));
        JM->SetStringField(TEXT("vertex_color_source"), GetMeshAssetVertexColorSource(Mesh));
        JM->SetNumberField(TEXT("binary_offset"), static_cast<double>(Mesh.BinaryOffset));
        JM->SetNumberField(TEXT("binary_size"), static_cast<double>(Mesh.BinarySize));

        TArray<TSharedPtr<FJsonValue>> LODsJson;
        for (const FMeshLODData& LOD : Mesh.LODs)
        {
            TSharedPtr<FJsonObject> JL = MakeShared<FJsonObject>();
            JL->SetNumberField(TEXT("lod_index"), LOD.LODIndex);
            JL->SetNumberField(TEXT("vertex_count"), LOD.Positions.Num());
            JL->SetNumberField(TEXT("index_count"), LOD.Indices.Num());
            JL->SetNumberField(TEXT("uv_set_count"), LOD.UVSets.Num());
            JL->SetNumberField(TEXT("section_count"), LOD.Sections.Num());
            JL->SetBoolField(TEXT("has_vertex_colors"), LOD.bHasVertexColors);
            JL->SetStringField(TEXT("vertex_color_source"), GetVertexColorSourceString(LOD.bHasVertexColors, LOD.bUsesComponentVertexColors));
            JL->SetNumberField(TEXT("vertex_color_count"), LOD.bHasVertexColors ? LOD.Colors.Num() : 0);
            JL->SetNumberField(TEXT("positions_offset"), static_cast<double>(LOD.PositionsOffset));
            JL->SetNumberField(TEXT("normals_offset"), static_cast<double>(LOD.NormalsOffset));
            JL->SetNumberField(TEXT("tangents_offset"), static_cast<double>(LOD.TangentsOffset));
            JL->SetNumberField(TEXT("colors_offset"), static_cast<double>(LOD.ColorsOffset));
            JL->SetNumberField(TEXT("uvs_offset"), static_cast<double>(LOD.UVsOffset));
            JL->SetNumberField(TEXT("indices_offset"), static_cast<double>(LOD.IndicesOffset));
            JL->SetNumberField(TEXT("sections_offset"), static_cast<double>(LOD.SectionsOffset));

            TArray<TSharedPtr<FJsonValue>> SectionsJson;
            for (const FMeshSection& Section : LOD.Sections)
            {
                TSharedPtr<FJsonObject> JS = MakeShared<FJsonObject>();
                JS->SetNumberField(TEXT("section_index"), Section.SectionIndex);
                JS->SetNumberField(TEXT("material_slot_index"), Section.MaterialSlotIndex);
                JS->SetNumberField(TEXT("first_index"), Section.FirstIndex);
                JS->SetNumberField(TEXT("index_count"), Section.IndexCount);
                if (!Section.MaterialSlotName.IsEmpty())
                {
                    JS->SetStringField(TEXT("material_slot_name"), Section.MaterialSlotName);
                }
                SectionsJson.Add(MakeShared<FJsonValueObject>(JS));
            }
            JL->SetArrayField(TEXT("sections"), SectionsJson);
            LODsJson.Add(MakeShared<FJsonValueObject>(JL));
        }
        JM->SetArrayField(TEXT("lods"), LODsJson);
        return JM;
    }

    static void BuildWorldInfo(UWorld* World, FExportContext& Context, TSharedRef<FJsonObject> Root)
    {
        TSharedPtr<FJsonObject> JW = MakeShared<FJsonObject>();
        JW->SetStringField(TEXT("name"), World->GetName());
        JW->SetStringField(TEXT("path"), World->GetPathName());
        JW->SetStringField(TEXT("map_name"), World->GetMapName());
        JW->SetStringField(TEXT("persistent_level_path"), World->PersistentLevel ? World->PersistentLevel->GetPathName() : FString());
        JW->SetBoolField(TEXT("has_world_partition"), World->GetWorldPartition() != nullptr);
        JW->SetNumberField(TEXT("world_type"), static_cast<int32>(World->WorldType));
        JW->SetNumberField(TEXT("unit_scale"), Context.Settings.UnitScale);
        JW->SetStringField(TEXT("handedness"), Context.Settings.bMirrorYToRightHanded ? TEXT("right-handed (Y mirrored from UE)") : TEXT("left-handed (UE native)"));
        JW->SetStringField(TEXT("meshes_file"), FPaths::GetCleanFilename(Context.Settings.MeshesBinaryPath));
        Root->SetObjectField(TEXT("world"), JW);
    }

    static void CollectScene(UWorld* World, AActor* ExporterActor, FExportContext& Context)
    {
        for (TActorIterator<AActor> It(World); It; ++It)
        {
            AActor* Actor = *It;
            if (!ShouldExportActor(Actor, Context.Settings, ExporterActor))
            {
                continue;
            }

            AddActorJson(Actor, Context);

            TArray<UActorComponent*> Components;
            Actor->GetComponents(Components);
            for (UActorComponent* BaseComponent : Components)
            {
                if (!ShouldExportComponent(BaseComponent, Context.Settings))
                {
                    continue;
                }

                if (UStaticMeshComponent* StaticMesh = Cast<UStaticMeshComponent>(BaseComponent))
                {
                    AddStaticMeshPrimitive(StaticMesh, Context);
                    continue;
                }
                if (USkeletalMeshComponent* Skeletal = Cast<USkeletalMeshComponent>(BaseComponent))
                {
                    AddSkeletalPrimitive(Skeletal, Context);
                    continue;
                }
                if (ULandscapeComponent* LandscapeComponent = Cast<ULandscapeComponent>(BaseComponent))
                {
                    AddLandscapePrimitive(LandscapeComponent, Context);
                    continue;
                }
                if (ULightComponent* Light = Cast<ULightComponent>(BaseComponent))
                {
                    AddLightJson(Light, Context);
                    continue;
                }
                if (UCameraComponent* Camera = Cast<UCameraComponent>(BaseComponent))
                {
                    AddCameraJson(Camera, Context);
                    continue;
                }
                if (UDecalComponent* Decal = Cast<UDecalComponent>(BaseComponent))
                {
                    AddDecalJson(Decal, Context);
                    continue;
                }
                if (UExponentialHeightFogComponent* Fog = Cast<UExponentialHeightFogComponent>(BaseComponent))
                {
                    AddFogJson(Fog, Context);
                    continue;
                }
                if (UPostProcessComponent* PostProcess = Cast<UPostProcessComponent>(BaseComponent))
                {
                    AddPostProcessJson(PostProcess, Context);
                    continue;
                }
                if (USkyAtmosphereComponent* SkyAtmosphere = Cast<USkyAtmosphereComponent>(BaseComponent))
                {
                    AddAtmosphereJson(SkyAtmosphere, TEXT("sky_atmosphere"), Context);
                    continue;
                }
                if (UVolumetricCloudComponent* VolumetricCloud = Cast<UVolumetricCloudComponent>(BaseComponent))
                {
                    AddAtmosphereJson(VolumetricCloud, TEXT("volumetric_cloud"), Context);
                    continue;
                }
                if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(BaseComponent))
                {
                    AddUnsupportedPrimitiveMetadata(Primitive, Context);
                    continue;
                }
            }
        }
    }

    static void WriteSceneJson(UWorld* World, FExportContext& Context)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("format"), TEXT("SceneRTXSceneExport"));
        Root->SetNumberField(TEXT("version"), 1);
        BuildWorldInfo(World, Context, Root);

        TArray<TSharedPtr<FJsonValue>> MeshAssetsJson;
        for (const FMeshAsset& Mesh : Context.MeshAssets)
        {
            MeshAssetsJson.Add(MakeShared<FJsonValueObject>(BuildMeshAssetJson(Mesh)));
        }

        Root->SetArrayField(TEXT("mesh_assets"), MeshAssetsJson);
        Root->SetArrayField(TEXT("actors"), Context.ActorsJson);
        Root->SetArrayField(TEXT("primitives"), Context.PrimitivesJson);
        Root->SetArrayField(TEXT("materials"), Context.MaterialsJson);
        Root->SetArrayField(TEXT("lights"), Context.LightsJson);
        Root->SetArrayField(TEXT("cameras"), Context.CamerasJson);
        Root->SetArrayField(TEXT("atmosphere"), Context.AtmosphereJson);
        Root->SetArrayField(TEXT("fog"), Context.FogJson);
        Root->SetArrayField(TEXT("postprocess"), Context.PostProcessJson);
        Root->SetArrayField(TEXT("decals"), Context.DecalsJson);
        Root->SetArrayField(TEXT("textures"), Context.TexturesJson);
        Root->SetArrayField(TEXT("landscapes"), Context.LandscapesJson);

        FString JsonText;
        auto Writer = TJsonWriterFactory<>::Create(&JsonText);
        FJsonSerializer::Serialize(Root, Writer);
        FFileHelper::SaveStringToFile(JsonText, *Context.Settings.SceneJsonPath);
    }
}



bool FSceneRTSceneExporterCore::ExportEditorWorld(const USceneRTSceneExportSettings* InSettings, FString* OutError)
{
#if !WITH_EDITOR
    if (OutError)
    {
        *OutError = TEXT("Scene export is only available in editor builds.");
    }
    return false;
#else
    if (!GEditor)
    {
        if (OutError)
        {
            *OutError = TEXT("GEditor is null.");
        }
        return false;
    }

    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World)
    {
        if (OutError)
        {
            *OutError = TEXT("Editor world is null.");
        }
        UE_LOG(LogSceneRTSceneExporter, Warning, TEXT("SceneRTSceneExporter: Editor world is null, export skipped"));
        return false;
    }

    const USceneRTSceneExportSettings* SettingsObject = InSettings ? InSettings : GetDefault<USceneRTSceneExportSettings>();
    if (!SettingsObject)
    {
        if (OutError)
        {
            *OutError = TEXT("Settings object is null.");
        }
        return false;
    }

    SceneRTSceneExporter::FSettings Settings;
    Settings.OutputDirectory = SettingsObject->OutputDirectory.IsEmpty() ? FPaths::Combine(FPaths::ProjectDir(), TEXT("Saved/SceneRTXExport")) : SettingsObject->OutputDirectory;
    Settings.SceneJsonPath = FPaths::Combine(Settings.OutputDirectory, SettingsObject->SceneFileName);
    Settings.MeshesBinaryPath = FPaths::Combine(Settings.OutputDirectory, SettingsObject->MeshesFileName);
    Settings.TexturesDir = FPaths::Combine(Settings.OutputDirectory, TEXT("Textures"));

    Settings.UnitScale = SettingsObject->UnitScale;
    Settings.bMirrorYToRightHanded = SettingsObject->bMirrorYToRightHanded;
    Settings.bExportOnlyVisibleComponents = SettingsObject->bExportOnlyVisibleComponents;
    Settings.bExportHiddenActors = SettingsObject->bExportHiddenActors;
    Settings.bExportAllStaticMeshLODs = SettingsObject->bExportAllStaticMeshLODs;
    Settings.bExportSplineMeshGeometry = SettingsObject->bExportSplineMeshGeometry;
    Settings.bExportLandscapeGeometry = SettingsObject->bExportLandscapeGeometry;
    Settings.bExportSkeletalMeshMetadata = SettingsObject->bExportSkeletalMeshMetadata;
    Settings.bExportMaterialTextures = SettingsObject->bExportMaterialTextures;
    Settings.bExportPostProcess = SettingsObject->bExportPostProcess;
    Settings.bExportFog = SettingsObject->bExportFog;
    Settings.bExportAtmosphere = SettingsObject->bExportAtmosphere;
    Settings.bExportDecals = SettingsObject->bExportDecals;
    Settings.bExportLights = SettingsObject->bExportLights;
    Settings.bExportCameras = SettingsObject->bExportCameras;
    Settings.bExportActorTags = SettingsObject->bExportActorTags;
    Settings.bExportDataLayers = SettingsObject->bExportDataLayers;
    Settings.MaxTextureExportSize = SettingsObject->MaxTextureExportSize;

    IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
    PF.CreateDirectoryTree(*Settings.OutputDirectory);
    PF.CreateDirectoryTree(*Settings.TexturesDir);

    SceneRTSceneExporter::FExportContext Context;
    Context.Settings = Settings;

    SceneRTSceneExporter::CollectScene(World, nullptr, Context);
    SceneRTSceneExporter::WriteMeshesBinary(Context);
    SceneRTSceneExporter::WriteSceneJson(World, Context);

    UE_LOG(LogSceneRTSceneExporter, Log,
        TEXT("SceneRTSceneExporter: exported scene to '%s' and meshes to '%s'"),
        *Settings.SceneJsonPath,
        *Settings.MeshesBinaryPath);

    if (OutError)
    {
        OutError->Reset();
    }
    return true;
#endif
}
