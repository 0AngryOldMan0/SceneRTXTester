#include "SceneRTV2Texture.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"

#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"
#include "Engine/VolumeTexture.h"
#include "Engine/TexturePlatformData.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/TextureRenderTargetResource.h"
#include "RenderingThread.h"
#include "VT/RuntimeVirtualTexture.h"

namespace SceneRTV2::Texture
{
    namespace
    {
        SceneRTSceneExporterV2::ETexturePixelFormat MapFormat(EPixelFormat PF)
        {
            using F = SceneRTSceneExporterV2::ETexturePixelFormat;
            switch (PF)
            {
                case PF_G8:        return F::R8;
                case PF_R8G8:      return F::RG8;
                case PF_R8G8B8A8:  return F::RGBA8;
                case PF_B8G8R8A8:  return F::BGRA8;
                case PF_R16F:      return F::R16F;
                case PF_G16R16F:   return F::RG16F;
                case PF_FloatRGBA: return F::RGBA16F;
                case PF_R32_FLOAT: return F::R32F;
                case PF_A32B32G32R32F: return F::RGBA32F;
                case PF_DXT1:      return F::BC1;
                case PF_DXT5:      return F::BC3;
                case PF_BC4:       return F::BC4;
                case PF_BC5:       return F::BC5;
                case PF_BC6H:      return F::BC6H;
                case PF_BC7:       return F::BC7;
                default:           return F::Unknown;
            }
        }

        // Reads all mips from FTexturePlatformData into Rec.Payload (raw BCn / uncompressed bytes).
        // Each mip is appended sequentially; consumer locates them by computing BCn block sizes.
        static void ExtractMipChain(FTexturePlatformData* PlatformData, FTextureRecord& Rec)
        {
            if (!PlatformData || PlatformData->Mips.IsEmpty()) { return; }

            int32 WrittenMips = 0;
            for (FTexture2DMipMap& Mip : PlatformData->Mips)
            {
                Mip.BulkData.ForceBulkDataResident();
                const int64 MipSize = Mip.BulkData.GetBulkDataSize();
                if (MipSize <= 0) { continue; }

                const void* Ptr = Mip.BulkData.LockReadOnly();
                if (Ptr)
                {
                    Rec.Payload.Append(static_cast<const uint8*>(Ptr),
                                       static_cast<int32>(MipSize));
                    ++WrittenMips;
                }
                Mip.BulkData.Unlock();
            }
            if (WrittenMips > 0) { Rec.NumMips = WrittenMips; }
        }
    }

    FStableId Resolve(UTexture* TextureInput, FExportContext& Ctx)
    {
        FStableId Empty;
        if (!TextureInput) { return Empty; }

        const FString Key = TextureInput->GetPathName();
        if (const int32* Existing = Ctx.TextureByPath.Find(Key))
        {
            return Ctx.Textures[*Existing].Id;
        }

        FTextureRecord Rec;
        Rec.Id = Identity::ForTexture(Ctx.SceneGuid, TextureInput);
        Rec.SourcePath = Key;
        Rec.bSrgb = TextureInput->SRGB;
        Rec.bNormalMap = TextureInput->IsNormalMap();

        if (UTexture2D* T2D = Cast<UTexture2D>(TextureInput))
        {
            Rec.Width = T2D->GetSizeX();
            Rec.Height = T2D->GetSizeY();
            Rec.NumMips = T2D->GetNumMips();
            Rec.PixelFormat = MapFormat(T2D->GetPixelFormat());
            ExtractMipChain(T2D->GetPlatformData(), Rec);
        }
        else if (UTextureCube* TCube = Cast<UTextureCube>(TextureInput))
        {
            Rec.Width = TCube->GetSizeX();
            Rec.Height = TCube->GetSizeY();
            Rec.NumSlices = 6;
            Rec.NumMips = TCube->GetNumMips();
            Rec.PixelFormat = MapFormat(TCube->GetPixelFormat());
            ExtractMipChain(TCube->GetPlatformData(), Rec);
        }
        else if (UVolumeTexture* TVol = Cast<UVolumeTexture>(TextureInput))
        {
            Rec.Width  = TVol->GetSizeX();
            Rec.Height = TVol->GetSizeY();
            Rec.Depth  = TVol->GetSizeZ();
            Rec.NumMips = TVol->GetNumMips();
            Rec.PixelFormat = MapFormat(TVol->GetPixelFormat());
            ExtractMipChain(TVol->GetPlatformData(), Rec);
        }
        else if (UTextureRenderTarget2D* TRT = Cast<UTextureRenderTarget2D>(TextureInput))
        {
            Rec.Width      = TRT->SizeX;
            Rec.Height     = TRT->SizeY;
            Rec.NumMips    = 1;
            Rec.PixelFormat = MapFormat(TRT->GetFormat());

            // Pixel readback: flush render commands first, then read via game-thread resource.
            FlushRenderingCommands();
            if (FTextureRenderTarget2DResource* RTRes =
                    static_cast<FTextureRenderTarget2DResource*>(
                        TRT->GameThread_GetRenderTargetResource()))
            {
                TArray<FColor> Pixels;
                if (RTRes->ReadPixels(Pixels) && Pixels.Num() == Rec.Width * Rec.Height)
                {
                    Rec.Payload.Append(reinterpret_cast<const uint8*>(Pixels.GetData()),
                                       Pixels.Num() * sizeof(FColor));
                    Rec.PixelFormat = SceneRTSceneExporterV2::ETexturePixelFormat::BGRA8;
                }
                else
                {
                    Ctx.AddIssue(TEXT("warn"), TEXT("texture"),
                        FString::Printf(TEXT("ReadPixels failed for RenderTarget %s"), *Key),
                        Rec.Id.Value);
                }
            }
        }
        else
        {
            Ctx.AddIssue(TEXT("warn"), TEXT("texture"),
                FString::Printf(TEXT("Unsupported texture class %s for %s"),
                    *TextureInput->GetClass()->GetName(), *Key),
                Rec.Id.Value);
        }

        const int32 Idx = Ctx.Textures.Add(MoveTemp(Rec));
        Ctx.TextureByPath.Add(Key, Idx);
        return Ctx.Textures[Idx].Id;
    }

    FStableId BakeRuntimeVirtualTexture(URuntimeVirtualTexture* Rvt,
                                        const FBox& /*WorldBounds*/,
                                        FExportContext& Ctx)
    {
        if (!Rvt) { return {}; }

        // If a pre-baked streaming texture exists, reuse it directly.
        if (UTexture2D* Streaming = Rvt->GetStreamingTexture())
        {
            return Resolve(Streaming, Ctx);
        }

        // Full FRVTBaker bake requires renderer access — not viable in export context.
        // Record the RVT with an empty payload so the material can still
        // reference a stable ID.
        const FString Key = Rvt->GetPathName();
        if (const int32* Existing = Ctx.TextureByPath.Find(Key))
        {
            return Ctx.Textures[*Existing].Id;
        }
        FTextureRecord Rec;
        Rec.Id         = Identity::ForTexture(Ctx.SceneGuid, Rvt);
        Rec.SourcePath = Key;
        Rec.bRvtBaked  = false;
        Ctx.AddIssue(TEXT("warn"), TEXT("texture"),
            FString::Printf(TEXT("RVT %s has no streaming texture; pixel data unavailable"), *Key),
            Rec.Id.Value);
        const int32 Idx = Ctx.Textures.Add(MoveTemp(Rec));
        Ctx.TextureByPath.Add(Key, Idx);
        return Ctx.Textures[Idx].Id;
    }

    void DetectOrmChannelLayout(const UTexture* Texture, int32& OutAo, int32& OutRough, int32& OutMetal)
    {
        // Conservative defaults — caller can override if material parameters
        // specify a different convention.
        OutAo = 0; OutRough = 1; OutMetal = 2;
        if (!Texture) { return; }
        const FString Name = Texture->GetName().ToLower();
        if (Name.Contains(TEXT("_orm"))) { OutAo = 0; OutRough = 1; OutMetal = 2; return; }
        if (Name.Contains(TEXT("_rma"))) { OutRough = 0; OutMetal = 1; OutAo = 2; return; }
        if (Name.Contains(TEXT("_mra"))) { OutMetal = 0; OutRough = 1; OutAo = 2; return; }
    }
}
