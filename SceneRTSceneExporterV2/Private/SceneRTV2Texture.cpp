#include "SceneRTV2Texture.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Identity.h"

#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"
#include "Engine/VolumeTexture.h"
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
        }
        else if (UTextureCube* TCube = Cast<UTextureCube>(TextureInput))
        {
            Rec.Width = TCube->GetSizeX();
            Rec.Height = TCube->GetSizeY();
            Rec.NumSlices = 6;
            Rec.NumMips = TCube->GetNumMips();
            Rec.PixelFormat = MapFormat(TCube->GetPixelFormat());
        }
        else if (UVolumeTexture* TVol = Cast<UVolumeTexture>(TextureInput))
        {
            Rec.Width  = TVol->GetSizeX();
            Rec.Height = TVol->GetSizeY();
            Rec.Depth  = TVol->GetSizeZ();
            Rec.NumMips = TVol->GetNumMips();
            Rec.PixelFormat = MapFormat(TVol->GetPixelFormat());
        }
        else
        {
            // VolumeTexture / RenderTarget / future types — TODO(iterative).
            Ctx.AddIssue(TEXT("warn"), TEXT("texture"),
                FString::Printf(TEXT("Unsupported texture class %s for %s"),
                    *TextureInput->GetClass()->GetName(), *Key),
                Rec.Id.Value);
        }

        // TODO(iterative): pull the source-art mip chain from Texture->Source
        // and pack into Rec.Payload. For now we register the descriptor only;
        // the consumer can locate the raw payload via SourcePath.

        const int32 Idx = Ctx.Textures.Add(MoveTemp(Rec));
        Ctx.TextureByPath.Add(Key, Idx);
        return Ctx.Textures[Idx].Id;
    }

    FStableId BakeRuntimeVirtualTexture(URuntimeVirtualTexture* /*Rvt*/,
                                        const FBox& /*WorldBounds*/,
                                        FExportContext& /*Ctx*/)
    {
        // TODO(iterative): allocate a UTextureRenderTarget2D matching the RVT
        // material set, render the world bounds into it via FRVTBaker, read
        // back the RT pixels into Rec.Payload, mark bRvtBaked.
        return {};
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
