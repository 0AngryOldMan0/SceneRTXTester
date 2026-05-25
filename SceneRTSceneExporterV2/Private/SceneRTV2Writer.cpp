#include "SceneRTV2Writer.h"
#include "SceneRTV2Context.h"
#include "SceneRTV2Format.h"

#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/SecureHash.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/BufferArchive.h"
#include "HAL/FileManager.h"

// Binary mesh + texture containers follow the layout in SceneRTV2Format.h:
//   [ContainerHeader] [TOC:uint64×N] [AssetHeader 0] [LodHeaders] [LOD payloads] ...
// All offsets stored in headers are absolute file offsets (zero-based start of
// container). Index buffers default to uint32; sections are stored as
// FMeshSectionRange arrays.
//
// JSON sidecars (meshes.json / textures.json) are kept as an inspectable index
// so downstream tooling can read the bundle without parsing the binary first.

namespace SceneRTV2::Writer
{
    namespace
    {
        static void HashStableIdToBytes(const FStableId& Id, uint8 (&OutBytes)[20])
        {
            FSHA1 Sha;
            const FTCHARToUTF8 Utf8(*Id.Value);
            Sha.Update(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
            Sha.Final();
            Sha.GetHash(OutBytes);
        }

        static uint64 DoubleToU64(double D)
        {
            uint64 Out = 0;
            FMemory::Memcpy(&Out, &D, sizeof(double));
            return Out;
        }

        /** Append helper — returns the offset at which the data was written. */
        template <typename T>
        static uint64 AppendArray(TArray<uint8>& Buf, const TArray<T>& Src)
        {
            const uint64 Offset = static_cast<uint64>(Buf.Num());
            if (Src.Num() > 0)
            {
                const int32 ByteCount = Src.Num() * sizeof(T);
                Buf.AddUninitialized(ByteCount);
                FMemory::Memcpy(Buf.GetData() + Offset, Src.GetData(), ByteCount);
            }
            return Offset;
        }

        template <typename T>
        static uint64 AppendStruct(TArray<uint8>& Buf, const T& S)
        {
            const uint64 Offset = static_cast<uint64>(Buf.Num());
            Buf.AddUninitialized(sizeof(T));
            FMemory::Memcpy(Buf.GetData() + Offset, &S, sizeof(T));
            return Offset;
        }

        static uint64 AppendBytes(TArray<uint8>& Buf, const uint8* Data, int32 Size)
        {
            const uint64 Offset = static_cast<uint64>(Buf.Num());
            if (Data && Size > 0)
            {
                Buf.Append(Data, Size);
            }
            return Offset;
        }

        static void PatchAt(TArray<uint8>& Buf, uint64 Offset, const void* Src, SIZE_T Size)
        {
            check(Offset + Size <= static_cast<uint64>(Buf.Num()));
            FMemory::Memcpy(Buf.GetData() + Offset, Src, Size);
        }

        static uint32 ComputeAttributeMask(const FMeshLod& Lod)
        {
            using A = SceneRTSceneExporterV2::EVertexAttributeMask;
            uint32 Mask = 0;
            if (Lod.Positions.Num() > 0)        Mask |= static_cast<uint32>(A::Position);
            if (Lod.Normals.Num() > 0)          Mask |= static_cast<uint32>(A::Normal);
            if (Lod.Tangents.Num() > 0)         Mask |= static_cast<uint32>(A::Tangent);
            if (Lod.Colors.Num() > 0)           Mask |= static_cast<uint32>(A::Color);
            if (Lod.UVs.Num() > 0)              Mask |= static_cast<uint32>(A::UV0);
            if (Lod.UVs.Num() > 1)              Mask |= static_cast<uint32>(A::UV1);
            if (Lod.UVs.Num() > 2)              Mask |= static_cast<uint32>(A::UV2);
            if (Lod.UVs.Num() > 3)              Mask |= static_cast<uint32>(A::UV3);
            if (Lod.SkinIndices.Num() > 0)      Mask |= static_cast<uint32>(A::SkinIndices);
            if (Lod.SkinWeights.Num() > 0)      Mask |= static_cast<uint32>(A::SkinWeights);
            if (Lod.ArclenAlongSpline.Num() > 0)Mask |= static_cast<uint32>(A::ArclenAlongSpline);
            return Mask;
        }
    }

    namespace
    {
        bool WriteJsonObject(const FString& Path, const TSharedRef<FJsonObject>& Object, FString* OutError)
        {
            FString Out;
            TSharedRef<TJsonWriter<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>> W =
                TJsonWriterFactory<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>::Create(&Out);
            if (!FJsonSerializer::Serialize(Object, W))
            {
                if (OutError) { *OutError = FString::Printf(TEXT("Serialize failed: %s"), *Path); }
                return false;
            }
            if (!FFileHelper::SaveStringToFile(Out, *Path))
            {
                if (OutError) { *OutError = FString::Printf(TEXT("SaveStringToFile failed: %s"), *Path); }
                return false;
            }
            return true;
        }

        TSharedPtr<FJsonObject> SerializeTransform(const FTransform& T)
        {
            TSharedRef<FJsonObject> O = MakeShared<FJsonObject>();
            const FVector L = T.GetLocation();
            const FQuat  R = T.GetRotation();
            const FVector S = T.GetScale3D();
            TArray<TSharedPtr<FJsonValue>> Loc { MakeShared<FJsonValueNumber>(L.X), MakeShared<FJsonValueNumber>(L.Y), MakeShared<FJsonValueNumber>(L.Z) };
            TArray<TSharedPtr<FJsonValue>> Rot { MakeShared<FJsonValueNumber>(R.X), MakeShared<FJsonValueNumber>(R.Y), MakeShared<FJsonValueNumber>(R.Z), MakeShared<FJsonValueNumber>(R.W) };
            TArray<TSharedPtr<FJsonValue>> Scl { MakeShared<FJsonValueNumber>(S.X), MakeShared<FJsonValueNumber>(S.Y), MakeShared<FJsonValueNumber>(S.Z) };
            O->SetArrayField(TEXT("location"), Loc);
            O->SetArrayField(TEXT("rotation_quat"), Rot);
            O->SetArrayField(TEXT("scale"), Scl);
            return O;
        }
    }

    bool WriteMeshContainer(const FExportContext& Ctx, FString* OutError)
    {
        using namespace SceneRTSceneExporterV2;

        const FString BinPath  = Ctx.OutputDir / TEXT("meshes_v2.bin");
        const FString JsonPath = Ctx.OutputDir / TEXT("meshes.json");

        // ---- Pass 1: build payload for every mesh asset and remember its offset
        //      relative to the start of the file. We don't know the final file
        //      layout yet (header + TOC + asset headers come first), so we build
        //      payloads in a single buffer with placeholder absolute offsets,
        //      then prepend the fixed-size prefix and patch absolute offsets.

        const int32 NumMeshes = Ctx.Meshes.Num();
        TArray<FMeshAssetHeader> AssetHeaders;
        AssetHeaders.AddDefaulted(NumMeshes);
        TArray<TArray<uint8>> AssetPayloads;
        AssetPayloads.SetNum(NumMeshes);

        // Per-mesh payload layout:
        //   FMeshLodHeader[NumLods]
        //   then for each LOD in order:
        //     Positions, Normals, Tangents, Colors, UV0..UVk, Indices,
        //     Sections (FMeshSectionRange[]), Arclen, SkinIndices, SkinWeights
        // Offsets inside FMeshLodHeader are absolute (file-relative); patched in
        // pass 2 once we know the asset's PayloadOffset.

        for (int32 mi = 0; mi < NumMeshes; ++mi)
        {
            const FMeshAsset& Asset = Ctx.Meshes[mi];
            FMeshAssetHeader& H = AssetHeaders[mi];
            H.Kind = static_cast<uint8>(Asset.Kind);
            H.NumLods = static_cast<uint8>(Asset.Lods.Num());
            uint32 SectionTotal = 0;
            uint32 AttrMaskUnion = 0;
            for (const FMeshLod& Lod : Asset.Lods)
            {
                SectionTotal += static_cast<uint32>(Lod.Sections.Num());
                AttrMaskUnion |= ComputeAttributeMask(Lod);
            }
            H.NumSections = static_cast<uint16>(FMath::Min<uint32>(SectionTotal, 0xFFFFu));
            H.AttributeMask = AttrMaskUnion;
            const FVector C = Asset.LocalBounds.IsValid ? Asset.LocalBounds.GetCenter() : FVector::ZeroVector;
            const FVector E = Asset.LocalBounds.IsValid ? Asset.LocalBounds.GetExtent() : FVector::ZeroVector;
            H.BoundsCenterX = DoubleToU64(C.X); H.BoundsCenterY = DoubleToU64(C.Y); H.BoundsCenterZ = DoubleToU64(C.Z);
            H.BoundsExtentX = DoubleToU64(E.X); H.BoundsExtentY = DoubleToU64(E.Y); H.BoundsExtentZ = DoubleToU64(E.Z);
            HashStableIdToBytes(Asset.Id, H.StableIdHash);

            TArray<uint8>& Payload = AssetPayloads[mi];

            // Reserve LOD headers up front so payload offsets that follow are stable.
            TArray<FMeshLodHeader> LodHdrs;
            LodHdrs.AddDefaulted(Asset.Lods.Num());
            const uint64 LodHdrBlockOffset = AppendBytes(
                Payload, reinterpret_cast<const uint8*>(LodHdrs.GetData()),
                LodHdrs.Num() * sizeof(FMeshLodHeader));

            for (int32 li = 0; li < Asset.Lods.Num(); ++li)
            {
                const FMeshLod& Lod = Asset.Lods[li];
                FMeshLodHeader Lh;
                Lh.LodIndex      = static_cast<uint32>(Lod.LodIndex);
                Lh.NumVertices   = static_cast<uint32>(Lod.Positions.Num());
                Lh.NumIndices    = static_cast<uint32>(Lod.Indices.Num());
                Lh.NumSections   = static_cast<uint32>(Lod.Sections.Num());
                Lh.IndexStride   = 4; // uint32 indices
                Lh.SkinInfluences = (Lod.SkinIndices.Num() > 0) ? 4u : 0u;

                Lh.PositionsOffset = AppendArray(Payload, Lod.Positions);
                Lh.NormalsOffset   = AppendArray(Payload, Lod.Normals);
                Lh.TangentsOffset  = AppendArray(Payload, Lod.Tangents);
                Lh.ColorsOffset    = AppendArray(Payload, Lod.Colors);

                // UVs: concatenate channels back-to-back; UVsOffset points at
                // channel 0; subsequent channels follow at +(NumVerts*8) each.
                if (Lod.UVs.Num() > 0)
                {
                    Lh.UVsOffset = static_cast<uint64>(Payload.Num());
                    for (const TArray<FVector2f>& Channel : Lod.UVs)
                    {
                        AppendArray(Payload, Channel);
                    }
                }
                Lh.IndicesOffset = AppendArray(Payload, Lod.Indices);

                // Sections — convert FMeshSection → FMeshSectionRange.
                TArray<FMeshSectionRange> SectionRanges;
                SectionRanges.Reserve(Lod.Sections.Num());
                for (const FMeshSection& S : Lod.Sections)
                {
                    FMeshSectionRange R;
                    R.SectionIndex      = S.SectionIndex;
                    R.MaterialSlotIndex = S.MaterialSlotIndex;
                    R.FirstIndex        = S.FirstIndex;
                    R.IndexCount        = S.IndexCount;
                    R.MinVertexIndex    = S.MinVertex;
                    R.MaxVertexIndex    = S.MaxVertex;
                    R.NumUvChannels     = static_cast<uint32>(Lod.UVs.Num());
                    R.CastShadowFlags   = S.bCastShadow ? 1u : 0u;
                    SectionRanges.Add(R);
                }
                Lh.SectionsOffset = AppendArray(Payload, SectionRanges);

                Lh.SkinIndicesOffset = AppendArray(Payload, Lod.SkinIndices);
                Lh.SkinWeightsOffset = AppendArray(Payload, Lod.SkinWeights);
                Lh.ArclenOffset      = AppendArray(Payload, Lod.ArclenAlongSpline);

                LodHdrs[li] = Lh;
            }

            // Write the now-filled LOD headers back to the reserved block.
            PatchAt(Payload, LodHdrBlockOffset, LodHdrs.GetData(), LodHdrs.Num() * sizeof(FMeshLodHeader));
        }

        // ---- Pass 2: build the final file with absolute offsets.

        // Layout: [Container header][TOC: uint64 × NumMeshes][Asset headers][Payloads...]
        const uint64 ContainerHeaderSize = sizeof(FMeshContainerHeader);
        const uint64 TocBytes = sizeof(uint64) * NumMeshes;
        const uint64 AssetHdrBytes = sizeof(FMeshAssetHeader) * NumMeshes;
        const uint64 PrefixSize = ContainerHeaderSize + TocBytes + AssetHdrBytes;

        // Compute absolute payload offsets and patch each LOD's offsets within
        // its payload to file-relative addresses.
        TArray<uint64> Toc;
        Toc.Reserve(NumMeshes);
        uint64 RunningOffset = PrefixSize;
        for (int32 mi = 0; mi < NumMeshes; ++mi)
        {
            AssetHeaders[mi].PayloadOffset = RunningOffset;
            AssetHeaders[mi].PayloadSize   = static_cast<uint64>(AssetPayloads[mi].Num());
            Toc.Add(RunningOffset);

            // Patch LOD-internal offsets: payload-local → absolute file offsets.
            const int32 NumLods = AssetHeaders[mi].NumLods;
            TArray<uint8>& P = AssetPayloads[mi];
            for (int32 li = 0; li < NumLods; ++li)
            {
                const SIZE_T HdrOffsetInPayload = li * sizeof(FMeshLodHeader);
                FMeshLodHeader Lh;
                FMemory::Memcpy(&Lh, P.GetData() + HdrOffsetInPayload, sizeof(FMeshLodHeader));
                auto Bump = [&](uint64& O){ if (O != 0 || true) { O += RunningOffset; } };
                // Each "offset" in the LOD header is a payload-local offset; convert
                // unconditionally so that even a zero-length array (offset == payload
                // length at time of append) becomes a well-defined absolute address.
                Bump(Lh.PositionsOffset);
                Bump(Lh.NormalsOffset);
                Bump(Lh.TangentsOffset);
                Bump(Lh.ColorsOffset);
                Bump(Lh.UVsOffset);
                Bump(Lh.IndicesOffset);
                Bump(Lh.SectionsOffset);
                Bump(Lh.SkinIndicesOffset);
                Bump(Lh.SkinWeightsOffset);
                Bump(Lh.ArclenOffset);
                FMemory::Memcpy(P.GetData() + HdrOffsetInPayload, &Lh, sizeof(FMeshLodHeader));
            }

            RunningOffset += AssetPayloads[mi].Num();
        }

        // ---- Build container header
        FMeshContainerHeader CH;
        CH.NumMeshes    = static_cast<uint32>(NumMeshes);
        CH.TocOffset    = static_cast<uint32>(ContainerHeaderSize);
        CH.PayloadStart = PrefixSize;
        CH.TotalSize    = RunningOffset;
        const uint64 GuidHi = (uint64(Ctx.SceneGuid.A) << 32) | uint64(Ctx.SceneGuid.B);
        const uint64 GuidLo = (uint64(Ctx.SceneGuid.C) << 32) | uint64(Ctx.SceneGuid.D);
        CH.SceneGuidHi = GuidHi;
        CH.SceneGuidLo = GuidLo;

        // ---- Assemble file
        TArray<uint8> File;
        File.Reserve(RunningOffset);
        AppendStruct(File, CH);
        AppendBytes(File, reinterpret_cast<const uint8*>(Toc.GetData()), Toc.Num() * sizeof(uint64));
        AppendBytes(File, reinterpret_cast<const uint8*>(AssetHeaders.GetData()),
                    AssetHeaders.Num() * sizeof(FMeshAssetHeader));
        for (const TArray<uint8>& P : AssetPayloads)
        {
            File.Append(P);
        }
        check(static_cast<uint64>(File.Num()) == RunningOffset);

        if (!FFileHelper::SaveArrayToFile(File, *BinPath))
        {
            if (OutError) { *OutError = FString::Printf(TEXT("Failed to write %s"), *BinPath); }
            return false;
        }

        // ---- JSON sidecar (debug index, unchanged schema).
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("binary_file"), TEXT("meshes_v2.bin"));
        Root->SetNumberField(TEXT("binary_size"), static_cast<double>(File.Num()));
        TArray<TSharedPtr<FJsonValue>> MeshArr;
        for (int32 mi = 0; mi < NumMeshes; ++mi)
        {
            const FMeshAsset& M = Ctx.Meshes[mi];
            const FMeshAssetHeader& H = AssetHeaders[mi];
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"), M.Id.Value);
            J->SetStringField(TEXT("source_asset"), M.SourceAssetPath);
            J->SetStringField(TEXT("display_name"), M.DisplayName);
            J->SetNumberField(TEXT("kind"), static_cast<int32>(M.Kind));
            J->SetNumberField(TEXT("num_lods"), M.Lods.Num());
            J->SetNumberField(TEXT("payload_offset"), static_cast<double>(H.PayloadOffset));
            J->SetNumberField(TEXT("payload_size"),   static_cast<double>(H.PayloadSize));
            J->SetNumberField(TEXT("attribute_mask"), static_cast<double>(H.AttributeMask));
            TArray<TSharedPtr<FJsonValue>> SlotArr;
            for (const FStableId& S : M.SlotMaterialIds)
            {
                SlotArr.Add(MakeShared<FJsonValueString>(S.Value));
            }
            J->SetArrayField(TEXT("slot_material_ids"), SlotArr);
            MeshArr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("meshes"), MeshArr);
        return WriteJsonObject(JsonPath, Root, OutError);
    }

    bool WriteTextureContainer(const FExportContext& Ctx, FString* OutError)
    {
        using namespace SceneRTSceneExporterV2;

        const int32 NumTextures = Ctx.Textures.Num();

        // ── Pass 1: build per-texture payloads (pixel data if present) ──────────
        TArray<TArray<uint8>> TexPayloads;
        TexPayloads.SetNum(NumTextures);
        for (int32 ti = 0; ti < NumTextures; ++ti)
        {
            const FTextureRecord& Rec = Ctx.Textures[ti];
            if (Rec.Payload.Num() > 0)
            {
                TexPayloads[ti] = Rec.Payload; // pixel data extracted by Texture::Resolve
            }
            // else: PayloadSize will stay 0 — consumer can detect and skip
        }

        // ── Pass 2: compute layout ───────────────────────────────────────────────
        // File layout:
        //   [FTextureContainerHeader]
        //   [TOC: uint64 × NumTextures]   ← absolute offsets to each FTextureRecord
        //   [FTextureRecord × NumTextures]
        //   [payloads …]

        const uint64 HeaderSize  = sizeof(FTextureContainerHeader);
        const uint64 TocSize     = static_cast<uint64>(NumTextures) * sizeof(uint64);
        const uint64 RecordsSize = static_cast<uint64>(NumTextures) * sizeof(FTextureRecord);
        const uint64 RecordsBase = HeaderSize + TocSize;
        const uint64 PayloadsBase = RecordsBase + RecordsSize;

        // TOC entries point to FTextureRecord positions (not payloads).
        TArray<uint64> Toc;
        Toc.Reserve(NumTextures);
        for (int32 ti = 0; ti < NumTextures; ++ti)
        {
            Toc.Add(RecordsBase + static_cast<uint64>(ti) * sizeof(FTextureRecord));
        }

        // Compute total payload section size
        uint64 TotalPayloadSize = 0;
        for (int32 ti = 0; ti < NumTextures; ++ti)
        {
            TotalPayloadSize += static_cast<uint64>(TexPayloads[ti].Num());
        }
        const uint64 TotalSize = PayloadsBase + TotalPayloadSize;

        // ── Pass 3: fill FTextureRecord array with absolute payload offsets ──────
        TArray<FTextureRecord> Records;
        Records.Reserve(NumTextures);
        uint64 PayloadCursor = PayloadsBase;
        for (int32 ti = 0; ti < NumTextures; ++ti)
        {
            const FTextureRecord& Src = Ctx.Textures[ti];
            FTextureRecord Rec{};
            Rec.PixelFormat    = static_cast<uint16>(Src.PixelFormat);
            Rec.NumMips        = static_cast<uint16>(Src.NumMips);
            Rec.Width          = Src.Width;
            Rec.Height         = Src.Height;
            Rec.Depth          = Src.Depth;
            Rec.NumSlices      = Src.NumSlices;
            Rec.FlagsSrgb      = Src.bSrgb      ? 1u : 0u;
            Rec.FlagsNormalMap = Src.bNormalMap  ? 1u : 0u;
            Rec.FlagsHDR       = Src.bHDR        ? 1u : 0u;
            Rec.FlagsRVTBaked  = Src.bRvtBaked   ? 1u : 0u;
            Rec.FlagsCompositeBaked = 0u;
            Rec.FlagsReserved  = 0u;

            const uint64 PSize = static_cast<uint64>(TexPayloads[ti].Num());
            if (PSize > 0)
            {
                Rec.PayloadOffset = PayloadCursor;
                Rec.PayloadSize   = PSize;
                PayloadCursor += PSize;
            }
            else
            {
                Rec.PayloadOffset = 0;
                Rec.PayloadSize   = 0;
            }

            HashStableIdToBytes(Src.Id, Rec.StableIdHash);
            Records.Add(Rec);
        }

        // ── Pass 4: build container header ───────────────────────────────────────
        FTextureContainerHeader Hdr{};
        Hdr.Magic        = kV2TextureMagic;
        Hdr.VersionMajor = kV2FormatMajor;
        Hdr.VersionMinor = kV2FormatMinor;
        Hdr.NumTextures  = static_cast<uint32>(NumTextures);
        Hdr.Reserved     = 0;
        Hdr.TocOffset    = HeaderSize;
        Hdr.TotalSize    = TotalSize;
        Hdr.SceneGuidLo  = static_cast<uint64>(Ctx.SceneGuid.A) |
                           (static_cast<uint64>(Ctx.SceneGuid.B) << 32);
        Hdr.SceneGuidHi  = static_cast<uint64>(Ctx.SceneGuid.C) |
                           (static_cast<uint64>(Ctx.SceneGuid.D) << 32);

        // ── Pass 5: serialise to buffer ───────────────────────────────────────────
        TArray<uint8> Buf;
        Buf.Reserve(static_cast<int32>(TotalSize));

        AppendStruct(Buf, Hdr);
        // TOC
        for (const uint64 Off : Toc)
        {
            AppendStruct(Buf, Off);
        }
        // Records
        for (const FTextureRecord& R : Records)
        {
            AppendStruct(Buf, R);
        }
        // Payloads
        for (int32 ti = 0; ti < NumTextures; ++ti)
        {
            if (TexPayloads[ti].Num() > 0)
            {
                AppendBytes(Buf, TexPayloads[ti].GetData(), TexPayloads[ti].Num());
            }
        }

        // ── Write binary file ─────────────────────────────────────────────────────
        const FString BinPath = Ctx.OutputDir / TEXT("textures_v2.bin");
        if (!FFileHelper::SaveArrayToFile(Buf, *BinPath))
        {
            if (OutError) { *OutError = FString::Printf(TEXT("Failed to write %s"), *BinPath); }
            return false;
        }

        // ── JSON sidecar (human-readable descriptor table) ────────────────────────
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("scene_guid"), Ctx.SceneGuid.ToString());
        Root->SetNumberField(TEXT("num_textures"), NumTextures);
        Root->SetNumberField(TEXT("total_size"), static_cast<double>(TotalSize));

        TArray<TSharedPtr<FJsonValue>> Arr;
        for (int32 ti = 0; ti < NumTextures; ++ti)
        {
            const FTextureRecord& Src = Ctx.Textures[ti];
            const FTextureRecord& R   = Records[ti];
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"),          Src.Id.Value);
            J->SetStringField(TEXT("source_path"), Src.SourcePath);
            J->SetNumberField(TEXT("width"),       Src.Width);
            J->SetNumberField(TEXT("height"),      Src.Height);
            J->SetNumberField(TEXT("depth"),       Src.Depth);
            J->SetNumberField(TEXT("num_slices"),  Src.NumSlices);
            J->SetNumberField(TEXT("num_mips"),    Src.NumMips);
            J->SetNumberField(TEXT("pixel_format"),static_cast<int32>(Src.PixelFormat));
            J->SetBoolField  (TEXT("srgb"),        Src.bSrgb);
            J->SetBoolField  (TEXT("normal_map"),  Src.bNormalMap);
            J->SetBoolField  (TEXT("hdr"),         Src.bHDR);
            J->SetBoolField  (TEXT("rvt_baked"),   Src.bRvtBaked);
            J->SetNumberField(TEXT("payload_offset"), static_cast<double>(R.PayloadOffset));
            J->SetNumberField(TEXT("payload_size"),   static_cast<double>(R.PayloadSize));
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("textures"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("textures.json"), Root, OutError);
    }

    bool WriteSceneJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("scene_guid"), Ctx.SceneGuid.ToString());
        Root->SetStringField(TEXT("bundle_name"), Ctx.BundleName);

        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FPrimitiveInstance& P : Ctx.Primitives)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"), P.Id.Value);
            J->SetStringField(TEXT("mesh_id"), P.MeshId.Value);
            J->SetStringField(TEXT("actor"), P.OwnerActorPath);
            J->SetStringField(TEXT("component"), P.ComponentPath);
            J->SetObjectField(TEXT("transform"), SerializeTransform(P.Transform));
            J->SetBoolField(TEXT("visible"), P.bVisible);
            J->SetBoolField(TEXT("cast_shadow"), P.bCastShadow);
            J->SetBoolField(TEXT("affects_ray_tracing"), P.bAffectsRayTracing);
            J->SetNumberField(TEXT("lighting_channels"), static_cast<double>(P.LightingChannels));
            TArray<TSharedPtr<FJsonValue>> Slots;
            for (const FStableId& M : P.OverrideMaterialIds)
            {
                Slots.Add(MakeShared<FJsonValueString>(M.Value));
            }
            J->SetArrayField(TEXT("material_ids"), Slots);
            if (!P.SplineGroupId.IsEmpty())
            {
                TSharedRef<FJsonObject> Spl = MakeShared<FJsonObject>();
                Spl->SetStringField(TEXT("group_id"), P.SplineGroupId);
                Spl->SetNumberField(TEXT("segment_index"), P.SplineSegmentIndex);
                Spl->SetNumberField(TEXT("arclen_start"), P.SplineArclenStart);
                Spl->SetNumberField(TEXT("arclen_end"), P.SplineArclenEnd);
                J->SetObjectField(TEXT("spline"), Spl);
            }
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("primitives"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("scene.json"), Root, OutError);
    }

    bool WriteMaterialsJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FMaterialRecord& M : Ctx.Materials)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"), M.Id.Value);
            J->SetStringField(TEXT("parent"), M.ParentMaterialPath);
            J->SetStringField(TEXT("instance"), M.MaterialInstancePath);
            J->SetNumberField(TEXT("domain"), static_cast<int32>(M.Domain));
            J->SetNumberField(TEXT("shading_model"), static_cast<int32>(M.ShadingModel));
            J->SetBoolField(TEXT("two_sided"), M.bTwoSided);
            J->SetStringField(TEXT("resolved_hash"), FString::Printf(TEXT("%llx"), M.ResolvedParamsHash));
            TArray<TSharedPtr<FJsonValue>> Params;
            for (const FMaterialParamRecord& P : M.Params)
            {
                TSharedRef<FJsonObject> JP = MakeShared<FJsonObject>();
                JP->SetStringField(TEXT("name"), P.Name);
                JP->SetNumberField(TEXT("kind"), static_cast<int32>(P.Kind));
                JP->SetNumberField(TEXT("association"), P.ParameterAssociation);
                JP->SetNumberField(TEXT("layer_index"), P.LayerIndex);
                switch (P.Kind)
                {
                    case EParamKind::Scalar: JP->SetNumberField(TEXT("scalar"), P.Scalar); break;
                    case EParamKind::Vector:
                    {
                        TArray<TSharedPtr<FJsonValue>> V {
                            MakeShared<FJsonValueNumber>(P.Vector.R),
                            MakeShared<FJsonValueNumber>(P.Vector.G),
                            MakeShared<FJsonValueNumber>(P.Vector.B),
                            MakeShared<FJsonValueNumber>(P.Vector.A) };
                        JP->SetArrayField(TEXT("vector"), V);
                        break;
                    }
                    case EParamKind::Texture: JP->SetStringField(TEXT("texture_id"), P.TextureId.Value); break;
                    case EParamKind::Switch:  JP->SetBoolField(TEXT("switch"), P.bSwitch); break;
                    default: break;
                }
                Params.Add(MakeShared<FJsonValueObject>(JP));
            }
            J->SetArrayField(TEXT("params"), Params);
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("materials"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("materials.json"), Root, OutError);
    }

    bool WriteLightsJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FLightRecord& L : Ctx.Lights)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"), L.Id.Value);
            J->SetStringField(TEXT("type"), L.LightType);
            J->SetStringField(TEXT("actor"), L.OwnerActorPath);
            J->SetObjectField(TEXT("transform"), SerializeTransform(L.Transform));
            J->SetNumberField(TEXT("intensity"), L.Intensity);
            J->SetStringField(TEXT("intensity_units"), L.IntensityUnits);
            J->SetNumberField(TEXT("temperature"), L.Temperature);
            J->SetBoolField(TEXT("use_temperature"), L.bUseTemperature);
            J->SetBoolField(TEXT("cast_shadows"), L.bCastShadows);
            if (!L.IesProfileTextureId.Value.IsEmpty())
            {
                J->SetStringField(TEXT("ies_texture_id"), L.IesProfileTextureId.Value);
            }
            if (!L.SkyCubemapTextureId.Value.IsEmpty())
            {
                J->SetStringField(TEXT("sky_cubemap_id"), L.SkyCubemapTextureId.Value);
            }
            if (L.ExtraReflection.IsValid())
            {
                J->SetObjectField(TEXT("reflection"), L.ExtraReflection);
            }
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("lights"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("lights.json"), Root, OutError);
    }

    bool WriteDecalsJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("scene_guid"), Ctx.SceneGuid.ToString());

        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FDecalRecord& D : Ctx.Decals)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"),               D.Id.Value);
            J->SetStringField(TEXT("owner_actor"),      D.OwnerActorPath);
            J->SetStringField(TEXT("component"),        D.ComponentPath);
            J->SetStringField(TEXT("material_id"),      D.MaterialId.Value);
            J->SetObjectField(TEXT("transform"),        MakeTransformJson(D.Transform));
            TArray<TSharedPtr<FJsonValue>> Size {
                MakeShared<FJsonValueNumber>(D.DecalSize.X),
                MakeShared<FJsonValueNumber>(D.DecalSize.Y),
                MakeShared<FJsonValueNumber>(D.DecalSize.Z) };
            J->SetArrayField(TEXT("decal_size"),        Size);
            J->SetNumberField(TEXT("sort_order"),       D.SortOrder);
            J->SetNumberField(TEXT("fade_screen_size"), D.FadeScreenSize);
            J->SetNumberField(TEXT("opacity"),          D.Opacity);
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("decals"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("decals.json"), Root, OutError);
    }

    bool WriteCamerasJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("scene_guid"), Ctx.SceneGuid.ToString());

        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FCameraRecord& C : Ctx.Cameras)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"),           C.Id.Value);
            J->SetStringField(TEXT("owner_actor"),  C.OwnerActorPath);
            J->SetObjectField(TEXT("transform"),    MakeTransformJson(C.Transform));
            J->SetNumberField(TEXT("fov"),          C.FieldOfView);
            J->SetNumberField(TEXT("aspect_ratio"), C.AspectRatio);
            if (C.Focal > 0.f)        { J->SetNumberField(TEXT("focal_length"),    C.Focal); }
            if (C.Aperture > 0.f)     { J->SetNumberField(TEXT("aperture"),         C.Aperture); }
            if (C.FocusDistance > 0.f){ J->SetNumberField(TEXT("focus_distance"),   C.FocusDistance); }
            if (C.PostProcess.IsValid()){ J->SetObjectField(TEXT("post_process"),   C.PostProcess.ToSharedRef()); }
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("cameras"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("cameras.json"), Root, OutError);
    }

    bool WriteAtmosphereJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("scene_guid"), Ctx.SceneGuid.ToString());

        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FAtmosphereRecord& A : Ctx.Atmosphere)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("kind"),        A.Kind);
            J->SetStringField(TEXT("owner_actor"), A.OwnerActorPath);
            if (A.Settings.IsValid())
            {
                J->SetObjectField(TEXT("settings"), A.Settings.ToSharedRef());
            }
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("atmosphere"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("atmosphere.json"), Root, OutError);
    }

    bool WritePostProcessJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FPostProcessRecord& P : Ctx.PostProcess)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"), P.Id.Value);
            J->SetStringField(TEXT("actor"), P.OwnerActorPath);
            J->SetNumberField(TEXT("priority"), P.Priority);
            J->SetNumberField(TEXT("blend_radius"), P.BlendRadius);
            J->SetNumberField(TEXT("blend_weight"), P.BlendWeight);
            J->SetBoolField(TEXT("unbound"), P.bUnbound);
            if (P.Settings.IsValid())
            {
                J->SetObjectField(TEXT("settings"), P.Settings);
            }
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("postprocess"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("postprocess.json"), Root, OutError);
    }

    bool WriteManifest(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        Root->SetStringField(TEXT("scene_guid"), Ctx.SceneGuid.ToString());
        Root->SetStringField(TEXT("bundle_name"), Ctx.BundleName);
        Root->SetNumberField(TEXT("format_major"), SceneRTSceneExporterV2::kV2FormatMajor);
        Root->SetNumberField(TEXT("format_minor"), SceneRTSceneExporterV2::kV2FormatMinor);
        TArray<TSharedPtr<FJsonValue>> Files;
        const TCHAR* Names[] = {
            TEXT("meshes_v2.bin"), TEXT("meshes.json"),
            TEXT("textures_v2.bin"), TEXT("textures.json"),
            TEXT("materials.json"), TEXT("scene.json"),
            TEXT("lights.json"), TEXT("decals.json"),
            TEXT("cameras.json"), TEXT("atmosphere.json"),
            TEXT("postprocess.json"), TEXT("landscapes.json"),
        };
        for (const TCHAR* N : Names)
        {
            Files.Add(MakeShared<FJsonValueString>(N));
        }
        Root->SetArrayField(TEXT("files"), Files);
        Root->SetNumberField(TEXT("counts_meshes"),      Ctx.Meshes.Num());
        Root->SetNumberField(TEXT("counts_materials"),   Ctx.Materials.Num());
        Root->SetNumberField(TEXT("counts_textures"),    Ctx.Textures.Num());
        Root->SetNumberField(TEXT("counts_primitives"),  Ctx.Primitives.Num());
        Root->SetNumberField(TEXT("counts_lights"),      Ctx.Lights.Num());
        Root->SetNumberField(TEXT("counts_decals"),      Ctx.Decals.Num());
        Root->SetNumberField(TEXT("counts_cameras"),     Ctx.Cameras.Num());
        Root->SetNumberField(TEXT("counts_atmosphere"),  Ctx.Atmosphere.Num());
        Root->SetNumberField(TEXT("counts_postprocess"), Ctx.PostProcess.Num());
        Root->SetNumberField(TEXT("counts_landscapes"),  Ctx.Landscapes.Num());
        Root->SetNumberField(TEXT("counts_issues"),      Ctx.Issues.Num());
        return WriteJsonObject(Ctx.OutputDir / TEXT("manifest.json"), Root, OutError);
    }

    bool WriteLandscapesJson(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FLandscapeRecord& L : Ctx.Landscapes)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("id"), L.Id.Value);
            J->SetStringField(TEXT("actor"), L.ActorPath);
            J->SetStringField(TEXT("landscape_guid"), L.LandscapeGuid);
            J->SetObjectField(TEXT("transform"), SerializeTransform(L.Transform));
            J->SetNumberField(TEXT("component_size_quads"), L.ComponentSizeQuads);
            J->SetNumberField(TEXT("subsection_size_quads"), L.SubsectionSizeQuads);
            J->SetNumberField(TEXT("num_subsections"), L.NumSubsections);
            J->SetNumberField(TEXT("export_lod"), L.ExportLOD);
            J->SetStringField(TEXT("base_material_id"), L.BaseMaterialId.Value);
            J->SetStringField(TEXT("hole_material_id"), L.HoleMaterialId.Value);

            TArray<TSharedPtr<FJsonValue>> CompArr;
            for (const FLandscapeComponentInfo& C : L.Components)
            {
                TSharedRef<FJsonObject> JC = MakeShared<FJsonObject>();
                JC->SetStringField(TEXT("mesh_id"), C.MeshId.Value);
                JC->SetStringField(TEXT("instance_id"), C.InstanceId.Value);
                JC->SetStringField(TEXT("component"), C.ComponentPath);
                JC->SetNumberField(TEXT("section_base_x"), C.SectionBaseX);
                JC->SetNumberField(TEXT("section_base_y"), C.SectionBaseY);
                if (!C.HeightmapTextureId.Value.IsEmpty())
                {
                    JC->SetStringField(TEXT("heightmap_id"), C.HeightmapTextureId.Value);
                }
                TArray<TSharedPtr<FJsonValue>> LayerArr;
                for (const FLandscapeLayerWeight& Lw : C.LayerWeights)
                {
                    TSharedRef<FJsonObject> JL = MakeShared<FJsonObject>();
                    JL->SetStringField(TEXT("layer"), Lw.LayerName);
                    JL->SetStringField(TEXT("weight_texture_id"), Lw.WeightTextureId.Value);
                    JL->SetNumberField(TEXT("channel"), Lw.ChannelIndex);
                    TArray<TSharedPtr<FJsonValue>> Color {
                        MakeShared<FJsonValueNumber>(Lw.PhysicalMaterialBaseColor.R),
                        MakeShared<FJsonValueNumber>(Lw.PhysicalMaterialBaseColor.G),
                        MakeShared<FJsonValueNumber>(Lw.PhysicalMaterialBaseColor.B),
                        MakeShared<FJsonValueNumber>(Lw.PhysicalMaterialBaseColor.A) };
                    JL->SetArrayField(TEXT("debug_color"), Color);
                    LayerArr.Add(MakeShared<FJsonValueObject>(JL));
                }
                JC->SetArrayField(TEXT("layer_weights"), LayerArr);
                CompArr.Add(MakeShared<FJsonValueObject>(JC));
            }
            J->SetArrayField(TEXT("components"), CompArr);
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("landscapes"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("landscapes.json"), Root, OutError);
    }

    bool WriteValidationReport(const FExportContext& Ctx, FString* OutError)
    {
        TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> Arr;
        for (const FValidationIssue& I : Ctx.Issues)
        {
            TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
            J->SetStringField(TEXT("severity"), I.Severity);
            J->SetStringField(TEXT("category"), I.Category);
            J->SetStringField(TEXT("message"), I.Message);
            J->SetStringField(TEXT("related_id"), I.RelatedId);
            Arr.Add(MakeShared<FJsonValueObject>(J));
        }
        Root->SetArrayField(TEXT("issues"), Arr);
        return WriteJsonObject(Ctx.OutputDir / TEXT("validation.json"), Root, OutError);
    }
}
