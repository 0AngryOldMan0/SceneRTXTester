#pragma once

#include "CoreMinimal.h"

namespace SceneRTExportFormat
{
    static constexpr uint32 MeshBinaryVersion = 1;
    static constexpr ANSICHAR MeshMagic[8] = { 'S', 'R', 'T', 'X', 'M', 'S', 'H', '\0' };

#pragma pack(push, 1)
    struct FMeshFileHeader
    {
        ANSICHAR Magic[8];
        uint32 Version = MeshBinaryVersion;
        uint32 HeaderSize = sizeof(FMeshFileHeader);
        uint64 MeshCount = 0;
        uint64 Reserved0 = 0;
        uint64 Reserved1 = 0;
    };

    struct FMeshBlockHeader
    {
        uint32 HeaderSize = sizeof(FMeshBlockHeader);
        uint32 LodCount = 0;
        uint64 StableIdHash = 0;
        uint64 PayloadSize = 0;
        uint64 Reserved0 = 0;
        uint64 Reserved1 = 0;
    };

    struct FLODHeader
    {
        uint32 LodIndex = 0;
        uint32 VertexCount = 0;
        uint32 IndexCount = 0;
        uint32 UVSetCount = 0;
        uint32 SectionCount = 0;
        uint32 Flags = 0;

        uint64 PositionsOffset = 0;
        uint64 NormalsOffset = 0;
        uint64 TangentsOffset = 0;
        uint64 ColorsOffset = 0;
        uint64 UVsOffset = 0;
        uint64 IndicesOffset = 0;
        uint64 SectionsOffset = 0;
    };

    struct FSectionRange
    {
        uint32 SectionIndex = 0;
        uint32 MaterialSlotIndex = 0;
        uint32 FirstIndex = 0;
        uint32 IndexCount = 0;
    };
#pragma pack(pop)

    static_assert(sizeof(FMeshFileHeader) == 40, "Unexpected FMeshFileHeader size");
    static_assert(sizeof(FMeshBlockHeader) == 40, "Unexpected FMeshBlockHeader size");
    static_assert(sizeof(FSectionRange) == 16, "Unexpected FSectionRange size");
}
