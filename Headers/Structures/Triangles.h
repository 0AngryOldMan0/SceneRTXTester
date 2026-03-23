#pragma once
#include "Point3D.h"
#include "Point4D.h"
#include "UV.h"
#include "AABB.h"

#include <cstdint>
#include <type_traits>

// CPU Triangle layout MUST match Metal Triangle layout (see Shader/macOS/RayTrace.metal).
// UVs are stored as 3 UV sets x 3 triangle vertices in channel-major order:
// uv[set * 3 + vertex].
// Per-vertex normals/tangents are stored as packed 10-bit signed unit vectors.
// Tangent.w stores handedness in bit 30 (0 = +1, 1 = -1).
// Per-vertex colors are stored as packed RGBA8 values.
// Target size: 220 bytes.
struct Triangle
{
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    std::uint32_t vertexNormal[3] = {
        0x3FF80200u,
        0x3FF80200u,
        0x3FF80200u
    };

    std::uint32_t vertexTangent[3] = {
        0x20080200u,
        0x20080200u,
        0x20080200u
    };

    Vec2 uv[9];

    Vec3 normal;
    AABB ABoBa;

    std::uint32_t vertexColor[3] = {
        0xFFFFFFFFu,
        0xFFFFFFFFu,
        0xFFFFFFFFu
    };

    Vec3 color;
    Vec3 emission;

    float   metallic;
    float   roughness;

    int32_t materialIndex = -1;
    float   _padMat       = 0.0f;
};

static_assert(sizeof(Triangle) == 220, "Triangle size must be 220 bytes");
static_assert(std::is_trivially_copyable_v<Triangle>, "Triangle must stay trivially copyable");

static inline constexpr int TriangleUvIndex(int uvSet, int vertexIndex)
{
    return uvSet * 3 + vertexIndex;
}

static inline Vec2& TriangleUV(Triangle& tri, int uvSet, int vertexIndex)
{
    return tri.uv[TriangleUvIndex(uvSet, vertexIndex)];
}

static inline const Vec2& TriangleUV(const Triangle& tri, int uvSet, int vertexIndex)
{
    return tri.uv[TriangleUvIndex(uvSet, vertexIndex)];
}
