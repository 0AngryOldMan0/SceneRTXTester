#pragma once
#include "Point3D.h"
#include "UV.h"
#include "AABB.h"

#include <cstdint>

// CPU Triangle layout MUST match Metal Triangle layout (see Shader/macOS/RayTrace.metal).
// UVs are stored as 3 UV sets x 3 triangle vertices in channel-major order:
// uv[set * 3 + vertex].
// Target size: 184 bytes.
struct Triangle
{
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    Vec2 uv[9];

    Vec3 normal;
    AABB ABoBa;

    Vec3 color;
    Vec3 emission;

    float   metallic;
    float   roughness;

    int32_t materialIndex = -1;
    float   _padMat       = 0.0f;
};

static_assert(sizeof(Triangle) == 184, "Triangle size must be 184 bytes");

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
