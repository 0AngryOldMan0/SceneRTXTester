#pragma once
#include "Point3D.h"
#include "UV.h"
#include "AABB.h"

#include <cstdint>

// CPU Triangle layout MUST match Metal Triangle layout (see Shader/macOS/RayTrace.metal).
// Target size: 136 bytes.
struct Triangle
{
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;

    Vec3 normal;
    AABB ABoBa;

    Vec3 color;
    Vec3 emission;

    float   metallic;
    float   roughness;

    int32_t materialIndex = -1;
    float   _padMat       = 0.0f;
};

static_assert(sizeof(Triangle) == 136, "Triangle size must be 136 bytes");
