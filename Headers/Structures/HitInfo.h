#pragma once
#include "Point3D.h"
#include "UV.h"

struct HitInfo
{
    bool  hit      = false;
    float t        = 0.0f;
    int   triIndex = -1;

    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 normal{0.0f, 1.0f, 0.0f};

    Vec3 color{1.0f, 1.0f, 1.0f};
    Vec3 emission{0.0f, 0.0f, 0.0f};

    float metallic  = 0.0f;
    float roughness = 0.5f;

    // 1.0f — полностью виден источником света, 0.0f — в полной тени
    float visibility = 1.0f;

    Vec2 uv{0.0f, 0.0f};
};
