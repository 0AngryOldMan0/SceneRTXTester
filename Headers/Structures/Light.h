#pragma once

#include "Point3D.h"

enum class LightType
{
    Point       = 0,       // точечный источник (Blender: POINT)
    Directional = 1, // направленный / солнце (Blender: SUN)
    Spot        = 2,        // прожектор (Blender: SPOT)
    Area        = 3         // площадной (Blender: AREA)
};

// Форма area-источника в Blender: SQUARE, RECTANGLE, DISK, ELLIPSE
enum class AreaLightShape
{
    Square      = 0,
    Rectangle   = 1,
    Disk        = 2,
    Ellipse     = 3
};

struct Light
{
    LightType type = LightType::Point;

    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 direction{0.0f, -1.0f, 0.0f};

    Vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;

    // Геометрический размер источника (для мягких теней / выборки по площади)
    float radius = 0.0f;

    // UE point/spot source shape parameters.
    // sourceLength extends the light into a tube/capsule along light.direction.
    // softSourceRadius expands the effective glossy/penumbra footprint.
    float sourceLength = 0.0f;
    float softSourceRadius = 0.0f;

    // Радиус затухания из UE (AttenuationRadius).
    // 0.0f означает «старое поведение» — бесконечный 1/r^2.
    float attenuationRadius = 0.0f;

    // Параметры Spot
    float spotSize = 0.0f;
    float spotBlend = 0.0f;

    // Параметры Area
    AreaLightShape areaShape = AreaLightShape::Square;
    float areaSizeX = 0.0f;
    float areaSizeY = 0.0f;
};
