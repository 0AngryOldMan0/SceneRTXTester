#pragma once

#include "Point3D.h"

struct CameraDataCPU
{
    Vec3 position; //  0
    Vec3 forward;  // 12
    Vec3 up;       // 24
    Vec3 right;    // 36

    float fovY;          // 48
    int   width;         // 52
    int   height;        // 56
    int   samplesPerPixel; // 60

    float nearPlane;     // 64
    float farPlane;      // 68
    float focusDistance; // 72
    float pad;           // 76 -> всего 80
};

static_assert(sizeof(CameraDataCPU) == 80); // Необходимое преобразование для сопоставления размера типа данных на ЦПУ и ГПУ