#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#include "Structures/Triangles.h"
#include "Structures/BVHNode.h"
#include "Structures/CameraData.h"
#include "Structures/Light.h"

// Хостовая копия структуры LightGPU, которую будем отправлять на GPU.
// Должна совпадать по размеру с GPU-версией (60 байт).
struct LightGPUHost
{
    int type;  // LightType как int
    int _pad0; // паддинг до 8 байт

    Vec3 position;  // 12 байт
    Vec3 direction; // 12 байт
    Vec3 color;     // 12 байт

    float intensity; // 4 байта
    float radius;    // 4 байта
    float spotSize;  // 4 байта
    float spotBlend; // 4 байта
};

static_assert(sizeof(LightGPUHost) == 60, "LightGPUHost must be 60 bytes");

// ---------------------------------------------------------------
// Низкоуровневый C-интерфейс, экспортируемый из RayTraceHIP.hip.cpp
// (работает с сырыми указателями и байтами).
// ---------------------------------------------------------------

extern "C" bool HIP_RenderFrame_C(
    const void *nodes,
    std::size_t nodeCount,
    const void *triangles,
    std::size_t triCount,
    const void *lights, // массив LightGPU (в хостовой раскладке)
    std::size_t lightCount,
    int rootIndex,
    const void *camera, // CameraDataCPU (layout == CameraData в HIP)
    void *framebuffer   // массив Vec3 (layout == float3[])
);

extern "C" bool HIP_RenderFrameAccum_C(
    const void *nodes,
    std::size_t nodeCount,
    const void *triangles,
    std::size_t triCount,
    const void *lights, // массив LightGPU (в хостовой раскладке)
    std::size_t lightCount,
    int rootIndex,
    const void *camera, // CameraDataCPU
    void *framebuffer   // массив Vec3
);

extern "C" void HIP_ResetAccumulation_C();

// ---------------------------------------------------------------
// Высокоуровневый C++ API, который будет использовать твой рендер
// ---------------------------------------------------------------

// Пока инициализация ничего особенного не делает — HIP сам поднимет контекст
// при первом вызове hip* API. Оставляем как "hook" на будущее.
bool InitHIPRenderer();

// Однократный рендер без накопления (одно ядро RayTraceKernel)
bool RenderFrameHIP(const std::vector<BVHNode> &nodes,
                    const std::vector<Triangle> &tris,
                    const std::vector<Light> &lights,
                    int rootIndex,
                    const CameraDataCPU &cameraCPU,
                    std::vector<Vec3> &framebuffer);

// Прогрессивный рендер с накоплением (ядро RayTraceAccumulateKernel)
bool RenderFrameHIPAccumulate(const std::vector<BVHNode> &nodes,
                              const std::vector<Triangle> &tris,
                              const std::vector<Light> &lights,
                              int rootIndex,
                              const CameraDataCPU &cameraCPU,
                              std::vector<Vec3> &framebuffer);

// Сброс накопительной текстуры / счётчика кадров
void ResetHIPAccumulation();