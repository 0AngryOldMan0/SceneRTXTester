#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#include "Triangles.h"
#include "BVHNode.h"
#include "CameraData.h"
#include "Light.h"

// Хостовая копия структуры LightGPU для CUDA.
// ДОЛЖНА совпадать по layout'у с LightGPU в RayTraceCuda.cu.
struct CudaLightGPUHost
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

// Если sizeof(Vec3) == 12, то sizeof(CudaLightGPUHost) должно быть 60.
static_assert(sizeof(Vec3) == 12, "Vec3 must be 12 bytes");
static_assert(sizeof(CudaLightGPUHost) == 60,
              "CudaLightGPUHost must be 60 bytes (match CUDA LightGPU)");

// ---------------------------------------------------------------
// Низкоуровневый C-интерфейс, экспортируемый из RayTraceCuda.cu.
// Работает с сырыми указателями и байтами, без STL.
// ---------------------------------------------------------------

extern "C" bool CUDA_RenderFrame_C(
    const void *nodes, // BVHNode[]
    std::size_t nodeCount,
    const void *triangles, // Triangle[]
    std::size_t triCount,
    const void *lights, // CudaLightGPUHost[]
    std::size_t lightCount,
    int rootIndex,
    const void *camera, // CameraDataCPU (layout == CameraData в .cu)
    void *framebuffer   // Vec3[] (layout == float3[])
);

extern "C" bool CUDA_RenderFrameAccum_C(
    const void *nodes, // BVHNode[]
    std::size_t nodeCount,
    const void *triangles, // Triangle[]
    std::size_t triCount,
    const void *lights, // CudaLightGPUHost[]
    std::size_t lightCount,
    int rootIndex,
    const void *camera, // CameraDataCPU
    void *framebuffer   // Vec3[]
);

extern "C" void CUDA_ResetAccumulation_C();
bool InitCudaRenderer();

// Однократный рендер без накопления (ядро RayTraceKernel)
bool RenderFrameCuda(const std::vector<BVHNode> &nodes,
                     const std::vector<Triangle> &tris,
                     const std::vector<Light> &lights,
                     int rootIndex,
                     const CameraDataCPU &cameraCPU,
                     std::vector<Vec3> &framebuffer);

// Прогрессивный рендер с накоплением (ядро RayTraceAccumulateKernel,
// если ты его добавишь в RayTraceCuda.cu; иначе можно будет сделать
// просто алиас на RenderFrameCuda)
bool RenderFrameCudaAccumulate(const std::vector<BVHNode> &nodes,
                               const std::vector<Triangle> &tris,
                               const std::vector<Light> &lights,
                               int rootIndex,
                               const CameraDataCPU &cameraCPU,
                               std::vector<Vec3> &framebuffer);

// Сброс накопительного состояния (счётчика кадров / буфера) на стороне CUDA
void ResetCudaAccumulation();