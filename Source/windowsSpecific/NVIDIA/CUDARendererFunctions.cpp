#include "CudaRendererFunctions.h"
#include <iostream>

// ----------------------------------------------------------
// Вспомогательная функция: Light -> CudaLightGPUHost
// ----------------------------------------------------------

static void ConvertLightsToCudaGPU(const std::vector<Light> &srcLights,
                                   std::vector<CudaLightGPUHost> &dstLights)
{
    dstLights.clear();
    dstLights.reserve(srcLights.size());

    for (const Light &src : srcLights)
    {
        CudaLightGPUHost gpu{};
        gpu.type = static_cast<int>(src.type);
        gpu._pad0 = 0;

        gpu.position = src.position;
        gpu.direction = src.direction;
        gpu.color = src.color;

        gpu.intensity = src.intensity;
        gpu.radius = src.radius;
        gpu.spotSize = src.spotSize;
        gpu.spotBlend = src.spotBlend;

        dstLights.push_back(gpu);
    }
}

// ----------------------------------------------------------
// High-level API
// ----------------------------------------------------------

bool InitCudaRenderer()
{
#ifdef USE_CUDA_RENDERER
    // Пока ничего особенного не делаем – CUDA поднимет контекст сама.
    // Здесь можно будет добавить cudaGetDeviceCount/cudaSetDevice и т.п.
    return true;
#else
    std::cerr << "InitCudaRenderer: проект собран без USE_CUDA_RENDERER\n";
    return false;
#endif
}

bool RenderFrameCuda(const std::vector<BVHNode> &nodes,
                     const std::vector<Triangle> &tris,
                     const std::vector<Light> &lights,
                     int rootIndex,
                     const CameraDataCPU &cameraCPU,
                     std::vector<Vec3> &framebuffer)
{
#ifndef USE_CUDA_RENDERER
    (void)nodes;
    (void)tris;
    (void)lights;
    (void)rootIndex;
    (void)cameraCPU;
    (void)framebuffer;
    std::cerr << "RenderFrameCuda: проект собран без USE_CUDA_RENDERER\n";
    return false;
#else
    const int width = cameraCPU.width;
    const int height = cameraCPU.height;

    if (width <= 0 || height <= 0)
    {
        std::cerr << "RenderFrameCuda: некорректный размер кадра ("
                  << width << "x" << height << ")\n";
        return false;
    }

    if (nodes.empty() || tris.empty())
    {
        std::cerr << "RenderFrameCuda: пустые BVH или треугольники\n";
        return false;
    }

    // Подготавливаем буфер кадра под Vec3
    framebuffer.resize(static_cast<std::size_t>(width) *
                       static_cast<std::size_t>(height));

    // Конвертируем источники света в компактный формат для GPU
    std::vector<CudaLightGPUHost> gpuLights;
    ConvertLightsToCudaGPU(lights, gpuLights);

    const void *nodesPtr = nodes.data();
    const void *trisPtr = tris.data();
    const void *lightsPtr = gpuLights.empty() ? nullptr : gpuLights.data();
    const void *cameraPtr = &cameraCPU;
    void *framebufferPtr = framebuffer.data();

    bool ok = CUDA_RenderFrame_C(
        nodesPtr,
        nodes.size(),
        trisPtr,
        tris.size(),
        lightsPtr,
        gpuLights.size(),
        rootIndex,
        cameraPtr,
        framebufferPtr);

    if (!ok)
    {
        std::cerr << "RenderFrameCuda: CUDA_RenderFrame_C вернул false\n";
    }

    return ok;
#endif
}

bool RenderFrameCudaAccumulate(const std::vector<BVHNode> &nodes,
                               const std::vector<Triangle> &tris,
                               const std::vector<Light> &lights,
                               int rootIndex,
                               const CameraDataCPU &cameraCPU,
                               std::vector<Vec3> &framebuffer)
{
#ifndef USE_CUDA_RENDERER
    (void)nodes;
    (void)tris;
    (void)lights;
    (void)rootIndex;
    (void)cameraCPU;
    (void)framebuffer;
    std::cerr << "RenderFrameCudaAccumulate: проект собран без USE_CUDA_RENDERER\n";
    return false;
#else
    const int width = cameraCPU.width;
    const int height = cameraCPU.height;

    if (width <= 0 || height <= 0)
    {
        std::cerr << "RenderFrameCudaAccumulate: некорректный размер кадра ("
                  << width << "x" << height << ")\n";
        return false;
    }

    if (nodes.empty() || tris.empty())
    {
        std::cerr << "RenderFrameCudaAccumulate: пустые BVH или треугольники\n";
        return false;
    }

    framebuffer.resize(static_cast<std::size_t>(width) *
                       static_cast<std::size_t>(height));

    std::vector<CudaLightGPUHost> gpuLights;
    ConvertLightsToCudaGPU(lights, gpuLights);

    const void *nodesPtr = nodes.data();
    const void *trisPtr = tris.data();
    const void *lightsPtr = gpuLights.empty() ? nullptr : gpuLights.data();
    const void *cameraPtr = &cameraCPU;
    void *framebufferPtr = framebuffer.data();

    bool ok = CUDA_RenderFrameAccum_C(
        nodesPtr,
        nodes.size(),
        trisPtr,
        tris.size(),
        lightsPtr,
        gpuLights.size(),
        rootIndex,
        cameraPtr,
        framebufferPtr);

    if (!ok)
    {
        std::cerr << "RenderFrameCudaAccumulate: CUDA_RenderFrameAccum_C вернул false\n";
    }

    return ok;
#endif
}

void ResetCudaAccumulation()
{
#ifdef USE_CUDA_RENDERER
    CUDA_ResetAccumulation_C();
#else
    std::cerr << "ResetCudaAccumulation: проект собран без USE_CUDA_RENDERER\n";
#endif
}