#include "HIPRendererFunctions.h"
#include <iostream>

// На будущее: сюда можно добавить проверку наличия HIP-устройства и т.п.
bool InitHIPRenderer()
{
    // Пока считаем, что HIP сам корректно инициализируется при первом обращении.
    return true;
}

// Вспомогательная функция: конвертация Light -> LightGPUHost
static void ConvertLightsToGPU(const std::vector<Light> &srcLights,
                               std::vector<LightGPUHost> &dstLights)
{
    dstLights.clear();
    dstLights.reserve(srcLights.size());

    for (const Light &src : srcLights)
    {
        LightGPUHost gpu{};
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

// Однократный рендер без накопления
bool RenderFrameHIP(const std::vector<BVHNode> &nodes,
                    const std::vector<Triangle> &tris,
                    const std::vector<Light> &lights,
                    int rootIndex,
                    const CameraDataCPU &cameraCPU,
                    std::vector<Vec3> &framebuffer)
{
#ifndef USE_HIP_RENDERER
    (void)nodes;
    (void)tris;
    (void)lights;
    (void)rootIndex;
    (void)cameraCPU;
    (void)framebuffer;
    std::cerr << "RenderFrameHIP: проект собран без USE_HIP_RENDERER\n";
    return false;
#else
    const int width = cameraCPU.width;
    const int height = cameraCPU.height;

    if (width <= 0 || height <= 0)
    {
        std::cerr << "RenderFrameHIP: некорректный размер кадра (" << width << "x" << height << ")\n";
        return false;
    }

    if (nodes.empty() || tris.empty())
    {
        std::cerr << "RenderFrameHIP: пустые BVH или треугольники\n";
        return false;
    }

    // Подготавливаем буфер кадра под Vec3
    framebuffer.resize(static_cast<std::size_t>(width) *
                       static_cast<std::size_t>(height));

    // Конвертируем источники света в компактный LightGPUHost
    std::vector<LightGPUHost> gpuLights;
    ConvertLightsToGPU(lights, gpuLights);

    const void *nodesPtr = nodes.data();
    const void *trisPtr = tris.data();
    const void *lightsPtr = gpuLights.empty() ? nullptr : gpuLights.data();
    const void *cameraPtr = &cameraCPU;
    void *framebufferPtr = framebuffer.data();

    bool ok = HIP_RenderFrame_C(
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
        std::cerr << "RenderFrameHIP: HIP_RenderFrame_C вернул false\n";
    }

    return ok;
#endif
}

// Прогрессивный рендер с накоплением
bool RenderFrameHIPAccumulate(const std::vector<BVHNode> &nodes,
                              const std::vector<Triangle> &tris,
                              const std::vector<Light> &lights,
                              int rootIndex,
                              const CameraDataCPU &cameraCPU,
                              std::vector<Vec3> &framebuffer)
{
#ifndef USE_HIP_RENDERER
    (void)nodes;
    (void)tris;
    (void)lights;
    (void)rootIndex;
    (void)cameraCPU;
    (void)framebuffer;
    std::cerr << "RenderFrameHIPAccumulate: проект собран без USE_HIP_RENDERER\n";
    return false;
#else
    const int width = cameraCPU.width;
    const int height = cameraCPU.height;

    if (width <= 0 || height <= 0)
    {
        std::cerr << "RenderFrameHIPAccumulate: некорректный размер кадра (" << width << "x" << height << ")\n";
        return false;
    }

    if (nodes.empty() || tris.empty())
    {
        std::cerr << "RenderFrameHIPAccumulate: пустые BVH или треугольники\n";
        return false;
    }

    framebuffer.resize(static_cast<std::size_t>(width) *
                       static_cast<std::size_t>(height));

    std::vector<LightGPUHost> gpuLights;
    ConvertLightsToGPU(lights, gpuLights);

    const void *nodesPtr = nodes.data();
    const void *trisPtr = tris.data();
    const void *lightsPtr = gpuLights.empty() ? nullptr : gpuLights.data();
    const void *cameraPtr = &cameraCPU;
    void *framebufferPtr = framebuffer.data();

    bool ok = HIP_RenderFrameAccum_C(
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
        std::cerr << "RenderFrameHIPAccumulate: HIP_RenderFrameAccum_C вернул false\n";
    }

    return ok;
#endif
}

// Сброс счётчика кадров / накопительной текстуры на GPU
void ResetHIPAccumulation()
{
#ifdef USE_HIP_RENDERER
    HIP_ResetAccumulation_C();
#endif
}