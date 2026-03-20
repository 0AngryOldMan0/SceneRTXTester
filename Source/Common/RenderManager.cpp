#include <fstream>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
//======
#include "RenderManager.h"
#include "Scene.h"
#include "Camera.h"
//======
#ifdef USE_HIP_RENDERER
#include "HIPRenderer.h"
#endif
#ifdef USE_CUDA_RENDERER
#include "CUDARenderer.h"
#endif
#ifdef USE_METAL_RENDERER
#include "MetalRenderer.h"
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ExternalLibs/stb-master/stb_image_write.h"
#pragma clang diagnostic pop

namespace
{
    inline float clamp01f(float v)
    {
        return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
    }

    // Reuses pixel buffer to avoid repeated allocations while saving multiple frames.
    void SaveFrameBufferToPNG(const std::vector<Vec3> &framebuffer,
                              int width,
                              int height,
                              const std::string &filename)
    {
        if (width <= 0 || height <= 0)
            throw std::runtime_error("Некорректный размер изображения при сохранении PNG");

        const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        if (framebuffer.size() != pixelCount)
            throw std::runtime_error("Некорректный размер framebuffer при сохранении PNG");

        static thread_local std::vector<unsigned char> pixels;
        pixels.resize(pixelCount * 3u);

        const Vec3 *src = framebuffer.data();
        unsigned char *dst = pixels.data();

        for (std::size_t i = 0; i < pixelCount; ++i)
        {
            const Vec3 &c = src[i];

            const float r = clamp01f(c.x);
            const float g = clamp01f(c.y);
            const float b = clamp01f(c.z);

            const std::size_t p = i * 3u;
            dst[p + 0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
            dst[p + 1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
            dst[p + 2] = static_cast<unsigned char>(b * 255.0f + 0.5f);
        }

        const int stride = width * 3;
        const int ok = stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), stride);
        if (!ok)
            throw std::runtime_error("Не удалось записать PNG-файл: " + filename);
    }

    std::vector<int> BuildMetalSppSchedule(TextureRenderMode mode,
                                           int baseSamplesPerPixel)
    {
        constexpr int kPreviewSpp = 2;
        constexpr int kMaxStableDispatchSpp = 4;
        constexpr int kDefaultProgressivePassCount = 4;

        std::vector<int> schedule;
        if (mode == TextureRenderMode::Preview)
        {
            schedule.push_back(std::max(1, std::min(baseSamplesPerPixel, kPreviewSpp)));
            return schedule;
        }

        const int requestedSpp = std::max(1, baseSamplesPerPixel);
        const int dispatchSpp = std::min(requestedSpp, kMaxStableDispatchSpp);
        const int targetTotalSpp =
            (requestedSpp <= kMaxStableDispatchSpp)
                ? (dispatchSpp * kDefaultProgressivePassCount)
                : requestedSpp;

        int accumulatedSpp = 0;
        schedule.reserve(static_cast<std::size_t>((targetTotalSpp + dispatchSpp - 1) / dispatchSpp));
        while (accumulatedSpp < targetTotalSpp)
        {
            const int remainingSpp = targetTotalSpp - accumulatedSpp;
            const int batchSpp = std::min(dispatchSpp, remainingSpp);
            schedule.push_back(batchSpp);
            accumulatedSpp += batchSpp;
        }

        return schedule;
    }
}

RenderManager::RenderManager()
{
#ifdef USE_CUDA_RENDERER
    addRenderer(std::make_unique<CudaRenderer>());
#endif
#ifdef USE_HIP_RENDERER
    addRenderer(std::make_unique<HIPRenderer>());
#endif
#ifdef USE_METAL_RENDERER
    addRenderer(std::make_unique<MetalRenderer>());
#endif
}

void RenderManager::addRenderer(std::unique_ptr<Renderer> renderer)
{
    if (!renderer)
        return;

    // Ensure unique names (rarely needed; usually you have 1 instance per backend)
    std::string name = renderer->getName();
    int counter = 1;
    while (getRenderer(name) != nullptr)
        name = renderer->getName() + "_" + std::to_string(counter++);

    renderers_.push_back(std::move(renderer));
}

void RenderManager::removeRenderer(const std::string &name)
{
    const auto it = std::find_if(renderers_.begin(), renderers_.end(),
                                 [&](const std::unique_ptr<Renderer> &r)
                                 { return r && r->getName() == name; });

    if (it != renderers_.end())
        renderers_.erase(it);
}

Renderer *RenderManager::getRenderer(const std::string &name) const
{
    const auto it = std::find_if(renderers_.begin(), renderers_.end(),
                                 [&](const std::unique_ptr<Renderer> &r)
                                 { return r && r->getName() == name; });

    return (it != renderers_.end()) ? it->get() : nullptr;
}

bool RenderManager::renderFrame(const Scene &scene,
                                const Camera &camera,
                                const std::string &rendererName,
                                const std::string &outputFilename)
{
    Renderer *renderer = getRenderer(rendererName);
    if (!renderer)
    {
        std::cerr << "Рендерер '" << rendererName << "' не найден\n";
        return false;
    }

    if (!renderer->initialize())
    {
        std::cerr << "Ошибка инициализации рендерера: " << rendererName << "\n";
        return false;
    }

    const int w = renderer->getImageWidth();
    const int h = renderer->getImageHeight();

    std::vector<Vec3> framebuffer;
    framebuffer.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));

    const bool success = renderer->render(scene, camera, framebuffer);

    if (success)
        SaveFrameBufferToPNG(framebuffer, w, h, outputFilename);

    renderer->cleanup();
    return success;
}

void RenderManager::saveStats(const std::string &filenamePrefix) const
{
    for (const auto &renderer : renderers_)
    {
        if (!renderer)
            continue;

        const auto &stats = renderer->getStats();
        if (stats.empty())
            continue;

        std::string filename = filenamePrefix + "_" + renderer->getName() + ".json";
        (void)filename;
        // TODO: implement JSON dump if/when needed
    }
}

bool RenderManager::renderFrameTexture(Scene &scene,
                                       Camera &camera,
                                       const std::string &rendererName,
                                       const std::string &outputPath,
                                       TextureRenderMode mode,
                                       int baseSamplesPerPixel)
{
    Renderer *renderer = getRenderer(rendererName);
    if (!renderer)
    {
        std::cerr << "Рендерер '" << rendererName << "' не найден\n";
        return false;
    }

    if (!renderer->initialize())
    {
        std::cerr << "Ошибка инициализации рендерера: " << rendererName << "\n";
        return false;
    }

    const int w = renderer->getImageWidth();
    const int h = renderer->getImageHeight();
    const std::size_t pixelCount = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);

    std::vector<Vec3> framebuffer;
    framebuffer.resize(pixelCount);

    bool success = true;
    bool handled = false;

    // Always reset accumulation once before progressive loop for GPU renderers.
    renderer->resetAccumulation();

#ifdef USE_METAL_RENDERER
    if (auto *metal = dynamic_cast<MetalRenderer *>(renderer))
    {
        const std::vector<int> sppSchedule = BuildMetalSppSchedule(mode, baseSamplesPerPixel);
        const MetalAccumulationMode previousMode = metal->getAccumulationMode();
        metal->setAccumulationMode(mode == TextureRenderMode::Preview
                                       ? MetalAccumulationMode::PreviewProgressive
                                       : MetalAccumulationMode::FinalStill);

        for (std::size_t i = 0; i < sppSchedule.size(); ++i)
        {
            renderer->setSamplesPerPixel(sppSchedule[i]);
            if (!metal->renderTexture(scene, camera, framebuffer))
            {
                success = false;
                break;
            }

            const std::string outPath = outputPath + "_" + std::to_string(i) + ".png";
            SaveFrameBufferToPNG(framebuffer, w, h, outPath);
        }

        renderer->setSamplesPerPixel(baseSamplesPerPixel);
        metal->setAccumulationMode(previousMode);
        handled = true;
    }
#endif

#ifdef USE_HIP_RENDERER
    if (!handled)
    {
        if (auto *hip = dynamic_cast<HIPRenderer *>(renderer))
        {
            if (!hip->renderTexture(scene, camera, framebuffer))
            {
                success = false;
            }
            else
            {
                const std::string outPath = outputPath + "_0.png";
                SaveFrameBufferToPNG(framebuffer, w, h, outPath);
            }
            handled = true;
        }
    }
#endif

#ifdef USE_CUDA_RENDERER
    if (!handled)
    {
        if (auto *cudaRenderer = dynamic_cast<CudaRenderer *>(renderer))
        {
            if (!cudaRenderer->renderTexture(scene, camera, framebuffer))
            {
                success = false;
            }
            else
            {
                const std::string outPath = outputPath + "_0.png";
                SaveFrameBufferToPNG(framebuffer, w, h, outPath);
            }
            handled = true;
        }
    }
#endif

    if (!handled)
    {
        std::cerr << "RenderManager: активный рендерер не поддерживает texture render "
                     "(ни Metal, ни HIP, ни CUDA)\n";
        success = false;
    }

    renderer->cleanup();
    return success;
}
