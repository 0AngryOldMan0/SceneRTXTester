#include "HIPRenderer.h"
#include "CameraData.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

// Простейший дамп статистики в .txt
static void SaveMonitoringValuesToFileHIP(const std::vector<MonitoringFrameStats> &frames,
                                          const std::string &outputBase = "Results/Info/FrameStats")
{
    if (frames.empty())
        return;

    std::string filename = outputBase + "_HIP_GPU.txt";
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "HIPRenderer: не удалось открыть файл для записи мониторинга: "
                  << filename << "\n";
        return;
    }

    out << std::fixed << std::setprecision(6);
    out << "FrameIndex;RaysTraced;RaysHit;VisitedNodes;FrameTimeMs;"
           "TotalTimeMs;RaysPerSecond\n";

    for (const auto &f : frames)
    {
        out << f.frameIndex << ';'
            << f.raysTraced << ';'
            << f.raysHit << ';'
            << f.totalVisitedNodes << ';'
            << f.frameRenderTimeMs << ';'
            << f.frameTotalTimeMs << ';'
            << f.raysPerSecond << '\n';
    }
}

// ----------------------------------------------------------
// Конструктор / деструктор
// ----------------------------------------------------------

HIPRenderer::HIPRenderer()
{
}

HIPRenderer::~HIPRenderer()
{
    cleanup();
}

// ----------------------------------------------------------
// Инициализация / очистка
// ----------------------------------------------------------

bool HIPRenderer::initialize()
{
    // Пока просто прокидываем в низкоуровневую инициализацию HIP
    return InitHIPRenderer();
}

void HIPRenderer::cleanup()
{
    // Если потом появится явная очистка ресурсов HIP – добавим сюда
}

// ----------------------------------------------------------
// Обычный рендер одного кадра (без накопления)
// ----------------------------------------------------------

bool HIPRenderer::render(const Scene &scene,
                         const Camera &camera,
                         std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "HIPRenderer: ошибка – глобальный BVH не построен\n";
        return false;
    }

    MonitoringFrameStats frameStats;
    frameStats.frameIndex = static_cast<int>(stats_.size());

    auto frameStart = Clock::now();

    // Данные камеры
    CameraDataCPU camData = prepareCameraData(camera, scene.getMainLight().position);

    bool success = RenderFrameHIP(scene.getGlobalNodes(),
                                  scene.getGlobalTriangles(),
                                  scene.getLights(),
                                  scene.getGlobalRootIndex(),
                                  camData,
                                  framebuffer);

    auto frameEnd = Clock::now();

    if (success)
    {
        // Простейшая оценка количества лучей
        frameStats.raysTraced =
            static_cast<std::uint64_t>(imageWidth_) *
            static_cast<std::uint64_t>(imageHeight_) *
            static_cast<std::uint64_t>(samplesPerPixel_);

        frameStats.frameRenderTimeMs =
            std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

        frameStats.frameTotalTimeMs = frameStats.frameRenderTimeMs;

        if (frameStats.frameRenderTimeMs > 0.0)
        {
            frameStats.raysPerSecond =
                static_cast<double>(frameStats.raysTraced) /
                (frameStats.frameRenderTimeMs / 1000.0);
        }

        stats_.push_back(frameStats);
    }

    SaveMonitoringValuesToFileHIP(stats_);
    return success;
}

// ----------------------------------------------------------
// Прогрессивный рендер с накоплением (аналог MetalRenderer::renderTexture)
// ----------------------------------------------------------

bool HIPRenderer::renderTexture(const Scene &scene,
                                const Camera &camera,
                                std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "HIPRenderer: ошибка – глобальный BVH не построен\n";
        return false;
    }

    MonitoringFrameStats frameStats;
    frameStats.frameIndex = static_cast<int>(stats_.size());

    auto frameStart = Clock::now();

    CameraDataCPU camData = prepareCameraData(camera, scene.getMainLight().position);

    bool success = RenderFrameHIPAccumulate(scene.getGlobalNodes(),
                                            scene.getGlobalTriangles(),
                                            scene.getLights(),
                                            scene.getGlobalRootIndex(),
                                            camData,
                                            framebuffer);

    auto frameEnd = Clock::now();

    if (success)
    {
        frameStats.raysTraced =
            static_cast<std::uint64_t>(imageWidth_) *
            static_cast<std::uint64_t>(imageHeight_) *
            static_cast<std::uint64_t>(samplesPerPixel_);

        frameStats.frameRenderTimeMs =
            std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        frameStats.frameTotalTimeMs = frameStats.frameRenderTimeMs;

        if (frameStats.frameRenderTimeMs > 0.0)
        {
            frameStats.raysPerSecond =
                static_cast<double>(frameStats.raysTraced) /
                (frameStats.frameRenderTimeMs / 1000.0);
        }

        stats_.push_back(frameStats);
    }

    SaveMonitoringValuesToFileHIP(stats_);
    return success;
}

// ----------------------------------------------------------
// Сброс накопления
// ----------------------------------------------------------

void HIPRenderer::resetAccumulation()
{
    ResetHIPAccumulation();
}

// ----------------------------------------------------------
// Подготовка данных камеры (такая же, как у MetalRenderer)
// ----------------------------------------------------------

CameraDataCPU HIPRenderer::prepareCameraData(const Camera &camera,
                                             const Vec3 &lightPos) const
{
    CameraDataCPU camData;
    camData.position = camera.getPosition();
    camData.forward = camera.getForward();
    camData.up = camera.getUp();
    camData.right = camera.getRight();
    camData.fovY = camera.getFovY();
    camData.width = imageWidth_;
    camData.height = imageHeight_;
    camData.samplesPerPixel = samplesPerPixel_;
    camData.lightPos = lightPos;
    camData.pad = 0;

    return camData;
}
