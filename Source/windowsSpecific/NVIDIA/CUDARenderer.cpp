#include "CudaRenderer.h"
#include "CameraData.h"
#include "CudaRendererFunctions.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

// Простейший дампер статистики в .txt (отдельный файл для CUDA)
static void SaveMonitoringValuesToFileCuda(const std::vector<MonitoringFrameStats> &frames,
                                           const std::string &outputBase = "Results/Info/FrameStats")
{
    if (frames.empty())
        return;

    std::string filename = outputBase + "_CUDA_GPU.txt";
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "CudaRenderer: не удалось открыть файл для записи мониторинга: "
                  << filename << "\n";
        return;
    }

    out << std::fixed << std::setprecision(6);
    out << "FrameIndex;FrameRenderTimeMs;FrameTotalTimeMs;RaysTraced;RaysPerSecond\n";

    for (const auto &f : frames)
    {
        out << f.frameIndex << ';'
            << f.frameRenderTimeMs << ';'
            << f.frameTotalTimeMs << ';'
            << f.raysTraced << ';'
            << f.raysPerSecond << '\n';
    }
}

// ----------------------------------------------------------
// Конструктор / деструктор
// ----------------------------------------------------------

CudaRenderer::CudaRenderer()
{
}

CudaRenderer::~CudaRenderer()
{
    cleanup();
}

// ----------------------------------------------------------
// Инициализация / очистка
// ----------------------------------------------------------

bool CudaRenderer::initialize()
{
    // Хук на будущее: выбор девайса и т.п. Сейчас достаточно того,
    // что делает InitCudaRenderer / CUDA runtime.
    return InitCudaRenderer();
}

void CudaRenderer::cleanup()
{
    // Пока явных ресурсов на стороне CUDA здесь нет
}

// ----------------------------------------------------------
// Обычный рендер одного кадра (без накопления)
// ----------------------------------------------------------

bool CudaRenderer::render(const Scene &scene,
                          const Camera &camera,
                          std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "CudaRenderer: ошибка – глобальный BVH не построен\n";
        return false;
    }

    MonitoringFrameStats frameStats;
    frameStats.frameIndex = static_cast<int>(stats_.size());

    auto frameStart = Clock::now();

    // Подготовка данных камеры
    CameraDataCPU camData = prepareCameraData(camera, scene.getMainLight().position);

    bool success = RenderFrameCuda(scene.getGlobalNodes(),
                                   scene.getGlobalTriangles(),
                                   scene.getLights(),
                                   scene.getGlobalRootIndex(),
                                   camData,
                                   framebuffer);

    auto frameEnd = Clock::now();

    if (success)
    {
        // Примерная оценка количества лучей
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

    SaveMonitoringValuesToFileCuda(stats_);
    return success;
}

// ----------------------------------------------------------
// Прогрессивный рендер с накоплением (многократные вызовы)
// ----------------------------------------------------------

bool CudaRenderer::renderTexture(const Scene &scene,
                                 const Camera &camera,
                                 std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "CudaRenderer::renderTexture: глобальный BVH не построен\n";
        return false;
    }

    MonitoringFrameStats frameStats;
    frameStats.frameIndex = static_cast<int>(stats_.size());

    auto frameStart = Clock::now();

    CameraDataCPU camData = prepareCameraData(camera, scene.getMainLight().position);

    bool success = RenderFrameCudaAccumulate(scene.getGlobalNodes(),
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

    SaveMonitoringValuesToFileCuda(stats_);
    return success;
}

// ----------------------------------------------------------
// Сброс накопительного состояния
// ----------------------------------------------------------

void CudaRenderer::resetAccumulation()
{
    ResetCudaAccumulation();
}

// ----------------------------------------------------------
// Подготовка структуры CameraDataCPU (как в Metal/HIP)
// ----------------------------------------------------------

CameraDataCPU CudaRenderer::prepareCameraData(const Camera &camera,
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