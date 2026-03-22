#include "CudaRenderer.h"
#include "CameraData.h"
#include "CudaRendererFunctions.h"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

static void SaveMonitoringValuesToFileCuda(const std::vector<MonitoringFrameStats> &frames,
                                           const std::string &outputBase = "Results/Info/FrameStats")
{
    if (frames.empty())
        return;

    const std::filesystem::path filename = std::filesystem::path(outputBase + "_CUDA_GPU.txt");
    const std::filesystem::path parentDir = filename.parent_path();
    if (!parentDir.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parentDir, ec);
        if (ec)
        {
            std::cerr << "CudaRenderer: failed to create stats output directory: "
                      << parentDir.string() << " (" << ec.message() << ")\n";
            return;
        }
    }

    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "CudaRenderer: failed to open monitoring output file: "
                  << filename.string() << "\n";
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

CudaRenderer::CudaRenderer()
{
}

CudaRenderer::~CudaRenderer()
{
    cleanup();
}

bool CudaRenderer::initialize()
{
    return InitCudaRenderer();
}

void CudaRenderer::cleanup()
{
}

bool CudaRenderer::render(const Scene &scene,
                          const Camera &camera,
                          std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "CudaRenderer: global BVH is not built\n";
        return false;
    }

    MonitoringFrameStats frameStats;
    frameStats.frameIndex = static_cast<int>(stats_.size());

    auto frameStart = Clock::now();

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

bool CudaRenderer::renderTexture(const Scene &scene,
                                 const Camera &camera,
                                 std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "CudaRenderer::renderTexture: global BVH is not built\n";
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

void CudaRenderer::resetAccumulation()
{
    ResetCudaAccumulation();
}

CameraDataCPU CudaRenderer::prepareCameraData(const Camera &camera,
                                              const Vec3 &lightPos) const
{
    (void)lightPos;

    CameraDataCPU camData;
    camData.position = camera.getPosition();
    camData.forward = camera.getForward();
    camData.up = camera.getUp();
    camData.right = camera.getRight();

    camData.fovY = camera.getFovY();
    camData.width = imageWidth_;
    camData.height = imageHeight_;
    camData.samplesPerPixel = samplesPerPixel_;
    camData.nearPlane = camera.getNearPlane();
    camData.farPlane = camera.getFarPlane();
    camData.focusDistance = camera.getFocusDistance();
    camData.aspectRatio = camera.getAspectRatio();

    return camData;
}
