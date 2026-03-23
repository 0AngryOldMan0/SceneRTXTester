#include "HIPRenderer.h"
#include "CameraData.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{
    void SaveMonitoringValuesToFileHIP(const std::vector<MonitoringFrameStats> &frames,
                                       const std::string &outputBase = "Results/Info/FrameStats")
    {
        if (frames.empty())
            return;

        const std::filesystem::path filename = std::filesystem::path(outputBase + "_HIP_GPU.txt");
        const std::filesystem::path parentDir = filename.parent_path();
        if (!parentDir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(parentDir, ec);
            if (ec)
            {
                std::cerr << "HIPRenderer: failed to create stats output directory: "
                          << parentDir.string() << " (" << ec.message() << ")\n";
                return;
            }
        }

        std::ofstream out(filename);
        if (!out)
        {
            std::cerr << "HIPRenderer: failed to open stats output file: "
                      << filename.string() << "\n";
            return;
        }

        out << std::fixed << std::setprecision(6);
        out << "FrameIndex;RaysTraced;RaysHit;VisitedNodes;FrameTimeMs;TotalTimeMs;RaysPerSecond\n";

        for (const MonitoringFrameStats &f : frames)
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
}

HIPRenderer::HIPRenderer() = default;

HIPRenderer::~HIPRenderer()
{
    cleanup();
}

void HIPRenderer::setMetaResources(const SceneMetaResources *metaRes)
{
    if (m_metaRes == metaRes)
        return;

    m_metaRes = metaRes;
    resetAccumulation();
}

void HIPRenderer::setAccumulationMode(HIPAccumulationMode mode)
{
    if (accumulationMode_ == mode)
        return;

    accumulationMode_ = mode;
    resetAccumulation();
}

void HIPRenderer::setDebugView(HIPDebugView view)
{
    if (debugView_ == view)
        return;

    debugView_ = view;
    resetAccumulation();
}

bool HIPRenderer::initialize()
{
    return InitHIPRenderer();
}

bool HIPRenderer::preloadSceneResources()
{
    return PreloadHIPSceneResources(m_metaRes);
}

void HIPRenderer::cleanup()
{
    SaveMonitoringValuesToFileHIP(stats_);
}

bool HIPRenderer::render(const Scene &scene,
                         const Camera &camera,
                         std::vector<Vec3> &framebuffer)
{
    return renderTexture(scene, camera, framebuffer);
}

CameraDataCPU HIPRenderer::prepareCameraData(const Camera &camera) const
{
    CameraDataCPU camData{};
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

void HIPRenderer::resetAccumulation()
{
    ResetHIPAccumulation();
}

bool HIPRenderer::renderTexture(const Scene &scene,
                                const Camera &camera,
                                std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "HIPRenderer: global BVH is not built\n";
        return false;
    }

    framebuffer.resize(static_cast<std::size_t>(imageWidth_) * static_cast<std::size_t>(imageHeight_));

    MonitoringFrameStats frameStats{};
    frameStats.frameIndex = static_cast<int>(stats_.size());

    const auto frameStart = Clock::now();
    const CameraDataCPU camData = prepareCameraData(camera);

    const bool success = RenderFrameHIPTexture(scene.getGlobalNodes(),
                                               scene.getGlobalMeshNodes(),
                                               scene.getGlobalTriangles(),
                                               scene.getGlobalInstances(),
                                               scene.getLights(),
                                               scene.getSceneRevision(),
                                               accumulationMode_,
                                               debugView_,
                                               scene.getGlobalRootIndex(),
                                               camData,
                                               m_metaRes,
                                               framebuffer);

    const auto frameEnd = Clock::now();

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
                static_cast<double>(frameStats.raysTraced) / (frameStats.frameRenderTimeMs / 1000.0);
        }

        stats_.push_back(frameStats);
    }

    return success;
}
