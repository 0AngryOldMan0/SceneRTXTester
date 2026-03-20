#include "MetalRenderer.h"
#include "CameraData.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace
{
    void SaveMonitoringValuesToFile(const std::vector<MonitoringFrameStats> &frames,
                                    const std::string &outputBase = "Results/Info/FrameStats")
    {
        if (frames.empty())
            return;

        const std::string filename = outputBase + "_GPU.txt";
        std::ofstream out(filename);
        if (!out)
            throw std::runtime_error("Не удалось открыть файл для записи мониторинга: " + filename);

        out << std::fixed << std::setprecision(6);

        std::uint64_t totalRaysTraced = 0;
        std::uint64_t totalRaysHit = 0;
        std::uint64_t totalVisitedNodes = 0;
        double totalRayTimeNs = 0.0;
        double totalRenderTimeMs = 0.0;

        for (const auto &f : frames)
        {
            totalRaysTraced += f.raysTraced;
            totalRaysHit += f.raysHit;
            totalVisitedNodes += f.totalVisitedNodes;
            totalRayTimeNs += f.avgRayTimeNs * (f.raysTraced > 0 ? static_cast<double>(f.raysTraced) : 0.0);
            totalRenderTimeMs += f.frameRenderTimeMs;
        }

        double globalAvgRayTimeNs = 0.0;
        double globalAvgVisitedPerRay = 0.0;
        double globalRaysPerSecond = 0.0;

        if (totalRaysTraced > 0)
        {
            globalAvgRayTimeNs = totalRayTimeNs / static_cast<double>(totalRaysTraced);
            globalAvgVisitedPerRay = static_cast<double>(totalVisitedNodes) / static_cast<double>(totalRaysTraced);

            if (totalRenderTimeMs > 0.0)
                globalRaysPerSecond = static_cast<double>(totalRaysTraced) / (totalRenderTimeMs / 1000.0);
        }

        out << "{\n";
        out << "  \"Global\": {\n";
        out << "    \"TotalFrames\": " << frames.size() << ",\n";
        out << "    \"RaysTraced\": " << totalRaysTraced << ",\n";
        out << "    \"RaysHit\": " << totalRaysHit << ",\n";
        out << "    \"TotalVisitedNodes\": " << totalVisitedNodes << ",\n";
        out << "    \"AvgVisitedNodesPerRay\": " << globalAvgVisitedPerRay << ",\n";
        out << "    \"AvgRayTime_ns\": " << globalAvgRayTimeNs << ",\n";
        out << "    \"RaysPerSecond\": " << globalRaysPerSecond << "\n";
        out << "  },\n";

        out << "  \"Frames\": [\n";
        for (std::size_t i = 0; i < frames.size(); ++i)
        {
            const auto &f = frames[i];
            out << "    {\n";
            out << "      \"FrameIndex\": " << f.frameIndex << ",\n";
            out << "      \"FrameRenderTime_ms\": " << f.frameRenderTimeMs << ",\n";
            out << "      \"FrameSaveTime_ms\": " << f.frameSaveTimeMs << ",\n";
            out << "      \"FrameTotalTime_ms\": " << f.frameTotalTimeMs << "\n";
            out << "    }";
            if (i + 1 != frames.size())
                out << ",";
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }
}

MetalRenderer::MetalRenderer() = default;

MetalRenderer::~MetalRenderer()
{
    cleanup();
}

void MetalRenderer::setMetaResources(const SceneMetaResources *metaRes)
{
    if (m_metaRes == metaRes)
        return;

    m_metaRes = metaRes;
    resetAccumulation();
}

bool MetalRenderer::initialize()
{
    return InitMetalRenderer();
}

void MetalRenderer::cleanup()
{
    // Dump stats once per session (especially important for progressive texture rendering)
    try
    {
        SaveMonitoringValuesToFile(stats_);
    }
    catch (const std::exception &e)
    {
        std::cerr << "MetalRenderer stats dump error: " << e.what() << "\n";
    }
}

static inline std::size_t pixelCount(int w, int h)
{
    return static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
}

bool MetalRenderer::render(const Scene &scene,
                           const Camera &camera,
                           std::vector<Vec3> &framebuffer)
{
    return renderTexture(scene, camera, framebuffer);
}

CameraDataCPU MetalRenderer::prepareCameraData(const Camera &camera, const Vec3 &lightPos) const
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
    (void)lightPos;
    camData.nearPlane = camera.getNearPlane();
    camData.farPlane = camera.getFarPlane();
    camData.focusDistance = camera.getFocusDistance();
    camData.pad = 0.0f;
    return camData;
}

void MetalRenderer::resetAccumulation()
{
    ResetMetalAccumulation();
}

bool MetalRenderer::renderTexture(const Scene &scene,
                                  const Camera &camera,
                                  std::vector<Vec3> &framebuffer)
{
    using Clock = std::chrono::high_resolution_clock;

    if (!scene.hasGlobalBVH())
    {
        std::cerr << "Ошибка: глобальный BVH не построен\n";
        return false;
    }

    framebuffer.resize(pixelCount(imageWidth_, imageHeight_));

    MonitoringFrameStats frameStats;
    frameStats.frameIndex = static_cast<int>(stats_.size());

    const auto frameStart = Clock::now();

    const CameraDataCPU camData = prepareCameraData(camera, scene.getMainLight().position);

    const bool success = RenderFrameMetalTexture(scene.getGlobalNodes(),
                                                 scene.getGlobalMeshNodes(),
                                                 scene.getGlobalTriangles(),
                                                 scene.getGlobalInstances(),
                                                 scene.getLights(),
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

        if (frameStats.frameRenderTimeMs > 0.0)
        {
            frameStats.raysPerSecond =
                static_cast<double>(frameStats.raysTraced) / (frameStats.frameRenderTimeMs / 1000.0);
        }

        stats_.push_back(frameStats);
    }

    return success;
}
