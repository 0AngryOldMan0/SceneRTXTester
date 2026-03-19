#pragma once

#include "Renderer.h"
#include "HIPRendererFunctions.h"

// Рендерер на базе HIP (AMD GPU / ROCm)
class HIPRenderer : public Renderer
{
public:
    HIPRenderer();
    ~HIPRenderer();

    bool initialize() override;
    void cleanup() override;

    bool render(const Scene &scene,
                const Camera &camera,
                std::vector<Vec3> &framebuffer) override;

    std::string getName() const override { return "HIP GPU Ray Tracer"; }

    // Прогрессивный рендер с накоплением (аналог MetalRenderer::renderTexture)
    bool renderTexture(const Scene &scene,
                       const Camera &camera,
                       std::vector<Vec3> &framebuffer);

    void resetAccumulation() override;

private:
    CameraDataCPU prepareCameraData(const Camera &camera, const Vec3 &lightPos) const;
};