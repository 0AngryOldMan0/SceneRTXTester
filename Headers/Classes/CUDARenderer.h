#pragma once
#include "Renderer.h"
#include "CudaRendererFunctions.h"
#include "Structures/CameraData.h"

class CudaRenderer : public Renderer
{
public:
    CudaRenderer();
    ~CudaRenderer();

    bool initialize() override;
    void cleanup() override;

    /**
     * @brief Initialize CUDA renderer with RenderCommand
     * CUDA currently doesn't use metadata, but this allows future extensibility
     */
    bool initializeWithCommand(const RenderCommand &command) override;

    /**
     * @brief Validate that CUDA renderer can execute the command
     */
    bool validateCommand(const RenderCommand &command) const override;

    bool render(const Scene &scene,
                const Camera &camera,
                std::vector<Vec3> &framebuffer) override;

    std::string getName() const override { return "CUDA GPU Ray Tracer"; }

    // Прогрессивный рендер с накоплением (аналог MetalRenderer::renderTexture)
    bool renderTexture(const Scene &scene,
                       const Camera &camera,
                       std::vector<Vec3> &framebuffer);

    void resetAccumulation() override;

private:
    CameraDataCPU prepareCameraData(const Camera &camera,
                                    const Vec3 &lightPos) const;
};