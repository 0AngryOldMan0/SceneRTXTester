#pragma once

#include "Renderer.h"
#include "MetalRendererFunctions.h"

struct SceneMetaResources;

class MetalRenderer : public Renderer
{
public:
    MetalRenderer();
    ~MetalRenderer() override;

    bool initialize() override;
    void cleanup() override;

    /**
     * @brief Initialize Metal renderer with RenderCommand
     * Handles Metal-specific parameters: metadata, accumulation mode
     */
    bool initializeWithCommand(const RenderCommand &command) override;

    /**
     * @brief Validate that Metal renderer can execute the command
     */
    bool validateCommand(const RenderCommand &command) const override;

    bool render(const Scene &scene,
                const Camera &camera,
                std::vector<Vec3> &framebuffer) override;

    std::string getName() const override { return "Metal GPU Ray Tracer"; }

    void setMetaResources(const SceneMetaResources *metaRes);
    void setAccumulationMode(MetalAccumulationMode mode);
    MetalAccumulationMode getAccumulationMode() const { return accumulationMode_; }

    bool renderTexture(const Scene &scene,
                       const Camera &camera,
                       std::vector<Vec3> &framebuffer);

    bool preloadSceneResources();
    void resetAccumulation() override;

    /**
     * @brief Helper: Convert generic RenderCommand::AccumulationMode to Metal-specific enum
     */
    static MetalAccumulationMode commandModeToMetalMode(RenderCommand::AccumulationMode mode)
    {
        return mode == RenderCommand::AccumulationMode::PreviewProgressive
                   ? MetalAccumulationMode::PreviewProgressive
                   : MetalAccumulationMode::FinalStill;
    }

private:
    const SceneMetaResources *m_metaRes = nullptr;
    MetalAccumulationMode accumulationMode_ = MetalAccumulationMode::PreviewProgressive;

    CameraDataCPU prepareCameraData(const Camera &camera) const;
};
