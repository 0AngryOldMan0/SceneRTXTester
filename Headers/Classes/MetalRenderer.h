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
    void setDebugView(RenderCommand::DebugView view);
    RenderCommand::DebugView getDebugView() const { return debugView_; }

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

    static std::uint32_t commandViewToMetalDebugMode(RenderCommand::DebugView view)
    {
        switch (view)
        {
            case RenderCommand::DebugView::Disabled:         return 0u;
            case RenderCommand::DebugView::AmbientOcclusion: return 7u;
            case RenderCommand::DebugView::Roughness:        return 8u;
            case RenderCommand::DebugView::Metallic:         return 9u;
            case RenderCommand::DebugView::ShadingNormals:   return 11u;
            case RenderCommand::DebugView::NormalDifference: return 12u;
            case RenderCommand::DebugView::Emissive:         return 13u;
            case RenderCommand::DebugView::BaseColor:        return 5u;
            case RenderCommand::DebugView::VertexColor:      return 17u;
            case RenderCommand::DebugView::MaterialModel:    return 4u;
        }

        return 0u;
    }

private:
    const SceneMetaResources *m_metaRes = nullptr;
    MetalAccumulationMode accumulationMode_ = MetalAccumulationMode::PreviewProgressive;
    RenderCommand::DebugView debugView_ = RenderCommand::DebugView::Disabled;

    CameraDataCPU prepareCameraData(const Camera &camera) const;
};
