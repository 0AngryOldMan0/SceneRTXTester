#pragma once

#include "Renderer.h"
#include "HIPRendererFunctions.h"

struct SceneMetaResources;

class HIPRenderer : public Renderer
{
public:
    HIPRenderer();
    ~HIPRenderer() override;

    bool initialize() override;
    void cleanup() override;

    /**
     * @brief Initialize HIP renderer with RenderCommand
     * Handles HIP-specific parameters: metadata, accumulation mode, debug views
     */
    bool initializeWithCommand(const RenderCommand &command) override;

    /**
     * @brief Validate that HIP renderer can execute the command
     * Currently always returns true; can be extended for HIP-specific validation
     */
    bool validateCommand(const RenderCommand &command) const override;

    bool render(const Scene &scene,
                const Camera &camera,
                std::vector<Vec3> &framebuffer) override;

    std::string getName() const override { return "HIP GPU Ray Tracer"; }

    void setMetaResources(const SceneMetaResources *metaRes);
    void setAccumulationMode(HIPAccumulationMode mode);
    HIPAccumulationMode getAccumulationMode() const { return accumulationMode_; }
    void setDebugView(HIPDebugView view);
    HIPDebugView getDebugView() const { return debugView_; }

    bool preloadSceneResources();
    bool renderTexture(const Scene &scene,
                       const Camera &camera,
                       std::vector<Vec3> &framebuffer);

    void resetAccumulation() override;

    /**
     * @brief Helper: Convert generic RenderCommand::AccumulationMode to HIP-specific enum
     */
    static HIPAccumulationMode commandModeToHIPMode(RenderCommand::AccumulationMode mode)
    {
        return mode == RenderCommand::AccumulationMode::PreviewProgressive
                   ? HIPAccumulationMode::PreviewProgressive
                   : HIPAccumulationMode::FinalStill;
    }

    /**
     * @brief Helper: Convert generic RenderCommand::DebugView to HIP-specific enum
     */
    static HIPDebugView commandViewToHIPView(RenderCommand::DebugView view)
    {
        return static_cast<HIPDebugView>(static_cast<std::uint32_t>(view));
    }

private:
    const SceneMetaResources *m_metaRes = nullptr;
    HIPAccumulationMode accumulationMode_ = HIPAccumulationMode::PreviewProgressive;
    HIPDebugView debugView_ = HIPDebugView::Disabled;

    CameraDataCPU prepareCameraData(const Camera &camera) const;
};
