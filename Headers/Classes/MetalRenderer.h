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

private:
    const SceneMetaResources *m_metaRes = nullptr;
    MetalAccumulationMode accumulationMode_ = MetalAccumulationMode::PreviewProgressive;

    CameraDataCPU prepareCameraData(const Camera &camera) const;
};
