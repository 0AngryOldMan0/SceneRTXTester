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

    bool render(const Scene &scene,
                const Camera &camera,
                std::vector<Vec3> &framebuffer) override;

    std::string getName() const override { return "HIP GPU Ray Tracer"; }

    void setMetaResources(const SceneMetaResources *metaRes);
    void setAccumulationMode(HIPAccumulationMode mode);
    HIPAccumulationMode getAccumulationMode() const { return accumulationMode_; }

    bool preloadSceneResources();
    bool renderTexture(const Scene &scene,
                       const Camera &camera,
                       std::vector<Vec3> &framebuffer);

    void resetAccumulation() override;

private:
    const SceneMetaResources *m_metaRes = nullptr;
    HIPAccumulationMode accumulationMode_ = HIPAccumulationMode::PreviewProgressive;

    CameraDataCPU prepareCameraData(const Camera &camera) const;
};
