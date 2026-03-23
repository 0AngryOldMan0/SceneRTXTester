#pragma once

#include "Renderer.h"
#include "Scene.h"
#include "Camera.h"
#include <memory>
#include <vector>
#include <string>

#ifdef USE_HIP_RENDERER
#include "HIPRenderer.h"
#endif
#ifdef USE_CUDA_RENDERER
#include "CUDARenderer.h"
#endif
#ifdef USE_METAL_RENDERER
#include "MetalRenderer.h"
#endif

enum class TextureRenderMode
{
    Progressive = 0,
    Preview = 1
};

class RenderManager
{
public:
    RenderManager();
    ~RenderManager() = default;

    // Управление рендерерами
    void addRenderer(std::unique_ptr<Renderer> renderer);
    void removeRenderer(const std::string &name);
    Renderer *getRenderer(const std::string &name) const;
    const std::vector<std::unique_ptr<Renderer>> &getRenderers() const { return renderers_; }

    // Рендеринг одного кадра
    bool renderFrame(const Scene &scene,
                     const Camera &camera,
                     const std::string &rendererName,
                     const std::string &outputFilename);

    // Сохранение статистики
    void saveStats(const std::string &filenamePrefix = "Results/Info/RenderStats") const;

    // Серия кадров с прогрессивным рендерингом в текстуру (Metal / HIP / CUDA)
    bool renderFrameTexture(Scene &scene,
                            Camera &camera,
                            const std::string &rendererName,
                            const std::string &outputPath,
                            TextureRenderMode mode = TextureRenderMode::Progressive,
                            int baseSamplesPerPixel = 4,
                            bool exportHipDebugViews = false);

private:
    std::vector<std::unique_ptr<Renderer>> renderers_;
};
