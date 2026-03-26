#pragma once

#include "Scene.h"
#include "Camera.h"
#include "MonitoringFrameStats.h"
#include "HitInfo.h"
#include "Light.h"
#include "../Patterns/RenderCommand.h"
#include <string>
#include <vector>

class Renderer
{
public:
    virtual ~Renderer() = default;

    // Основной метод рендеринга
    virtual bool render(const Scene &scene,
                        const Camera &camera,
                        std::vector<Vec3> &framebuffer) = 0;

    // Рендеринг с текстурой (для GPU-рендереров с накоплением)
    virtual bool renderTexture(const Scene &scene,
                               const Camera &camera,
                               std::vector<Vec3> &framebuffer)
    {
        // Default implementation: call render()
        return render(scene, camera, framebuffer);
    }

    // Настройки рендеринга
    virtual void setSamplesPerPixel(int samples) { samplesPerPixel_ = samples; }

    virtual void setImageSize(int width, int height)
    {
        imageWidth_ = width;
        imageHeight_ = height;
    }

    // Управление рендерингом
    virtual bool initialize() { return true; }
    virtual void cleanup() {}

    /**
     * @brief Initialize renderer with RenderCommand parameters
     *
     * This method replaces the need for dynamic_cast and per-renderer configuration.
     * Called after initialize() to apply command-specific settings.
     *
     * Default implementation updates image size and samples from command.
     * Override in subclasses to handle backend-specific parameters (metadata,
     * accumulation modes, debug views, etc).
     *
     * @param command The render command with all parameters
     * @return true if initialization succeeded, false otherwise
     */
    virtual bool initializeWithCommand(const RenderCommand &command)
    {
        // Default implementation: apply common parameters
        setImageSize(command.getImageWidth(), command.getImageHeight());
        setSamplesPerPixel(command.getSamplesPerPixel());
        return true;
    }

    /**
     * @brief Validate that renderer supports the given command
     *
     * Called to check if renderer can execute the command.
     * Override to validate backend-specific requirements.
     *
     * Default implementation always returns true.
     *
     * @param command The render command to validate
     * @return true if command is valid for this renderer, false otherwise
     */
    virtual bool validateCommand(const RenderCommand & /*command*/) const
    {
        return true; // Default: always valid
    }

    // Статистика
    const std::vector<MonitoringFrameStats> &getStats() const { return stats_; } // no copy
    std::vector<MonitoringFrameStats> getStatsCopy() const { return stats_; }    // explicit copy if needed
    virtual void clearStats() { stats_.clear(); }

    // Информация
    virtual std::string getName() const = 0;

    int getImageWidth() const { return imageWidth_; }
    int getImageHeight() const { return imageHeight_; }

    // Прогрессивное накопление (GPU-рендереры)
    virtual void resetAccumulation() = 0;

protected:
    int imageWidth_ = 1920;
    int imageHeight_ = 1080;
    int samplesPerPixel_ = 1;

    std::vector<MonitoringFrameStats> stats_;

    // Вспомогательные методы для наследников
    Vec3 calculateLighting(const HitInfo &hit,
                           const Light &light) const;
};
