#pragma once

#include "Scene.h"
#include "Camera.h"
#include "MonitoringFrameStats.h"
#include "HitInfo.h"
#include "Light.h"
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

    // Статистика
    const std::vector<MonitoringFrameStats> &getStats() const { return stats_; } // no copy
    std::vector<MonitoringFrameStats> getStatsCopy() const { return stats_; }   // explicit copy if needed
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
